import gym 
import copy
import gym.wrappers
import numpy as np
import time
from tqdm import tqdm
from procgen import ProcgenGym3Env
from collections import deque

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image
import random

import wandb

import utils
from ppo import PPO
from data import TransitionsDataset
from utils import seed_everything
from recorder_wrapper import RecorderWrapper


# TODO: riordina il codice in più file )main è molto lungo!), magari alla fine dopo aver testato tutto su kaggle!


def compute_advantages(values, rewards, gamma, lambda_):
    # GAE estimator
    deltas = np.array(rewards) + gamma * np.array(values[1:]) - np.array(values[:-1])
    advantages = [deltas[-1]] 

    for t in range(len(deltas)-2, -1, -1):
        advantage_t = deltas[t] + gamma * lambda_ * advantages[-1]
        advantages.append(advantage_t)

    advantages = advantages[::-1]
    return advantages

def compute_value_targets(advantages, values, rewards, config):
    value_targets = []
    if config.v_target == "TD-lambda":
        for t in range(len(advantages)):
            value_targets.append(advantages[t] + values[t])
    elif config.v_target == "MC":
        value_targets.append(rewards[-1])
        for t in range(len(rewards)-2, -1, -1):
            value_targets.append(rewards[t] + config.gamma * value_targets[-1])
        value_targets = value_targets[::-1]
    else:
        raise ValueError(f"Unknown value target type {config.v_target}, choose between 'TD-lambda' and 'MC'.")
    return value_targets

def play(env, policy, num_steps, config):
    transitions = []

    obs, _ = env.reset()

    # stack frames together to introduce temporal information
    state_deque = deque()
    for _ in range(config.stack_size):
        state_deque.append(state_processing(obs))

    state = torch.concatenate(list(state_deque), axis=0)

    trajectory = {
        'states': [state],
        'actions': [],
        'rewards': [],
        'values': [],
    }

    policy.eval()

    for t in tqdm(range(num_steps)):
        assert not policy.policy_net.training and not policy.value_net.training, "Policy should be in evaluation mode here"

        state = state.unsqueeze(0).to(device)
        action, value = policy.act(state)

        next_obs, reward, terminated, truncated, info = env.step(action)
        truncated = truncated or t == num_steps - 1

        # update step count
        utils.global_step += 1

        # collect transition info in trajectory
        trajectory['values'].append(value)

        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)

        # udpate state to become next state using the new observation
        state_deque.popleft()
        state_deque.append(state_processing(next_obs))
        state = torch.concatenate(list(state_deque), axis=0)

        trajectory['states'].append(state)


        if terminated or truncated:
            # see terminated vs truncated API at https://farama.org/Gymnasium-Terminated-Truncated-Step-API
            if terminated:
                # final value is 0 if the episode terminated, i.e. reached a final state
                trajectory['values'].append(0)
            else:
                # bootstrap if the episode was truncated, i.e. didn't reach a final state
                state = state.unsqueeze(0).to(device)
                _, value = policy.act(state)
                trajectory['values'].append(value)
            
            assert len(trajectory['states']) >= 2, "Trajectory must have at least 2 states to compute advantages."
            assert len(trajectory['states']) == len(trajectory['actions']) + 1 , "Trajectory must have one more state than actions."
            advantages = compute_advantages(trajectory['values'], trajectory['rewards'], config.gamma, config.lambda_)

            value_targets = compute_value_targets(advantages, trajectory['values'], trajectory['rewards'], config)

            if config.normalize_v_targets:
                policy.update_v_target_stats(np.array(value_targets))


            # convert trajectory into list of transitions
            for t in range(len(trajectory['states'])-1):    # -1 because advantages already encode the value of state t+1
                transitions.append({
                    's_t': trajectory['states'][t],
                    'a_t': trajectory['actions'][t],
                    'A_t': advantages[t],
                    'v_target_t': value_targets[t],
                })

            # log and update episodes count only if episode terminated
            if terminated:
                wandb.log({"play/episodic_reward": sum(trajectory['rewards']), 
                        "play/episode_length": len(trajectory['states'])-1,
                        "play/step": utils.global_step})
            
            if t < num_steps - 1:
                # reset env and trajectory
                obs, _ = env.reset()

                state_deque = deque()
                for _ in range(config.stack_size):
                    state_deque.append(state_processing(obs))

                state = torch.concatenate(list(state_deque), axis=0)

                trajectory = {
                    'states': [state],
                    'actions': [],
                    'rewards': [],
                    'values': [],
                }

    return transitions


def train(policy, policy_old, train_dataloader, optimizer_policy, optimizer_value, config, scheduler_policy=None, scheduler_value=None):

    policy.train()
    policy_old.eval()
    assert policy_old.policy_net.training == False and policy_old.value_net.training == False, "Old policy should be in evaluation mode here"
    assert policy.policy_net.training == True and policy.value_net.training == True, "Policy should be in training mode here"
    for epoch in tqdm(range(config.epochs)):
        for batch, (states, actions, advantages, value_targets) in enumerate(train_dataloader):
            # normalize advantages between 0 and 1
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = states.to(device)
            actions = actions.to(device)
            advantages = advantages.to(device)
            value_targets = value_targets.to(device)
            
            dists, values = policy.actions_dist_and_v(states)
            old_dists, _ = policy_old.actions_dist_and_v(states)

            log_probs = dists.log_prob(actions)
            old_log_probs = old_dists.log_prob(actions)

            # Equivalent of doing exp(log_probs) / exp(old_log_probs) 
            # but avoids overflows and division by (potentially if underflown) zero, breaking loss function
            ratios = torch.exp(log_probs - old_log_probs)

            # TODO: remove
            if torch.isnan(ratios).any() or torch.isinf(ratios).any():
                print("Ratios are nan or inf! ")
                print(ratios)
                print(log_probs)
                print(old_log_probs)
                raise ValueError("Ratios are nan or inf! ")
                
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print("Advantages are nan or inf! ")
                print(f"Advantages: {advantages}")
                print(f"Value targets: {value_targets}")
                print(f"Actions: {actions}")
                print(f"Len states: {len(states)}")
                raise ValueError("Advantages are nan or inf! ")

            # clipped surrogate loss
            l_clips = -torch.min(ratios * advantages, torch.clip(ratios, 1-config.eps_clip, 1+config.eps_clip) * advantages)
            loss_pi = torch.mean(l_clips)
            loss_entropy = dists.entropy().mean()
            loss_policy = loss_pi - config.entropy_bonus * loss_entropy

            # mse loss
            loss_value = torch.nn.functional.mse_loss(values, value_targets)

            # with two different optimizers
            loss_policy.backward()
            optimizer_policy.step()
            optimizer_policy.zero_grad()

            loss_value.backward()
            optimizer_value.step()
            optimizer_value.zero_grad()

            if utils.global_batch % config.log_frequency == 0:
                wandb.log({"train/loss_pi": loss_pi, 
                           "train/loss_v": loss_value,
                           "train/entropy": loss_entropy,
                           "train/lr_policy": optimizer_policy.param_groups[0]['lr'],
                           "train/lr_value": optimizer_value.param_groups[0]['lr'],
                           "train/batch": utils.global_batch})
            
            utils.global_batch += 1
        
        if scheduler_policy is not None:
            scheduler_policy.step()
        if scheduler_value is not None:
            scheduler_value.step()

        with torch.no_grad():
            # KL divergence between old and new policy for early stopping
            kl_div = torch.distributions.kl.kl_divergence(dists, old_dists).mean().item()
            wandb.log({"train/kl_div": kl_div, "train/batch": utils.global_batch})
            if kl_div > config.kl_limit:
                print(f"Early stopping at epoch {epoch} due to KL divergence {round(kl_div, 4)} > {config.kl_limit}")
                break

def test(env, policy, num_steps, config):
    obs, _ = env.reset()

    # stack frames together to introduce temporal information
    state_deque = deque()
    for _ in range(config.stack_size):
        state_deque.append(state_processing(obs))
    state = torch.concatenate(list(state_deque), axis=0)

    policy.eval()
    assert not policy.policy_net.training and not policy.value_net.training, "Policy should be in evaluation mode here"
    
    episode_steps = 0
    cum_reward = 0

    for step in tqdm(range(num_steps)):

        state = state.unsqueeze(0).to(device)
        action, _ = policy.act(state)

        next_obs, reward, terminated, truncated, info = env.step(action)

        episode_steps += 1
        cum_reward += reward

        # udpate state to become next state using the new observation
        state_deque.popleft()
        state_deque.append(state_processing(next_obs))
        state = torch.concatenate(list(state_deque), axis=0)

        if terminated or truncated:
            wandb.log({"test/episodic_reward": cum_reward, 
                    "test/episode_length": episode_steps,
                    "test/step": step})
            
            episode_steps = 0
            cum_reward = 0
                
            if step < num_steps - 1:
                # reset env and initial obs
                obs, _ = env.reset()

                state_deque = deque()
                for _ in range(config.stack_size):
                    state_deque.append(state_processing(obs))
                state = torch.concatenate(list(state_deque), axis=0)


### CONFIGURATION ###
TOT_TIMESTEPS = int(2**18)  #int(2**20)  # approx 1M
ITER_TIMESTEPS = 1024
NUM_ITERATIONS = TOT_TIMESTEPS // ITER_TIMESTEPS
CONFIG = {
    # Game
    "game": "coinrun",
    "num_levels": 200,
    "seed": 6,
    "difficulty": "easy",
    "backgrounds": False,
    "stack_size": 4,

    # Timesteps and iterations
    "tot_timesteps": TOT_TIMESTEPS,
    "iteration_timesteps": ITER_TIMESTEPS,
    "num_iterations": NUM_ITERATIONS,

    # Network architecture
    "batch_norm": True,

    # Training params
    "epochs": 3,
    "batch_size": 64,
    "lr_policy_network": 5e-4,
    "lr_value_network": 5e-4,
    "kl_limit": 0.015,

    # PPO params
    "gamma": 0.999,
    "lambda_": 0.95,
    "eps_clip": 0.2,
    "entropy_bonus": 0.01,
    "v_target": "TD-lambda",  # "TD-lambda" (for advantage + value) or "MC" (for cumulative reward)
    "normalize_v_targets": True,

    # Logging
    "log_frequency": 5,
    "log_video": True,
    "episode_video_frequency": 5,
}


### WANDB ###
wandb.login()
wandb.init(project="ppo-procgen", name=f"{CONFIG['game']}_{CONFIG['num_levels']}_{CONFIG['difficulty']}", config=CONFIG)
config = wandb.config

wandb.define_metric("play/step")
wandb.define_metric("train/batch")
wandb.define_metric("test/step")

wandb.define_metric("play/episodic_reward", step_metric="play/step")
wandb.define_metric("play/episode_length", step_metric="play/step")
wandb.define_metric("train/loss_pi", step_metric="train/batch")
wandb.define_metric("train/loss_v", step_metric="train/batch")
wandb.define_metric("train/entropy", step_metric="train/batch")
wandb.define_metric("train/lr_policy", step_metric="train/batch")
wandb.define_metric("train/lr_value", step_metric="train/batch")
wandb.define_metric("test/episodic_reward", step_metric="test/step")
wandb.define_metric("test/episode_length", step_metric="test/step")


### CREATE ENVIRONMENT ###
env = gym.make(
    f"procgen:procgen-{config.game}-v0",
    num_levels=config.num_levels,
    start_level=config.seed,
    distribution_mode=config.difficulty,
    use_backgrounds=config.backgrounds,
    render_mode='rgb_array',
    apply_api_compatibility=True,
    rand_seed=config.seed
)

if config.log_video:
    env = RecorderWrapper(env, config.episode_video_frequency)

seed_everything(config.seed)

### CREATE PPO AGENTS ###
policy = PPO(env, config)
policy_old = copy.deepcopy(policy)

print(f"Model has {sum(p.numel() for p in policy.policy_net.parameters()) + sum(p.numel() for p in policy.value_net.parameters())} total parameters.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

policy.to(device)
policy_old.to(device)

optimizer_policy = torch.optim.Adam(policy.policy_net.parameters(), lr=config.lr_policy_network)
scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=config.num_iterations*config.epochs, eta_min=1e-6)

optimizer_value = torch.optim.Adam(policy.value_net.parameters(), lr=config.lr_value_network)
scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_value, T_max=config.num_iterations*config.epochs, eta_min=1e-6)

state_processing = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


### MAIN LOOP ###
for iteration in range(NUM_ITERATIONS):
    print(f"===============Iteration {iteration+1}===============")
    transitions = play(env, policy, ITER_TIMESTEPS, config)

    if config.normalize_v_targets:
        dataset = TransitionsDataset(transitions, normalize_v_targets=True, v_mu=policy.value_mean, v_std=policy.value_std)
    else:
        dataset = TransitionsDataset(transitions)
    train_dataloader = DataLoader(dataset, 
                                  batch_size=config.batch_size, 
                                  shuffle=True)

    print(f"Collected {len(transitions)} transitions, starting training...")

    # update policy
    train(policy, policy_old, train_dataloader, optimizer_policy, optimizer_value, config, scheduler_policy, scheduler_value)
    print("Training done!")

    # TODO: double check this below! For speed and correctness
    del policy_old
    policy_old = copy.deepcopy(policy)  # TODO: just copy the parameters? It should work like this tho
    policy_old.to(device)

### TEST LOOP ###
env_test = gym.make(
    f"procgen:procgen-{config.game}-v0",
    num_levels=0,
    start_level=config.seed,
    distribution_mode=config.difficulty,
    use_backgrounds=config.backgrounds,
    render_mode='rgb_array',
    apply_api_compatibility=True,
    rand_seed=config.seed
)

test(env_test, policy, TOT_TIMESTEPS, config)


wandb.finish()