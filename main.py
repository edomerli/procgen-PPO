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

from ppo import PPO
from data import TransitionsDataset
from utils import seed_everything


# TODO: riordina il codice in più file )main è molto lungo!), magari alla fine dopo aver testato tutto su kaggle!

global_batch = 0
global_episode = 0
global_step = 0


def compute_advantages(values, rewards, gamma, lambda_):
    # GAE estimator
    deltas = np.array(rewards) + gamma * np.array(values[1:]) - np.array(values[:-1])
    advantages = [deltas[-1]] 

    for t in range(len(deltas)-2, -1, -1):
        advantage_t = deltas[t] + gamma * lambda_ * advantages[-1]
        advantages.append(advantage_t)

    advantages = advantages[::-1]
    return advantages

def compute_value_targets(advantages, values):
    value_targets = []
    for t in range(len(advantages)):
        value_targets.append(advantages[t] + values[t])
    return value_targets

def play(env, policy, num_steps, config):
    transitions = []
    global global_episode
    global global_step

    obs, _ = env.reset()

    # stack frames together to introduce temporal information
    state_deque = deque()
    for _ in range(config.stack_size):
        state_deque.append(data_processing(obs))

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
        global_step += 1

        # collect transition info in trajectory
        trajectory['values'].append(value)

        trajectory['actions'].append(action)
        trajectory['rewards'].append(reward)

        # udpate state to become next state using the new observation
        state_deque.popleft()
        state_deque.append(data_processing(next_obs))
        state = torch.concatenate(list(state_deque), axis=0)

        trajectory['states'].append(state)


        if terminated or truncated:
            # see terminated vs truncated API at https://farama.org/Gymnasium-Terminated-Truncated-Step-API
            if terminated:
                # TODO: check that the final frame is not reset but it's when he reaches the coin
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

            value_targets = compute_value_targets(advantages, trajectory['values'])

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
                        "play/step": global_step})
                global_episode += 1

            # video logging
            if config.log_video and global_episode % config.video_log_frequency == 0:
                pass    # TODO: implement
            
            # reset env and trajectory
            obs, _ = env.reset()

            state_deque = deque()
            for _ in range(config.stack_size):
                state_deque.append(data_processing(obs))

            state = torch.concatenate(list(state_deque), axis=0)

            trajectory = {
                'states': [state],
                'actions': [],
                'rewards': [],
                'values': [],
            }

    return transitions


def train(policy, policy_old, train_dataloader, optimizer_policy, optimizer_value, config, scheduler_policy=None, scheduler_value=None):

    global global_batch

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

            if global_batch % config.log_frequency == 0:
                wandb.log({"loss/loss_pi": loss_pi, 
                           "loss/loss_v": loss_value,
                           "loss/entropy": loss_entropy,
                           "loss/lr_policy": optimizer_policy.param_groups[0]['lr'],
                           "loss/lr_value": optimizer_value.param_groups[0]['lr'],
                           "loss/batch": global_batch})
            
            global_batch += 1
        
        if scheduler_policy is not None:
            scheduler_policy.step()
        if scheduler_value is not None:
            scheduler_value.step()

### CONFIGURATION ###
TOT_TIMESTEPS = int(2**17)  #int(2**20)  # approx 1M
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

    # PPO params
    "gamma": 0.999,
    "lambda_": 0.95,
    "eps_clip": 0.2,
    "entropy_bonus": 0.01,
    "normalize_v_targets": True,

    # Logging
    "log_frequency": 5,
    "log_video": False,
    "video_log_frequency": 10,
}


### WANDB ###
wandb.login()
wandb.init(project="ppo-procgen", name=f"{CONFIG['game']}_{CONFIG['num_levels']}_{CONFIG['difficulty']}", config=CONFIG)
config = wandb.config

wandb.define_metric("play/step")
wandb.define_metric("loss/batch")

wandb.define_metric("play/episodic_reward", step_metric="play/step")
wandb.define_metric("play/episode_length", step_metric="play/step")
wandb.define_metric("loss/loss_pi", step_metric="loss/batch")
wandb.define_metric("loss/loss_v", step_metric="loss/batch")
wandb.define_metric("loss/entropy", step_metric="loss/batch")
wandb.define_metric("loss/lr_policy", step_metric="loss/batch")
wandb.define_metric("loss/lr_value", step_metric="loss/batch")


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

seed_everything(config.seed)#, env)

### CREATE PPO AGENTS ###
policy = PPO(env, config)
policy_old = copy.deepcopy(policy)

print(f"Model has {sum(p.numel() for p in policy.policy_net.parameters()) + sum(p.numel() for p in policy.value_net.parameters())} total parameters.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

policy.to(device)
policy_old.to(device)

optimizer_policy = torch.optim.Adam(policy.policy_net.parameters(), lr=config.lr_policy_network)
scheduler_policy = torch.optim.lr_scheduler.StepLR(optimizer_policy, step_size=600, gamma=0.9)

optimizer_value = torch.optim.Adam(policy.value_net.parameters(), lr=config.lr_value_network)
scheduler_value = torch.optim.lr_scheduler.StepLR(optimizer_value, step_size=600, gamma=0.9)

data_processing = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


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


wandb.finish()