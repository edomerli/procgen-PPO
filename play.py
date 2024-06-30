import numpy as np
import torch
from collections import deque
from tqdm import tqdm
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

import utils
from train import train
from data import TransitionsDataset

frame_to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

def compute_advantages(values, rewards, gamma, lambda_):
    """Compute advantages using Generalized Advantage Estimation (GAE).

    Args:
        values (list[float]): the list of value estimates for each state
        rewards (list[float]): the list of rewards for each state
        gamma (float): discount factor
        lambda_ (float): GAE parameter

    Returns:
        list[float]: the list of advantages for each state
    """
    assert len(values) >= 2, "Values should have at least 2 elements."
    assert len(values) == len(rewards) + 1, "Values and rewards should have the same length, with values having one more element."
    # GAE estimator
    deltas = np.array(rewards) + gamma * np.array(values[1:]) - np.array(values[:-1])
    advantages = [deltas[-1]] 

    for t in range(len(deltas)-2, -1, -1):
        advantage_t = deltas[t] + gamma * lambda_ * advantages[-1]
        advantages.append(advantage_t)

    advantages = advantages[::-1]
    return advantages

def compute_value_targets(advantages, values, rewards, config):
    """Compute value targets for training the value network.

    Args:
        advantages (list[float]): the list of advantages for each state
        values (list[float]): the list of value estimates for each state
        rewards (list[float]): the list of rewards for each state
        config (dict): the configuration dictionary

    Raises:
        ValueError: if the value target type is not in ["TD-lambda", "MC"]

    Returns:
        list[float]: the list of value targets for each state
    """
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


def play_and_train(env, env_test, policy, policy_old, optimizer_policy, optimizer_value, device, config, **kwargs):  
    """Play the game to generate transitions and train the policy on those. Interleaves also testing of the policy.

    Args:
        env (gym.Env): the training environment
        env_test (gym.Env): the testing environment
        policy (PPO): the policy to train
        policy_old (PPO): a copy of the policy to use for training
        optimizer_policy (torch.optim.Optimizer): optimizer for the policy network
        optimizer_value (torch.optim.Optimizer): optimizer for the value network
        device (torch.device): the device to run the computations on
        config (dict): the configuration dictionary
    """

    for iteration in range(config.num_iterations):
        print(f"===============Iteration {iteration+1}===============")
        print(f"Playing {config.iteration_timesteps} steps...")

        transitions = []

        obs, _ = env.reset()

        # stack frames together to introduce temporal information
        state_deque = deque()
        for _ in range(config.stack_size):
            state_deque.append(frame_to_tensor(obs))

        state = torch.concatenate(list(state_deque), axis=0)

        # initialize trajectory
        trajectory = {
            'states': [state],
            'actions': [],
            'rewards': [],
            'values': [],
        }

        policy.eval()

        for step in tqdm(range(config.iteration_timesteps)):
            assert not policy.policy_net.training and not policy.value_net.training, "Policy should be in evaluation mode here"

            state = state.unsqueeze(0).to(device)
            action, value = policy.act_and_v(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            truncated = truncated or step == config.iteration_timesteps - 1

            # update step count
            utils.global_step += 1

            # collect transition info in trajectory
            trajectory['values'].append(value)

            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)

            # udpate state to become next state using the new observation
            state_deque.popleft()
            state_deque.append(frame_to_tensor(next_obs))
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
                    value = policy.value(state)
                    trajectory['values'].append(value)
                
                # compute advantages and value targets
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
                
                # if iteration is not over, reset env and trajectory
                if step < config.iteration_timesteps - 1:
                    obs, _ = env.reset()

                    state_deque = deque()
                    for _ in range(config.stack_size):
                        state_deque.append(frame_to_tensor(obs))

                    state = torch.concatenate(list(state_deque), axis=0)

                    trajectory = {
                        'states': [state],
                        'actions': [],
                        'rewards': [],
                        'values': [],
                    }
        # end of play loop

        # create dataset and dataloader for training 
        if config.normalize_v_targets:
            dataset = TransitionsDataset(transitions, normalize_v_targets=True, v_mu=policy.value_mean, v_std=policy.value_std)
        else:
            dataset = TransitionsDataset(transitions)
        train_dataloader = DataLoader(dataset, 
                                    batch_size=config.batch_size, 
                                    shuffle=True)

        print(f"Collected {len(transitions)} transitions, starting training...")

        ### TRAIN LOOP ###
        train(policy, policy_old, train_dataloader, optimizer_policy, optimizer_value, device, config, **kwargs)
        print("Training done!")

        del policy_old
        policy_old = copy.deepcopy(policy)
        policy_old.to(device)

        ### TEST LOOP ###
        print(f"Now testing policy...")
        test(env_test, policy, device, config)


def test(env_test, policy, device, config):
    """Test the policy on the test environment.

    Args:
        env_test (gym.Env): the testing environment
        policy (PPO): the policy to test
        device (torch.device): the device to run the computations on
        config (dict): the configuration dictionary
    """
    obs, _ = env_test.reset()

    # stack frames together to introduce temporal information
    state_deque = deque()
    for _ in range(config.stack_size):
        state_deque.append(frame_to_tensor(obs))
    state = torch.concatenate(list(state_deque), axis=0)

    policy.eval()
    assert not policy.policy_net.training and not policy.value_net.training, "Policy should be in evaluation mode here"
    
    # reset global step to avoid influencing the training loop step count
    utils.global_step -= config.iteration_timesteps
    episode_steps = 0
    cum_reward = 0

    for step in tqdm(range(config.iteration_timesteps)):

        state = state.unsqueeze(0).to(device)
        action = policy.act(state)

        next_obs, reward, terminated, truncated, info = env_test.step(action)

        utils.global_step += 1
        episode_steps += 1
        cum_reward += reward

        # udpate state to become next state using the new observation
        state_deque.popleft()
        state_deque.append(frame_to_tensor(next_obs))
        state = torch.concatenate(list(state_deque), axis=0)

        if terminated or truncated:
            wandb.log({"test/episodic_reward": cum_reward, 
                    "test/episode_length": episode_steps,
                    "play/step": utils.global_step})
            
            episode_steps = 0
            cum_reward = 0
                
            if step < config.iteration_timesteps - 1:
                # reset env and initial obs
                obs, _ = env_test.reset()

                state_deque = deque()
                for _ in range(config.stack_size):
                    state_deque.append(frame_to_tensor(obs))
                state = torch.concatenate(list(state_deque), axis=0)