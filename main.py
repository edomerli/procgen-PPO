import gym 
import gym.wrappers
import copy
import os

import torch
import wandb

from ppo import PPO
from utils import seed_everything
from recorder_wrapper import RecorderWrapper
from play import play_and_train, test

# argparse missing since will train on kaggle/colab most probably

### CONFIGURATION ###
TOT_TIMESTEPS = int(2**20)  # approx 1M
ITER_TIMESTEPS = 1024
NUM_ITERATIONS = TOT_TIMESTEPS // ITER_TIMESTEPS
DIFFICULTY = "hard"
CONFIG = {
    # Game
    "game": "coinrun",
    "num_levels": 200 if DIFFICULTY == "easy" else 500,
    "seed": 6,
    "difficulty": DIFFICULTY,
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
    "log_video": False,
    "episode_video_frequency": 100,
}


### WANDB ###
wandb.login(key="14a7d0e7554bbddd13ca1a8d45472f7a95e73ca4")
wandb.init(project="ppo-procgen", name=f"{CONFIG['game']}_{CONFIG['num_levels']}_{CONFIG['difficulty']}", config=CONFIG)
config = wandb.config

wandb.define_metric("play/step")
wandb.define_metric("train/batch")

wandb.define_metric("play/episodic_reward", step_metric="play/step")
wandb.define_metric("play/episode_length", step_metric="play/step")
wandb.define_metric("train/loss_pi", step_metric="train/batch")
wandb.define_metric("train/loss_v", step_metric="train/batch")
wandb.define_metric("train/entropy", step_metric="train/batch")
wandb.define_metric("train/lr_policy", step_metric="train/batch")
wandb.define_metric("train/lr_value", step_metric="train/batch")
wandb.define_metric("test/episodic_reward", step_metric="play/step")
wandb.define_metric("test/episode_length", step_metric="play/step")


### CREATE ENVIRONMENTS ###
env_train = gym.make(
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
    env_train = RecorderWrapper(env_train, config.episode_video_frequency)

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

seed_everything(config.seed)

### CREATE PPO AGENTS AND OPTIMIZERS ###
policy = PPO(env_train, config)
policy_old = copy.deepcopy(policy)

print(f"Policy network has {sum(p.numel() for p in policy.policy_net.parameters())} parameters.")
print(f"Value network has {sum(p.numel() for p in policy.value_net.parameters())} parameters.")
print(f"Total parameters: {sum(p.numel() for p in policy.policy_net.parameters()) + sum(p.numel() for p in policy.value_net.parameters())}.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

policy.to(device)
policy_old.to(device)

optimizer_policy = torch.optim.Adam(policy.policy_net.parameters(), lr=config.lr_policy_network)
# scheduler_policy = torch.optim.lr_scheduler.OneCycleLR(optimizer_policy, max_lr=config.lr_policy_network, total_steps=config.num_iterations*config.epochs, pct_start=0.1)
scheduler_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=config.num_iterations*config.epochs, eta_min=1e-6)

optimizer_value = torch.optim.Adam(policy.value_net.parameters(), lr=config.lr_value_network)
# scheduler_value = torch.optim.lr_scheduler.OneCycleLR(optimizer_value, max_lr=config.lr_value_network, total_steps=config.num_iterations*config.epochs, pct_start=0.1)
scheduler_value = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_value, T_max=config.num_iterations*config.epochs, eta_min=1e-6)

### MAIN ###
play_and_train(env_train, env_test, policy, policy_old, optimizer_policy, optimizer_value, device, config, scheduler_policy=scheduler_policy, scheduler_value=scheduler_value)

### SAVE MODEL ###
if not os.path.exists("models"):
    os.makedirs("models")

print("Saving best model to wandb...")
save_path = f"models/{config.game}_{config.difficulty}.pt"
torch.save(policy.state_dict(), save_path)
# upload to wandb
artifact = wandb.Artifact(f"model_{config.game}_{config.difficulty}", type='model')
artifact.add_file(save_path)
wandb.log_artifact(artifact)
print("Saved successfully!")


wandb.finish()