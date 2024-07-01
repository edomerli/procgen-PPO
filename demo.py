import gym
import argparse
import wandb
import os
import torch
from collections import deque
import torchvision.transforms as transforms

from ppo import PPO


"""Script to launch the demo of a trained agent on a selected game and difficulty level.
"""


supported_games = ["coinrun", "bossfight"]   # TODO: add more games
supported_difficulties = ["easy", "hard"]


# parse arguments
parser = argparse.ArgumentParser("Script to launch the demo of a trained agent on a selected game and difficulty level")
game_arg = parser.add_argument("--game", type=str, help=f"Game to play. Supported games are: {supported_games}")
diff_arg = parser.add_argument("--difficulty", type=str, help=f"Difficulty level of the game. Possible choices are: {supported_difficulties}")
parser.add_argument('--stack_size', default=4, type=int, help=f"Number of frames to stack together. Default is 4.")
parser.add_argument('--batch_norm', default=False, type=lambda x: (str(x).lower() == 'true'), help=f"Whether to use batch normalization in the network or not. Default is False.")
parser.add_argument('--normalize_v_targets', default=True, type=lambda x: (str(x).lower() == 'true'), help=f"Whether to normalize the value targets. Default is True.")


args = parser.parse_args()

if args.game is None:
    raise argparse.ArgumentError(game_arg, "Please specify the game to play")

if args.difficulty is None:
    raise argparse.ArgumentError(diff_arg, "Please specify the difficulty level of the game")

if args.game not in supported_games:
    raise argparse.ArgumentError(game_arg, f"Game {args.game} is not supported. Supported games are: {supported_games}")

if args.difficulty not in supported_difficulties:
    raise argparse.ArgumentError(diff_arg, f"Difficulty level {args.difficulty} is not supported. Possible choices are: {supported_difficulties}")

# download the model from wandb if needed
model_path = f"models/{args.game}_{args.difficulty}.pt"

if os.path.exists(model_path):
    print(f"Model already exists at {model_path}. Skipping download...")

else:
    print(f"Model does not exist at {model_path}. Downloading it from wandb...")
    if not os.path.exists("models"):
        os.makedirs("models")

    wandb.init(project="ppo-procgen", name=f"{args.game}_demo")
    artifact = wandb.use_artifact(f"model_{args.game}_{args.difficulty}:latest", type="model")
    artifact.download("models")
    wandb.finish()


# create the environment
env = gym.make(
    f"procgen:procgen-{args.game}-v0",
    num_levels=0,
    distribution_mode=args.difficulty,
    use_backgrounds=False,
    render_mode='human',
    apply_api_compatibility=False,
)    


# create the agent and load the model
class Config:
    def __init__(self, stack_size, batch_norm, normalize_v_targets):
        self.stack_size = stack_size
        self.batch_norm = batch_norm
        self.normalize_v_targets = normalize_v_targets

config = Config(args.stack_size, args.batch_norm, args.normalize_v_targets)
policy = PPO(env, config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.load_state_dict(torch.load(model_path, map_location=device))
print("Model loaded successfully! Starting the game...")


# play the game
frame_to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

obs = env.reset()

state_deque = deque()
for _ in range(config.stack_size):
    state_deque.append(frame_to_tensor(obs))
state = torch.concatenate(list(state_deque), axis=0)

policy.eval()
assert not policy.policy_net.training and not policy.value_net.training, "Policy should be in evaluation mode here"


while True:
    env.render()
    state = state.unsqueeze(0).to(device)
    action = policy.act(state)

    next_obs, _, done, _ = env.step(action)

    # udpate state to become next state using the new observation
    state_deque.popleft()
    state_deque.append(frame_to_tensor(next_obs))
    state = torch.concatenate(list(state_deque), axis=0)

    if done:
        # reset env and initial obs
        obs = env.reset()

        state_deque = deque()
        for _ in range(config.stack_size):
            state_deque.append(frame_to_tensor(obs))
        state = torch.concatenate(list(state_deque), axis=0)
