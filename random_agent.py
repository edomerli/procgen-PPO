import gym
import argparse
import numpy as np

supported_games = ["coinrun"]   # TODO: add more games
supported_difficulties = ["easy", "hard"]


# parse arguments
parser = argparse.ArgumentParser("Script to launch a random agent on a selected game and difficulty level and compute the average cumulative reward")
game_arg = parser.add_argument("--game", type=str, help=f"Game to play. Supported games are: {supported_games}")
diff_arg = parser.add_argument("--difficulty", type=str, help=f"Difficulty level of the game. Possible choices are: {supported_difficulties}")

args = parser.parse_args()

if args.game is None:
    raise argparse.ArgumentError(game_arg, "Please specify the game to play")

if args.difficulty is None:
    raise argparse.ArgumentError(diff_arg, "Please specify the difficulty level of the game")

if args.game not in supported_games:
    raise argparse.ArgumentError(game_arg, f"Game {args.game} is not supported. Supported games are: {supported_games}")

if args.difficulty not in supported_difficulties:
    raise argparse.ArgumentError(diff_arg, f"Difficulty level {args.difficulty} is not supported. Possible choices are: {supported_difficulties}")


# create the environment
env = gym.make(
    f"procgen:procgen-{args.game}-v0",
    num_levels=0,
    distribution_mode=args.difficulty,
    use_backgrounds=False,
    render_mode='rgb_array',
    apply_api_compatibility=False,
    rand_seed=6
)    

_ = env.reset()

total_cum_reward = 0
num_episodes = 0

while num_episodes < 1000:

    action = np.random.choice(env.action_space.n)

    _, reward, done, _ = env.step(action)

    total_cum_reward += reward

    if done:
        # reset env and initial obs
        obs = env.reset()
        # count episode for the final average
        num_episodes += 1
    
        if num_episodes % 10 == 0:
            print(f"Episode {num_episodes} - Average cumulative reward: {total_cum_reward/num_episodes}")

print(f"Average cumulative reward over {num_episodes} episodes: {total_cum_reward/num_episodes}")