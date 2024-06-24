import gym
import pathlib
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import wandb
import numpy as np

import utils

class RecorderWrapper(gym.Wrapper):
    def __init__(self, env, episode_frequency_rec):
        super().__init__(env)
        self.env = env
        self.episode_frequency_rec = episode_frequency_rec

        self.episode_counter = 1
        self.recording = False if episode_frequency_rec > 1 else True
        self.frames = []

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.recording:
            self.frames.append(np.moveaxis(obs, -1, 0))
            
            if terminated:
                self.save_video()
                self.recording = False
                self.frames = []
        
        if terminated:
            self.episode_counter += 1
            if self.episode_counter % self.episode_frequency_rec == 0:
                self.recording = True
            

        return obs, reward, terminated, truncated, info
    
    def save_video(self):
        wandb.log({"video": wandb.Video(np.array(self.frames), caption=f"step: {utils.global_step} - episode: {self.episode_counter}", fps=30, format="mp4")})

    def close(self):
        super().close()
    