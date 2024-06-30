import gym
import wandb
import numpy as np

import utils

class RecorderWrapper(gym.Wrapper):
    """gym.Env wrapper for recording episodes as videos in wandb."""
    def __init__(self, env, episode_frequency_rec):
        """Constructor

        Args:
            env (gym.Env): the environment to wrap
            episode_frequency_rec (int): the frequency at which to record episodes
        """
        super().__init__(env)
        self.env = env
        self.episode_frequency_rec = episode_frequency_rec

        self.episode_counter = 1
        self.recording = False if episode_frequency_rec > 1 else True
        self.frames = []

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Overrides gym.Env.step() to record episodes as videos in wandb, before forwarding the outputs of the call to the wrapped environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.recording:
            # record a frame
            self.frames.append(np.moveaxis(obs, -1, 0))
            
            if terminated:
                # save the whole video recorded so far
                self.save_video()
                self.recording = False
                self.frames = []
        
        if terminated:
            # count the finished episode and start recording if it's the right time wrt the recording frequency
            self.episode_counter += 1
            if self.episode_counter % self.episode_frequency_rec == 0:
                self.recording = True
            

        return obs, reward, terminated, truncated, info
    
    def save_video(self):
        wandb.log({"video": wandb.Video(np.array(self.frames), caption=f"step: {utils.global_step} - episode: {self.episode_counter}", fps=30, format="mp4")})

    def close(self):
        super().close()
    