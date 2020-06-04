import gym
import numpy as np

# Wrapper to return one hot version of observations
class OneHotWrapper(gym.core.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.eye(self.observation_space.n)[obs]

    def step(self, action):
        obs, rew, done, info = super().step(action)

        # TODO: Might want to add original observation into the info object

        return self.observation(obs), rew, done, info

    def reset(self, **kwargs):
        initial_obs = self.env.reset(**kwargs)

        # One hot encode the initial observation
        return self.observation(initial_obs)
