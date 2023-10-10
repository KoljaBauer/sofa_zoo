import gym
import numpy as np
from gym.spaces import Box


class RopeThreadingStiffnessObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_stiffness = 8e3
        self.max_stiffness = 1e5

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        stiffness = self.unwrapped.rope.node.BeamFEMForceField.youngModulus.value
        # normalize stiffness to range [-1.0, 1.0]
        stiffness_norm = (((stiffness - self.min_stiffness) / (self.max_stiffness - self.min_stiffness)) * 2) - 1
        extended_obs = np.append(obs, stiffness_norm)
        return extended_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        stiffness = self.unwrapped.rope.node.BeamFEMForceField.youngModulus.value
        # normalize stiffness to range [-1.0, 1.0]
        stiffness_norm = (((stiffness - self.min_stiffness) / (self.max_stiffness - self.min_stiffness)) * 2) - 1
        extended_obs = np.append(obs, stiffness_norm)
        return extended_obs

    @property
    def observation_space(self):
        low_old = self.env.observation_space.low
        high_old = self.env.observation_space.high
        low_new = np.append(low_old, -np.inf)
        high_new = np.append(high_old, np.inf)

        expanded_observation_space = Box(low=low_new, high=high_new, shape=high_new.shape)
        return expanded_observation_space
