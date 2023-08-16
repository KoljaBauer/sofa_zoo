import gym
import numpy as np
from gym.spaces import Box

from sofa_env.scenes.rope_threading.rope_threading_env import ActionType


class RopeThreadingNoGripperApertureWrapper(gym.Wrapper):
    @property
    def action_space(self):
        low_indexed = self.env.action_space.low[:-1]
        high_indexed = self.env.action_space.high[:-1]
        reduced_action_space = Box(low=low_indexed, high=high_indexed, shape=high_indexed.shape)
        return reduced_action_space

    def step(self, action):
        if self.action_type == ActionType.VELOCITY or self.action_type == ActionType.CONTINUOUS:
            pad_action = 0
        else: # position control
            pad_action = -1
        action = np.pad(action, pad_width=(0, 1), mode='constant', constant_values=(pad_action,))
        return self.env.step(action)

    @property
    def current_pos(self) -> np.ndarray:
        return self.env.current_pos[:-1]

    @property
    def current_vel(self) -> np.ndarray:
        return self.env.current_vel[:-1]