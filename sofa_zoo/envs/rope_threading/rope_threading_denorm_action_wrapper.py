import numpy as np

from copy import deepcopy
from sofa_env.scenes.rope_threading.rope_threading_env import ActionType
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper

from gym.spaces import Box


class RopeThreadingDenormActionWrapper(RawInterfaceWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # min and max for ptsda states in sofa_env
        self.p_min = np.array([-90.0, -90.0, 0.0, 0.0, 0.0])
        self.p_max = np.array([90.0, 90.0, 360.0, 100.0, 60.0])

    def denormalize_pos(self, pos):
        assert pos.shape == self.p_min.shape
        assert pos.shape == self.p_max.shape

        pos_denorm = self.p_min + ((pos + 1) / 2) * (self.p_max - self.p_min)
        return pos_denorm

    def step(self, action):
        # Receive normalized velocities or positions as action, denormalize and input to step function
        if self.action_type == ActionType.VELOCITY:
            action_denorm = self.env.action_space.low + ((action + 1) / 2) * (self.env.action_space.high - self.env.action_space.low)
        elif self.action_type == ActionType.POSITION:  # action_type position
            action_denorm = self.denormalize_pos(action)
        else: # action_type continuous, don't denormalize
            action_denorm = action

        return self.env.step(action_denorm)

    @property
    def action_space(self):
        if self.action_type == ActionType.VELOCITY:
            low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])
            high = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            clipped_action_space = Box(low=low, high=high, shape=high.shape)
            return clipped_action_space
        else:
            return self.env.action_space


