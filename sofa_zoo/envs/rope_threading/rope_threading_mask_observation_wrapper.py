import gym
import numpy as np
from gym.spaces import Box


class RopeThreadingMaskObsWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.state_obs_mask = np.concatenate(([False], # exclude right_has_grasped
                               [True] * 4, [False],  # include right_ptsd but not right_a state
                               [False] * 7, # exclude right_gripper_pose
                               [False] * 3, # exclude rope_tip_position
                               [True] * 4, # include active_eye_pose
                            ))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        assert obs.shape[0] == 20 # right_has_grasped, right_ptsda_state, right_gripper_pose, rope_tip_position, active_eye_pose
        masked_obs = obs[self.state_obs_mask]
        assert masked_obs.shape[0] == 8
        return masked_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        assert obs.shape[0] == 20  # right_has_grasped, right_ptsda_state, right_gripper_pose, rope_tip_position, active_eye_pose
        masked_obs = obs[self.state_obs_mask]
        assert masked_obs.shape[0] == 8
        return masked_obs

    @property
    def observation_space(self):
        low_new = np.array([-np.inf] * 8)
        high_new = np.array([np.inf] * 8)
        masked_observation_space = Box(low=low_new, high=high_new, shape=high_new.shape)
        return masked_observation_space
