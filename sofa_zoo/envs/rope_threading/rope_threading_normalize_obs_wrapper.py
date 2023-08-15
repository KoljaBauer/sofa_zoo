import gym
import numpy as np


class RopeThreadingNormalizeObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Components of observation space, if observation_type == ObservationType.STATE
        # has_grasped -> 1 if single_agent else 2
        # ptsda_state -> 5 if single_agent else 10
        # gripper_pose -> 7 if single_agent else 14
        # position of rope tip -> 3
        # rope_tracking_point_positions -> num_rope_tracking_points * 3
        # active eye pose -> 4

        self.obs_min = np.array([0.0, # has_grasped
                            -90.0, -90.0, 0.0, 0.0, 0.0,  # ptsda_state
                            -20.0, -20.0, 0.0, -1.0, -1.0, -1.0, -1.0, # gripper_pose
                            -70.0, -70.0, -50.0, # position of rope tip (can leave working space, limit is only for gripper)
                            -20.0, -20.0, 0.0, 0.0]) # active eye pose

        self.obs_max = np.array([1.0,  # has_grasped
                            90.0, 90.0, 360.0, 100.0, 60.0,  # ptsda_state
                            230.0, 180.0, 100.0, 1.0, 1.0, 1.0, 1.0,  # gripper_pose
                            280.0, 230.0, 150.0,  # position of rope tip (can leave working space, limit is only for gripper)
                            230.0, 180.0, 100.0, 180.0])  # active eye pose

        self.xyz_min = np.array([-70.0, -70.0, -50.0])
        self.xyz_max = np.array([280.0, 230.0, 150.0])

    def normalize_obs(self, obs):
        assert obs.shape[0] >= self.obs_min.shape[0]
        assert obs.shape[0] >= self.obs_max.shape[0]

        if obs.shape[0] > self.obs_min.shape[0]:
            assert (obs.shape[0] - self.obs_min.shape[0]) % 3 == 0
            num_rope_tracking_points = int((obs.shape[0] - self.obs_min.shape[0]) / 3)
            obs_min_ = np.concatenate((self.obs_min, np.tile(self.xyz_min, num_rope_tracking_points)))
            obs_max_ = np.concatenate((self.obs_max, np.tile(self.xyz_max, num_rope_tracking_points)))
        else:
            obs_min_ = self.obs_min
            obs_max_ = self.obs_max

        obs_norm = 2 * np.divide(obs - obs_min_, obs_max_ - obs_min_) - 1

        if np.any(np.greater(obs_norm, 1)) or np.any(np.less(obs_norm, -1)):
            print("WARNING: UNEXPECTED OBSERVATION")
            print(f"unnormalized obs: {obs}")
            print(f"normalized obs: {obs_norm}")
        return obs_norm

    def step(self, action):
        obs, rewards, dones, info = self.env.step(action)
        normalized_obs = self.normalize_obs(obs)
        return normalized_obs, rewards, dones, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        normalized_obs = self.normalize_obs(obs)
        return normalized_obs
