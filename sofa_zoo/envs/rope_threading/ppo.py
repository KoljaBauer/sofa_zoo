import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType, RopeThreadingEnv

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

import wandb

from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger

from sofa_env.scenes.rope_threading.rope_threading_env import ActionType


class PPOIterativeExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        wandb.init(sync_tensorboard=True)

        add_render_callback = True
        continuous_actions = True
        normalize_reward = True
        reward_clip = np.inf

        # observations, bimanual, randomized, eyes
        self.parameters = ["STATE", "False", "True", "1"]

        observation_type = ObservationType[self.parameters[0]]
        image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

        eye_configs = {
            "1": [
                (60, 10, 0, 90),
            ],
            "2": [
                (60, 10, 0, 90),
                (10, 10, 0, 90),
            ],
        }

        bimanual_grasp = self.parameters[1] == "True"
        randomized_eye = self.parameters[2] == "True"
        image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

        env_kwargs = {
            "image_shape": (64, 64),
            "window_size": (200, 200),
            "observation_type": observation_type,
            "time_step": 0.01,
            "frame_skip": 10,
            "settle_steps": 20,
            #"render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
            "render_mode": RenderMode.HUMAN,
            "reward_amount_dict": {
                "passed_eye": 10.0,
                "lost_eye": -20.0,  # more than passed_eye
                "goal_reached": 100.0,
                "distance_to_active_eye": -0.0,
                "lost_grasp": -0.1,
                "collision": -0.1,
                "floor_collision": -0.1,
                "bimanual_grasp": 0.0,
                "moved_towards_eye": 200.0,
                "moved_away_from_eye": -200.0,
                "workspace_violation": -0.01,
                "state_limit_violation": -0.01,
                "distance_to_lost_rope": -0.0,
                "delta_distance_to_lost_rope": -0.0,
                "fraction_rope_passed": 0.0,
                "delta_fraction_rope_passed": 200.0,
            },
            "create_scene_kwargs": {
                "eye_config": eye_configs[self.parameters[3]],
                "randomize_gripper": False,
                "start_grasped": True,
                "randomize_grasp_index": False,
            },
            "on_reset_callbacks": None,
            "color_eyes": True,
            "individual_agents": False,
            "only_right_gripper": not bimanual_grasp,
            "fraction_of_rope_to_pass": 0.05,
            "num_rope_tracking_points": 10,
        }

        create_scene_kwargs = config['params'].get('create_scene_kwargs', {})
        env_kwargs['control_gripper_aperture'] = create_scene_kwargs.get('control_gripper_aperture', False)

        if bimanual_grasp:
            env_kwargs["reward_amount_dict"]["bimanual_grasp"] = 100.0
            env_kwargs["reward_amount_dict"]["distance_to_bimanual_grasp"] = -0.0
            env_kwargs["reward_amount_dict"]["delta_distance_to_bimanual_grasp"] = -200.0

        if randomized_eye:
            env_kwargs["create_scene_kwargs"]["eye_reset_noise"] = {
                "low": np.array([-20.0, -20.0, 0.0, -15]),
                "high": np.array([20.0, 20.0, 0.0, 15]),
            }

        # config = {"max_episode_steps": 200 + 150 * (len(eye_configs[self.parameters[3]]) - 1), **CONFIG}
        self.config = {"max_episode_steps": 100, **CONFIG}

        if image_based:
            ppo_kwargs = PPO_KWARGS["image_based"]
        else:
            ppo_kwargs = PPO_KWARGS["state_based"]

        info_keywords = [
            "distance_to_active_eye",
            "lost_grasps",
            "recovered_lost_grasps",
            "passed_eyes",
            "lost_eyes",
            "collisions",
            "floor_collisions",
            "successful_task",
            "rew_delta_distance",
            "rew_absolute_distance",
            "rew_losing_eyes",
            "rew_losing_grasp",
            "rew_collisions",
            "rew_floor_collisions",
            "rew_workspace_violation",
            "rew_state_limit_violation",
            "rew_dist_to_lost_rope",
            "rew_delt_dist_to_lost_rope",
            "rew_passed_eyes",
            "rew_bimanual_grasp",
            "rew_dist_to_bimanual_grasp",
            "rew_delt_dist_to_bimanual_grasp",
            "rew_fraction_passed",
            "rew_delta_fraction_passed",
        ]

        self.config["ppo_config"] = ppo_kwargs
        self.config["env_kwargs"] = env_kwargs
        self.config["info_keywords"] = info_keywords

        self.config["videos_per_run"] = 0
        self.config["frame_stack"] = 1
        self.config['total_timesteps'] = 5e5
        env_kwargs['action_type'] = ActionType.VELOCITY

        self.model, self.callback = configure_learning_pipeline(
            env_class=RopeThreadingEnv,
            env_kwargs=env_kwargs,
            pipeline_config=self.config,
            monitoring_keywords=info_keywords,
            normalize_observations=False if image_based else True,
            algo_class=PPO,
            algo_kwargs=ppo_kwargs,
            render=add_render_callback,
            normalize_reward=normalize_reward,
            reward_clip=reward_clip,
            use_wandb=True,
            use_watchdog_vec_env=True,
            watchdog_vec_env_timeout=20.0,
            reset_process_on_env_reset=False,
        )

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.callback,
            tb_log_name=f"{self.parameters[0]}_{self.parameters[1]}Biman_{self.parameters[2]}Random_{self.parameters[3]}",
        )

        return None

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        log_path = str(self.model.logger.dir)
        self.model.save(log_path + "saved_model.pth")



if __name__ == "__main__":
    cw = cluster_work.ClusterWork(PPOIterativeExperiment)
    # RUN
    cw.run()