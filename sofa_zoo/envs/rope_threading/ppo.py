import numpy as np
import torch as ch
from stable_baselines3 import PPO

from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType, RopeThreadingEnv

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

import wandb

from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work

from sofa_env.scenes.rope_threading.rope_threading_env import ActionType
from sofa_zoo.common.schedules import linear_schedule


class PPOIterativeExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        wandb.init(name=config['params']['exp_name'], sync_tensorboard=True, config=config['params'])

        add_render_callback = True
        continuous_actions = True
        normalize_reward = True
        reward_clip = np.inf


        env_kwargs = {
            "image_shape": (64, 64),
            "window_size": (200, 200),
            "observation_type": ObservationType["STATE"],
            "time_step": 0.01,
            "frame_skip": 10,
            "settle_steps": 20,
            "render_mode": RenderMode.NONE,
            #"render_mode": RenderMode.HUMAN,
            "on_reset_callbacks": None,
            "color_eyes": True,
            "individual_agents": False,
            "only_right_gripper": True,
            "fraction_of_rope_to_pass": 0.05,
        }

        create_scene_kwargs = config['params'].get('create_scene_kwargs', {})
        env_kwargs['control_gripper_aperture'] = create_scene_kwargs.pop('control_gripper_aperture', False)
        reward_amount_dict = config['params'].get('reward_amount_dict', {})
        env_kwargs['create_scene_kwargs'] = create_scene_kwargs
        env_kwargs['reward_amount_dict'] = reward_amount_dict

        num_rope_tracking_points = config['params'].get('num_rope_tracking_points', 0)
        env_kwargs['num_rope_tracking_points'] = num_rope_tracking_points


        env_kwargs['normalize_obs_static'] = config['params'].get('normalize_obs_static', False)
        normalize_obs_dynamic = config['params'].get('normalize_obs_dynamic', False)

        self.config = {"max_episode_steps": 100, **CONFIG}
        if config['params'].get('number_of_envs', False):
            self.config['number_of_envs'] = config['params']['number_of_envs']

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
        action_type = config['params'].get('action_type', "velocity")
        if action_type == "velocity":
            env_kwargs['action_type'] = ActionType.VELOCITY
        else:
            env_kwargs['action_type'] = ActionType.CONTINUOUS

        seed = config['params'].get('seed', 0) + rep
        ch.manual_seed(seed)

        learning_rate = config['params'].get('learning_rate', None)
        if learning_rate is not None:
            ppo_kwargs['learning_rate'] = linear_schedule(learning_rate)

        total_timesteps = config['params'].get('total_timesteps', None)
        if total_timesteps is not None:
            self.config['total_timesteps'] = total_timesteps

        move_board_during_execution = config['params'].get('move_board_during_execution')

        if move_board_during_execution:
            env_kwargs['move_board_during_execution'] = True


        self.model, self.callback = configure_learning_pipeline(
            env_class=RopeThreadingEnv,
            env_kwargs=env_kwargs,
            pipeline_config=self.config,
            monitoring_keywords=info_keywords,
            normalize_observations=normalize_obs_dynamic,
            algo_class=PPO,
            algo_kwargs=ppo_kwargs,
            render=add_render_callback,
            normalize_reward=normalize_reward,
            random_seed=seed,
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
            tb_log_name=f"test_PPO",
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