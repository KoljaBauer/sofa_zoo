import numpy as np
import wandb

from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
from cw2.cw_data.cw_wandb_logger import WandBLogger

from stable_baselines3 import PPO

from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType, RopeThreadingEnv, ActionType
from sofa_zoo.envs.rope_threading.rope_threading_no_gripper_aperture_wrapper import RopeThreadingNoGripperApertureWrapper
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS
from sofa_zoo.envs.rope_threading.experiment_params import env_kwargs as env_kwargs_default


def build_model(params, n_envs=None, render_mode= None):
    add_render_callback = False
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observations, bimanual, randomized, eyes
    parameters = ["STATE", "False", "True", "1"]
    bimanual_grasp = parameters[1] == "True"
    randomized_eye = parameters[2] == "True"

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = env_kwargs_default

    env_kwargs['reward_amount_dict'] = params['reward_amount_dict']
    create_scene_kwargs = params.get('create_scene_kwargs', {})
    env_kwargs['control_gripper_aperture'] = create_scene_kwargs.get('control_gripper_aperture', False)
    env_kwargs['normalize_obs_static'] = params.get('normalize_obs_static', False)
    normalize_obs_dynamic = params.get('normalize_obs_dynamic', False)
    env_kwargs['num_rope_tracking_points'] = params.get('num_rope_tracking_points', 0)
    env_kwargs['action_type'] = ActionType.VELOCITY
    if render_mode is not None:
        env_kwargs['render_mode'] = render_mode

    env_kwargs["create_scene_kwargs"]["eye_reset_noise"] = create_scene_kwargs.get('eye_reset_noise', None)

    # config = {"max_episode_steps": 200 + 150 * (len(eye_configs[parameters[3]]) - 1), **CONFIG}
    config = {"max_episode_steps": 100, **CONFIG}  # TODO: Is this correct?

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

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords
    config["videos_per_run"] = 0
    config["frame_stack"] = 1
    config["total_timesteps"] = int(1e6)
    config["checkpoint_distance"] = int(1e4)
    if n_envs is not None:
        config['number_of_envs'] = n_envs

    model, callback = configure_learning_pipeline(
        env_class=RopeThreadingEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=normalize_obs_dynamic,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
        use_wandb=True,
        use_watchdog_vec_env=False, #TODO
        watchdog_vec_env_timeout=20.0,
        reset_process_on_env_reset=False,
        model_checkpoint_distance=config["checkpoint_distance"]
    )

    return model, callback, config


class PPOIterativeExperiment(experiment.AbstractIterativeExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        wandb.init(name=config['params']['exp_name'], sync_tensorboard=True, config=config['params'])
        self.model, self.callback, self.config = build_model(config['params'])
        self.save_path = config['params']['path'] + '/log/rep_' + str(rep)

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        # observations, bimanual, randomized, eyes
        parameters = ["STATE", "False", "True", "1"]
        self.model.learn(
            total_timesteps=self.config["total_timesteps"],
            callback=self.callback,
            tb_log_name=self.save_path,
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
