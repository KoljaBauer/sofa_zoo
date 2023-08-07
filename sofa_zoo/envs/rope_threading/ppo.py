import numpy as np
import wandb
from stable_baselines3 import PPO

from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType, RopeThreadingEnv, ActionType
from sofa_zoo.envs.rope_threading.rope_threading_no_gripper_aperture_wrapper import RopeThreadingNoGripperApertureWrapper

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

from sofa_zoo.envs.rope_threading.experiment_params import env_kwargs as env_kwargs_default


def build_model():
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

    env_kwargs['reward_amount_dict']['workspace_violation'] = -1.0
    env_kwargs['control_gripper_aperture'] = False
    env_kwargs['action_type'] = ActionType.VELOCITY

    if bimanual_grasp:
        env_kwargs["reward_amount_dict"]["bimanual_grasp"] = 100.0
        env_kwargs["reward_amount_dict"]["distance_to_bimanual_grasp"] = -0.0
        env_kwargs["reward_amount_dict"]["delta_distance_to_bimanual_grasp"] = -200.0

    if randomized_eye:
        env_kwargs["create_scene_kwargs"]["eye_reset_noise"] = {
            "low": np.array([-15.0, -15.0, 0.0, -35.0]),
            "high": np.array([15.0, 15.0, 0.0, 35.0]),
        }

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

    model, callback = configure_learning_pipeline(
        env_class=RopeThreadingEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        #normalize_observations=False if image_based else True,
        normalize_observations=True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
        use_wandb=True,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=20.0,
        reset_process_on_env_reset=False,
        model_checkpoint_distance=config["checkpoint_distance"]
    )

    return model, callback, config


if __name__ == "__main__":
    wandb.init(sync_tensorboard=True)

    model, callback, config = build_model()

    # observations, bimanual, randomized, eyes
    parameters = ["STATE", "False", "True", "1"]
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name=f"{parameters[0]}_{parameters[1]}Biman_{parameters[2]}Random_{parameters[3]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
