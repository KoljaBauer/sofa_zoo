import numpy as np
import os
import gym
import gym.spaces
from gym.wrappers import TimeLimit, ResizeObservation

from copy import deepcopy

from typing import Callable, Type, Tuple, Optional, Union, List
from pathlib import Path

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecMonitor, VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from sofa_env.base import SofaEnv, RenderMode
from sofa_env.wrappers.trajectory_recorder import TrajectoryRecorder
from sofa_zoo.common.callbacks import RenderCallback, EpisodeInfoLoggerCallback, AdjustLoggingWindow
from sofa_zoo.common.reset_process_vec_env import WatchdogVecEnv
from sofa_zoo.envs.rope_threading.rope_threading_no_gripper_aperture_wrapper import RopeThreadingNoGripperApertureWrapper
from sofa_zoo.envs.rope_threading.rope_threading_normalize_obs_wrapper import RopeThreadingNormalizeObsWrapper


def configure_make_env(env_kwargs: dict, EnvClass: Type[SofaEnv], max_episode_steps: int) -> Callable:
    """Returns a make_env function that is configured with given env_kwargs."""

    def make_env() -> gym.Env:

        env_kwargs_local = deepcopy(env_kwargs)

        add_trajectoy_recorder_wrapper = env_kwargs_local.pop("add_trajectoy_recorder_wrapper", False)
        add_resize_observation_wrapper = False
        window_size = env_kwargs_local.pop("window_size", None)
        observation_shape = env_kwargs_local.pop("image_shape", None)
        trajectory_dir = env_kwargs_local.pop("trajectory_dir", "default_traj_dir")

        after_step_callbacks = env_kwargs_local.pop("after_step_callbacks", [])
        after_reset_callbacks = env_kwargs_local.pop("after_reset_callbacks", [])
        save_compressed_keys = env_kwargs_local.pop("save_compressed_keys", [])

        user_specified_observation_shape = False if observation_shape is None else True

        control_gripper_aperture = env_kwargs_local.pop("control_gripper_aperture", True)
        normalize_obs_static = env_kwargs_local.pop("normalize_obs_static", False)

        if env_kwargs_local.get("render_mode", None) == RenderMode.HUMAN:

            if not window_size == observation_shape:
                assert window_size is not None
                env_kwargs_local["image_shape"] = window_size
                add_resize_observation_wrapper = True
            else:
                # if there was an entry of image_shape, put it back in
                if user_specified_observation_shape:
                    env_kwargs_local["image_shape"] = observation_shape
        else:
            env_kwargs_local["image_shape"] = observation_shape

        env = EnvClass(**env_kwargs_local)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

        if not control_gripper_aperture:
            env = RopeThreadingNoGripperApertureWrapper(env)

        if normalize_obs_static:
            env = RopeThreadingNormalizeObsWrapper(env)

        # TODO observation_type.name is used, because the enum is created in every env -> direct comparison not
        # possible. Is there a cleaner way?
        if add_resize_observation_wrapper and env.observation_type.name == "RGB" and isinstance(env.observation_space, gym.spaces.Box):

            assert observation_shape is not None
            env = ResizeObservation(env, observation_shape)

        if add_trajectoy_recorder_wrapper:

            env = TrajectoryRecorder(
                env,
                log_dir=trajectory_dir,
                metadata=None,
                store_info=True,
                save_compressed_keys=save_compressed_keys,
                after_step_callbacks=after_step_callbacks,
                after_reset_callbacks=after_reset_callbacks,
            )

        return env

    return make_env


def configure_learning_pipeline(
    env_class: Type[SofaEnv],
    env_kwargs: dict,
    pipeline_config: dict,
    monitoring_keywords: Union[Tuple[str, ...], List[str]],
    normalize_observations: bool,
    normalize_reward: bool,
    algo_class: Union[Type[OnPolicyAlgorithm], Type[OffPolicyAlgorithm]],
    algo_kwargs: dict,
    render: bool,
    log_dir: str = "runs/",
    extra_callbacks: Optional[List[Type[BaseCallback]]] = None,
    random_seed: Optional[int] = None,
    dummy_run: bool = False,
    use_wandb: bool = False,
    model_checkpoint_distance: Optional[int] = None,
    use_watchdog_vec_env: bool = False,
    watchdog_vec_env_timeout: Optional[float] = None,
    reset_process_on_env_reset: bool = False,
    reward_clip: Optional[float] = None,
):

    if use_wandb:
        import wandb

    make_env = configure_make_env(
        env_kwargs,
        EnvClass=env_class,
        max_episode_steps=pipeline_config["max_episode_steps"],
    )

    if not dummy_run:
        if use_watchdog_vec_env:
            env = WatchdogVecEnv(
                [make_env] * pipeline_config["number_of_envs"],
                step_timeout_sec=watchdog_vec_env_timeout,
                reset_process_on_env_reset=reset_process_on_env_reset,
            )
        else:
            env = SubprocVecEnv([make_env] * pipeline_config["number_of_envs"])
    else:
        env = DummyVecEnv([make_env])

    # TODO: Is it okay to just comment this out?
    # env.seed(np.random.randint(0, 99999) if random_seed is None else random_seed)

    env = VecMonitor(
        env,
        info_keywords=monitoring_keywords,
    )

    if pipeline_config["videos_per_run"] > 0:
        # the video recorder counts steps per step_await -> additionally devide by the number_of_envs
        recorder_distance = int(np.floor(pipeline_config["total_timesteps"] / pipeline_config["videos_per_run"] / pipeline_config["number_of_envs"]))
        recorder_steps = list(range(0, int(pipeline_config["total_timesteps"] / pipeline_config["number_of_envs"]), recorder_distance))

        # if the video is longer than the time steps spent in an env for a batch, the video will be cut off -> go back until it fits
        try:
            extra_batches_necessary_to_fit_video = int(np.floor(pipeline_config["video_length"] / algo_kwargs["n_steps"]))
        except KeyError:
            extra_batches_necessary_to_fit_video = 0

        recorder_steps[-1] = recorder_steps[-1] - extra_batches_necessary_to_fit_video

        env = VecVideoRecorder(
            venv=env,
            video_folder=str(Path(log_dir) / f"videos{(f'/{wandb.run.id}') if use_wandb else ''}"),
            record_video_trigger=lambda x: x in recorder_steps,
            video_length=pipeline_config["video_length"],
        )

    # Reward and observation normalization
    if reward_clip is not None:
        normalize_kwargs = {"clip_reward": reward_clip}
    else:
        normalize_kwargs = {}
    env = VecNormalize(
        env,
        training=True,
        norm_obs=normalize_observations,
        norm_reward=normalize_reward,
        gamma=algo_kwargs["gamma"],
        **normalize_kwargs,
    )

    env = VecFrameStack(
        venv=env,
        n_stack=pipeline_config["frame_stack"],
    )

    model = algo_class(
        env=env,
        verbose=2,
        tensorboard_log=log_dir,
        **algo_kwargs,
    )

    callback_list = []

    callback_list.append(AdjustLoggingWindow(window_length=pipeline_config["number_of_envs"]))
    callback_list.append(EpisodeInfoLoggerCallback())

    if render:
        callback_list.append(RenderCallback())

    if use_wandb:
        from wandb.integration.sb3 import WandbCallback

        callback_list.append(
            WandbCallback(
                gradient_save_freq=10000,
                model_save_path=log_dir,
                verbose=2,
            )
        )
        model_log_dir=log_dir
    else:
        model_log_dir = log_dir

    if model_checkpoint_distance is not None:
        callback_list.append(
            CheckpointCallback(
                save_freq=max(model_checkpoint_distance // pipeline_config["number_of_envs"], 1),
                save_path=model_log_dir,
                name_prefix="rl_model",
            )
        )

    if extra_callbacks is not None:
        callback_list = callback_list + extra_callbacks

    callback = CallbackList(callback_list)

    return model, callback
