import numpy as np
import torch as ch
import wandb

from gymnasium import spaces
from sofa_zoo.envs.rope_threading.ppo import build_model
from stable_baselines3 import PPO

from stable_baselines3.common.utils import obs_as_tensor
from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode

def normalize_obs(obs):
    obs_min = np.array([0.0,  # right_has_grasped
                        -90.0, -90.0, 0.0, 0.0, 0.0,  # right_ptsda
                        -20.0, -20.0, 0.0, -1.0, -1.0, -1.0, -1.0,  # right_gripper_pose
                        -20.0, -20.0, 0.0,  # rope_tip_pos
                        -20.0, -20.0, 0.0, 0.0])  # active_eye_pose

    obs_max = np.array([1.0,  # right_has_grasped
                        90.0, 90.0, 360.0, 100.0, 60.0,  # right_ptsda
                        230.0, 180.0, 100.0, 1.0, 1.0, 1.0, 1.0,  # right_gripper_pose
                        230.0, 180.0, 100.0,  # rope_tip_pos
                        230.0, 180.0, 100.0, 180.0])  # active_eye_pose

    # obs_min = np.array([-20.0, -20.0, 0.0, 0.0])
    # obs_max = np.array([230.0, 180.0, 100.0, 180.0])
    # self.p_min = np.array([-90.0, -90.0, 0.0, 0.0, 0.0])
    # self.p_max = np.array([90.0, 90.0, 360.0, 100.0, 60.0])

    obs_norm = 2 * np.divide(obs - obs_min, obs_max - obs_min) - 1
    return obs_norm

def get_trajs_from_policy(model, n_samples=8):

    time_steps = 100
    rewards = ch.zeros((model.env.num_envs, time_steps), dtype=ch.float)
    trajs_gripper_ptsda = np.zeros((model.env.num_envs, time_steps, 5))
    trajs_gripper_xyz = np.zeros((model.env.num_envs, time_steps, 3))
    trajs_rope_tip_xyz = np.zeros((model.env.num_envs, time_steps, 3))
    eye_pose = np.zeros((model.env.num_envs, 4))

    obs = model.env.reset()
    #obs_unnorm = ppo_model.env.reset()
    #obs = normalize_obs(obs_unnorm)
    eye_poses = obs[:, 46:]

    repeat = int(np.ceil(n_samples / model.env.num_envs))

    for sample in range(repeat):
        for i in range(100):

            actions, _ = model.predict(obs, deterministic=True)

            #obs_unnorm, rew, done, info = ppo_model.env.step(actions)
            #obs = normalize_obs(obs_unnorm)
            obs, rew, done, info = model.env.step(actions)

            gripper_ptsda = obs[:, 1:6]
            gripper_xyz = obs[:, 6:9]
            rope_tip_xyz = obs[:, 13:16]


            trajs_gripper_ptsda[:, i, :] = gripper_ptsda
            trajs_gripper_xyz[:, i, :] = gripper_xyz
            trajs_rope_tip_xyz[:, i, :] = rope_tip_xyz

            rewards[done==False, i] = ch.from_numpy(rew).float()[done==False]
            #rewards[:, i] = ch.from_numpy(rew)

            acc_rewards = ch.sum(rewards, dim=1)
    with open('trajs_gripper_ptsda.npy', 'wb') as f:
        np.save(f, trajs_gripper_ptsda)
    with open('trajs_gripper_xyz.npy', 'wb') as f:
        np.save(f, trajs_gripper_xyz)
    with open('trajs_rope_tip_xyz.npy', 'wb') as f:
        np.save(f, trajs_rope_tip_xyz)
    with open('eye_poses.npy', 'wb') as f:
        np.save(f, eye_poses)


if __name__ == "__main__":
    #state_dict = ch.load('/home/kolja/code/sofa_zoo/sofa_zoo/test/save_dir/1e6_model/policy.pth')
    wandb.init()
    ppo_model, callback, _ = build_model(render_mode=RenderMode.HUMAN, n_envs=1)

    #policy = ppo_model.policy
    #policy.load_state_dict(state_dict)
    #policy.set_training_mode(False)

    #ppo_model.env.render_mode = RenderMode.HUMAN


    #model = PPO.load('/home/kolja/code/sofa_zoo/sofa_zoo/envs/rope_threading/wandb/run-20230531_193219-uukw0st7/files/logs/rl_model_1000000_steps.zip', env=ppo_model.env)
    model = PPO.load('/home/kolja/code/sofa_zoo/sofa_zoo/envs/rope_threading/wandb/run-20230814_180654-ry07gqkj/files/logs/rl_model_800000_steps.zip', env=ppo_model.env)

    get_trajs_from_policy(model)
    '''
    # observations, bimanual, randomized, eyes
    parameters = ["STATE", "False", "True", "1"]
    model.learn(
        total_timesteps=1000,
        callback=callback,
        tb_log_name=f"{parameters[0]}_{parameters[1]}Biman_{parameters[2]}Random_{parameters[3]}",
    )
    '''




