from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType

parameters = ["STATE", "False", "True", "1"]

observation_type = ObservationType[parameters[0]]

eye_configs = {
        "1": [
            (60, 10, 0, 90),
        ],
    }

bimanual_grasp = parameters[1] == "True"
add_render_callback = False
randomized_eye = parameters[2] == "True"
image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (400, 400),
        "observation_type": observation_type,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 20,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        #"render_mode": RenderMode.HUMAN,
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
            "eye_config": eye_configs[parameters[3]],
            "randomize_gripper": False,
            "start_grasped": True,
            "randomize_grasp_index": False,
        },
        "on_reset_callbacks": None,
        "color_eyes": True,
        "individual_agents": False,
        "only_right_gripper": not bimanual_grasp,
        "fraction_of_rope_to_pass": 0.05,
        "num_rope_tracking_points": 0,
    }