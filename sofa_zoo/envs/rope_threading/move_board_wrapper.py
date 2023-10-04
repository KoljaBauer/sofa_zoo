import numpy as np
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper
from sofa_env.utils.math_helper import point_rotation_by_quaternion


class MoveBoardWrapper(RawInterfaceWrapper):
    # Move the board during execution of trajectory (for replanning)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_counter = 0
        self.move_time_step = 29  # time step at which to move the board
        self.context_change_noise_low = np.array([0.0, -10.0, 0.0, 0.0])
        self.context_change_noise_high = np.array([0.0, 10.0, 0.0, 0.0])

    def step(self, action):
        self.step_counter += 1
        if self.step_counter == self.move_time_step:

            offset_random = self.env.unwrapped.eyes[0].rng.uniform(self.context_change_noise_low,
                                                                   self.context_change_noise_high)
            self.move_board(offset=offset_random)

        return self.env.step(action)

    def reset(self):
        self.step_counter = 0
        return self.env.reset()

    def set_context_change_noise(self, context_change_noise: dict):
        self.context_change_noise_low = context_change_noise['low']
        self.context_change_noise_high = context_change_noise['high']

    def move_board(self, offset):
        print("MOVING BOARD!")
        eye = self.env.unwrapped.eyes[0]
        old_eye_pose = np.append(eye.position, eye.rotation)
        xyzphi = old_eye_pose + offset
        quaternion = np.array([0, 0, np.sin(xyzphi[-1] * np.pi / 360), np.cos(xyzphi[-1] * np.pi / 360)])
        new_pose = np.append(xyzphi[:3], quaternion)

        with eye.rigid_object.mechanical_object.position.writeable() as eye_pose:
            eye_pose[:] = new_pose

        eye.peg_pose[:3] = xyzphi[:3]
        eye.peg_pose[3:] = quaternion
        eye.center_pose[:3] = xyzphi[:3] + point_rotation_by_quaternion(np.array([1.55, 0.0, 18.7]), quaternion)
        eye.center_pose[3:] = quaternion
        eye.position = xyzphi[:3]
        eye.rotation = xyzphi[3]

        board = self.env.unwrapped.scene_creation_result['board_mo']
        old_board_pose = np.squeeze(board.position.value)
        new_board_position = old_board_pose[:3] + offset[:3]
        #new_board_pose = np.append(new_board_position , quaternion)
        # board rotation would not match the movement of eye, so don't rotate the board for now
        new_board_pose = np.append(new_board_position, old_board_pose[3:])
        with self.env.unwrapped.scene_creation_result['board_mo'].position.writeable() as board_pose:
            board_pose[:] = new_board_pose

