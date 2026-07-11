import math
import unittest

import numpy as np

from fetch.policy_observation import (
    TimedCubeState,
    build_pushcube_observation,
    reorder_and_correct_foot_force,
    select_cube_state,
    world_vector_to_base_xy,
)


class PolicyObservationTest(unittest.TestCase):
    def _observation(self, **overrides):
        values = dict(
            angular_velocity_base=[1.0, 2.0, 3.0],
            projected_gravity=[0.0, 0.0, -1.0],
            joint_position_relative=np.arange(12, dtype=np.float32),
            joint_velocity=np.full(12, 2.0, dtype=np.float32),
            previous_clamped_command=[0.6, -0.4, 0.8],
            robot_position_world_xy=[3.0, 2.0],
            robot_velocity_world_xy=[0.2, -0.1],
            cube_position_world_xy=[2.0, 2.0],
            cube_velocity_world_xy=[1.0, 0.0],
            goal_position_world_xy=[1.0, 2.0],
            goal_radius=0.2,
            lf_foot_position_world_xy=[1.5, 2.0],
            foot_force=[10.0, 20.0, 30.0, 200.0],
            quaternion_world_from_base_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        values.update(overrides)
        return build_pushcube_observation(**values)

    def test_exact_order_scaling_and_clipping(self):
        obs = self._observation()
        self.assertEqual(obs.shape, (52,))
        np.testing.assert_allclose(obs[0:3], [0.2, 0.4, 0.6])
        np.testing.assert_allclose(obs[30:33], [0.6, -0.4, 0.8])
        np.testing.assert_allclose(obs[33:35], [1.0, 0.0])
        np.testing.assert_allclose(obs[37:39], [0.5, 0.0])
        np.testing.assert_allclose(obs[39:41], [1.0, 0.0])
        np.testing.assert_allclose(obs[41:44], [0.0, 0.0, 0.2])
        np.testing.assert_allclose(obs[44:46], [-0.5, 0.0])
        np.testing.assert_allclose(obs[46:48], [0.5, 0.0])
        np.testing.assert_allclose(obs[48:52], [0.1, 0.2, 0.3, 1.5])

    def test_inverse_full_quaternion_rotation(self):
        yaw = math.pi / 2.0
        quaternion = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
        np.testing.assert_allclose(
            world_vector_to_base_xy([1.0, 0.0], quaternion), [0.0, -1.0], atol=1e-6
        )

    def test_foot_force_order_and_offset(self):
        corrected = reorder_and_correct_foot_force([11, 22, 33, 44], [1, 2, 3, 4])
        np.testing.assert_allclose(corrected, [20, 10, 40, 30])

    def test_cube_history_target_and_dropout_hold(self):
        states = [
            TimedCubeState(float(stamp), np.array([stamp, 0]), np.zeros(2))
            for stamp in (1, 2, 3)
        ]
        self.assertEqual(select_cube_state(states, 2.5).stamp_s, 2)
        self.assertEqual(select_cube_state(states, 10.0).stamp_s, 3)
        self.assertIsNone(select_cube_state(states, 0.5))

if __name__ == "__main__":
    unittest.main()
