from __future__ import annotations

import unittest

import numpy as np
import spot_pick_place_demo_robust_grasp as demo


class TestSpotPickPlaceDemoRobustGrasp(unittest.TestCase):
    def test_default_policy_uses_lunar_best_eval_artifacts(self) -> None:
        run_dir = demo.SCRIPT_DIR / "runs" / "spot_go2_style_lunar_strategy_v3_gpu1_30m"

        self.assertEqual(demo.MODEL_PATH, run_dir / "best_eval" / "best_model.zip")
        self.assertEqual(demo.VECNORMALIZE_PATH, run_dir / "best_eval" / "best_vecnormalize.pkl")
        self.assertTrue(demo.MODEL_PATH.is_file())
        self.assertTrue(demo.VECNORMALIZE_PATH.is_file())

    def test_policy_and_vecnormalize_use_58_dimensions(self) -> None:
        demo.patch_sb3_zip_loader()
        model = demo.PPO.load(demo.MODEL_PATH, device="cpu")
        self.assertEqual(model.observation_space.shape, (demo.OBS_DIM,))

        vecnormalize = demo.VecNormalize.load(
            demo.VECNORMALIZE_PATH,
            demo.DummyVecEnv([demo._VecNormalizeEnv]),
        )
        self.assertEqual(vecnormalize.obs_rms.mean.shape, (demo.OBS_DIM,))
        vecnormalize.close()

    def test_policy_contract_matches_lunar_training_configuration(self) -> None:
        self.assertEqual(demo.OBS_DIM, 58)
        np.testing.assert_allclose(
            demo.NOMINAL_LEG_CTRL,
            np.array((0.0, -0.22, 0.55) * 4, dtype=np.float32),
        )
        np.testing.assert_allclose(
            demo.ACTION_SCALE,
            np.array((0.18, 0.30, 0.48) * 4, dtype=np.float32),
        )
        self.assertEqual(demo.ACTUATOR_GAIN_SCALE, 1.8)
        self.assertEqual(demo.GAIT_PERIOD, 0.56)
        self.assertEqual(demo.GAIT_CONTACT_SHARPNESS, 3.0)

    def test_arm_control_uses_low_wide_fixed_stance(self) -> None:
        example = object.__new__(demo.SpotPickPlaceDemo)
        example.last_action = np.ones(demo.ACT_DIM, dtype=np.float32)
        example.gait_phase = 1.0
        example.nominal_leg_ctrl = demo.NOMINAL_LEG_CTRL.copy()
        writes: list[tuple[np.ndarray, np.ndarray]] = []
        example._write_ctrl = lambda leg_ctrl, arm_q: writes.append((leg_ctrl.copy(), arm_q.copy()))
        arm_q = np.arange(demo.ARM_ACTUATOR_COUNT, dtype=np.float32)

        example._write_arm_ctrl(arm_q)

        expected_leg_ctrl = np.array(
            (
                0.18,
                -0.10,
                0.15,
                -0.18,
                -0.10,
                0.15,
                0.18,
                -0.10,
                0.15,
                -0.18,
                -0.10,
                0.15,
            ),
            dtype=np.float32,
        )
        np.testing.assert_allclose(demo.MANIPULATION_LEG_CTRL, expected_leg_ctrl)
        np.testing.assert_allclose(writes[0][0], expected_leg_ctrl)
        np.testing.assert_array_equal(writes[0][1], arm_q)
        np.testing.assert_array_equal(example.last_action, np.zeros(demo.ACT_DIM, dtype=np.float32))
        self.assertEqual(example.gait_phase, 0.0)

    def test_build_locomotion_observation_uses_v2_layout(self) -> None:
        base_angular = np.array((1.0, 2.0, 3.0), dtype=np.float32)
        projected_gravity = np.array((4.0, 5.0, 6.0), dtype=np.float32)
        command = np.array((7.0, 8.0, 9.0), dtype=np.float32)
        gait_observation = np.arange(10.0, 16.0, dtype=np.float32)
        joint_pos = np.arange(16.0, 28.0, dtype=np.float32)
        joint_vel = np.arange(28.0, 40.0, dtype=np.float32)
        last_action = np.arange(40.0, 52.0, dtype=np.float32)
        foot_contacts = np.arange(52.0, 56.0, dtype=np.float32)
        base_linear = np.arange(56.0, 59.0, dtype=np.float32)

        observation = demo._build_locomotion_observation(
            base_angular=base_angular,
            projected_gravity=projected_gravity,
            command=command,
            gait_observation=gait_observation,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            last_action=last_action,
            foot_contacts=foot_contacts,
            base_linear=base_linear,
        )

        expected = np.concatenate(
            [
                base_angular * 0.25,
                projected_gravity,
                command * np.array((2.0, 2.0, 0.25), dtype=np.float32),
                gait_observation,
                joint_pos,
                joint_vel * 0.05,
                last_action,
                foot_contacts,
                base_linear * 0.5,
            ]
        ).astype(np.float32)
        np.testing.assert_array_equal(observation, expected)
        self.assertEqual(observation.shape, (58,))


if __name__ == "__main__":
    unittest.main()
