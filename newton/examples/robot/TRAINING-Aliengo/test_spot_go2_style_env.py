from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from spot_go2_style_env import LEGACY_OBS_DIM, OBS_DIM, SpotGo2StyleEnv

SCRIPT_DIR = Path(__file__).resolve().parent


class TestSpotGo2StyleEnv(unittest.TestCase):
    def make_env(self, **kwargs) -> SpotGo2StyleEnv:
        return SpotGo2StyleEnv(
            xml_path=SCRIPT_DIR / "spot_scene.xml",
            use_curriculum=False,
            command_resample_seconds=0.0,
            **kwargs,
        )

    def test_observation_versions(self) -> None:
        for version, expected_dim in (("v1", LEGACY_OBS_DIM), ("v2", OBS_DIM)):
            with self.subTest(version=version):
                env = self.make_env(observation_version=version)
                try:
                    obs, _ = env.reset(seed=1)
                    self.assertEqual(obs.shape, (expected_dim,))
                    self.assertTrue(np.isfinite(obs).all())
                finally:
                    env.close()

    def test_reset_uses_local_terrain_height(self) -> None:
        env = self.make_env(observation_version="v2")
        try:
            env.reset(seed=2, options={"spawn_offset": (1.0, -0.75)})
            clearance = env._base_clearance()
            self.assertGreaterEqual(clearance, env.reset_base_clearance - 0.02)
            self.assertLess(clearance, env.reset_base_clearance + 0.20)
        finally:
            env.close()

    def test_gait_switches_between_stand_and_trot(self) -> None:
        env = self.make_env(observation_version="v2")
        try:
            env.reset(seed=3, options={"command": (0.0, 0.0, 0.0)})
            np.testing.assert_allclose(env._desired_contacts(), np.ones(4), atol=1e-6)
            env.step(np.zeros(12, dtype=np.float32))
            self.assertEqual(env._gait_phase(), 0.0)

            env.command[:] = (0.5, 0.0, 0.0)
            for _ in range(7):
                env.step(np.zeros(12, dtype=np.float32))
            self.assertAlmostEqual(env._gait_phase(), 0.25, places=6)
            contacts = env._desired_contacts()
            self.assertGreater(contacts[0], 0.99)
            self.assertGreater(contacts[3], 0.99)
            self.assertLess(contacts[1], 0.01)
            self.assertLess(contacts[2], 0.01)
        finally:
            env.close()

    def test_randomized_reset_and_step_are_finite(self) -> None:
        env = self.make_env(
            observation_version="v2",
            randomize_domain=True,
            max_action_delay=1,
        )
        try:
            obs, _ = env.reset(seed=4)
            self.assertTrue(np.isfinite(obs).all())
            for _ in range(25):
                obs, reward, _, _, _ = env.step(np.zeros(12, dtype=np.float32))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
