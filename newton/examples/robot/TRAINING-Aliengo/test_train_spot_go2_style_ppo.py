from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train_spot_go2_style_ppo import LocomotionEvalCallback, _resolve_resume_vecnormalize


class TestTrainingResume(unittest.TestCase):
    def test_resolve_best_model_vecnormalize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for directory_name in ("best_eval", "best_tracking", "best_gait"):
                with self.subTest(directory_name=directory_name):
                    best_dir = Path(tmp_dir) / directory_name
                    best_dir.mkdir()
                    model_path = best_dir / "best_model.zip"
                    vecnormalize_path = best_dir / "best_vecnormalize.pkl"
                    vecnormalize_path.touch()

                    self.assertEqual(_resolve_resume_vecnormalize(model_path, None), vecnormalize_path)

    def test_eval_callback_restores_existing_best_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            self._write_score(run_dir / "best_eval", robust_score=10.0)
            self._write_score(
                run_dir / "best_tracking",
                mean_survival=0.9,
                mean_linear_rmse=0.2,
                mean_yaw_rmse=0.4,
            )
            self._write_score(
                run_dir / "best_gait",
                mean_survival=0.9,
                mean_gait_match=0.8,
                mean_foot_slip=0.5,
            )

            callback = LocomotionEvalCallback(
                eval_env=None,
                save_dir=run_dir / "best_eval",
                eval_freq=1,
                eval_repetitions=1,
                scenarios=(),
                early_stop_patience=0,
                early_stop_min_timesteps=0,
                early_stop_min_delta=0.0,
                restore_best_scores=True,
            )

            self.assertEqual(callback.best_score, 10.0)
            self.assertAlmostEqual(callback.best_tracking_score, 0.6)
            self.assertAlmostEqual(callback.best_gait_score, 1.65)

    def test_eval_callback_starts_fresh_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir = Path(tmp_dir) / "best_eval"
            self._write_score(save_dir, robust_score=10.0)

            callback = LocomotionEvalCallback(
                eval_env=None,
                save_dir=save_dir,
                eval_freq=1,
                eval_repetitions=1,
                scenarios=(),
                early_stop_patience=0,
                early_stop_min_timesteps=0,
                early_stop_min_delta=0.0,
                restore_best_scores=False,
            )

            self.assertEqual(callback.best_score, -float("inf"))

    @staticmethod
    def _write_score(directory: Path, **metrics: float) -> None:
        directory.mkdir()
        (directory / "best_score.txt").write_text(
            "".join(f"{name}={value}\n" for name, value in metrics.items()),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
