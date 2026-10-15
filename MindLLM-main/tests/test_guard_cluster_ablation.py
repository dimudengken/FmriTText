import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

if os.name == "posix":
    from scripts import guard_cluster_ablation as guardian


@unittest.skipUnless(os.name == "posix", "server-side POSIX guardian")
class GuardianTest(unittest.TestCase):
    def test_early_stop_protocol_blocks_testing_after_epoch_limit(self):
        output = guardian.PROJECT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output, prefix="guard-test-") as directory:
            root = Path(directory)
            self.fixture(root)
            protocol = json.loads((root / "protocol.json").read_text())
            protocol["require_early_stop"] = True
            guardian.save(root / "protocol.json", protocol)
            status = json.loads((root / "status.json").read_text())
            status.update(phase="evaluate_functional", completed=["train_baseline", "train_functional", "train_random", "evaluate_baseline"])
            guardian.save(root / "status.json", status)
            for arm in ("functional", "random"):
                (root / arm).mkdir()
                guardian.save(root / arm / "result.json", {"stop_reason": "epoch_limit"})
            with patch.object(guardian, "alive", return_value=False), \
                 patch.object(guardian, "gpu_is_available", return_value=True), \
                 patch.object(guardian.subprocess, "Popen") as launch:
                with self.assertRaisesRegex(RuntimeError, "did not reach validation early stopping"):
                    guardian.guard(root)
            launch.assert_not_called()

    def fixture(self, root, error="exit code -11"):
        now = time.time()
        protocol = {"max_hours": 24, "source_sha256": {}, "seed": 42,
                    "initial_checkpoint": "base.ckpt", "feature_bank": "features.pt",
                    "max_epochs": 20, "batch_size": 4, "accumulate_grad_batches": 8,
                    "early_stop_patience": 5}
        guardian.save(root / "protocol.json", protocol)
        (root / "train_baseline.log").write_text(error)
        guardian.save(root / "status.json", {"pid": 999999, "child_pid": 999998,
            "state": "failed", "phase": "train_baseline", "started_unix": now,
            "phase_started_unix": now, "completed": [], "error": error,
            "log": str(root / "train_baseline.log")})
        (root / "baseline").mkdir()
        (root / "baseline/config.yaml").write_text("resume: false")

    def test_takeover_resumes_existing_arm_then_runs_all_remaining_stages(self):
        output = guardian.PROJECT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output, prefix="guard-test-") as directory:
            root = Path(directory)
            self.fixture(root)
            commands = []

            class FinishedProcess:
                pid = 999997

                def __init__(self, command, **kwargs):
                    commands.append(command)
                    if "--recover-train" in command:
                        arm = command[command.index("--recover-train") + 1]
                        mode = "train"
                    else:
                        mode = command[command.index("--mode") + 1]
                        arm = command[command.index("--arm") + 1] if "--arm" in command else None
                    if arm:
                        (root / arm).mkdir(exist_ok=True)
                    if mode == "train":
                        guardian.save(root / arm / "split.json", {"sha256": "identical"})
                        guardian.save(root / arm / "result.json", {"ok": True})
                    elif mode == "evaluate":
                        guardian.save(root / arm / "evaluation.json", {"ok": True})
                    else:
                        metrics = {arm: {"real": {"CIDEr_x100": 100.0, "BLEU4": .2, "ROUGE_L": .4},
                                   "brain_dependence_shuffle_CIDEr_x100": 20.0}
                                   for arm in ("baseline", "functional", "random")}
                        guardian.save(root / "summary.json", {"metrics": metrics, "paired_comparisons": {}})

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            with patch.object(guardian, "alive", return_value=False), \
                 patch.object(guardian, "gpu_is_available", return_value=True), \
                 patch.object(guardian.subprocess, "Popen", FinishedProcess), \
                 patch.object(guardian.time, "sleep"):
                guardian.guard(root)
            self.assertEqual(len(commands), 7)
            self.assertIn("--recover-train", commands[0])
            self.assertEqual(commands[1][commands[1].index("--mode") + 1], "train")
            self.assertEqual(json.loads((root / "status.json").read_text())["state"], "complete")
            self.assertEqual(json.loads((root / "recovery_attempts.json").read_text()), {"train_baseline": 1})
            self.assertTrue((root / "RESULTS.md").exists())

    def test_configuration_error_stops_without_retry(self):
        output = guardian.PROJECT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output, prefix="guard-test-") as directory:
            root = Path(directory)
            self.fixture(root, "ValueError: wrong mask order")
            with patch.object(guardian, "alive", return_value=False), \
                 patch.object(guardian, "gpu_is_available", return_value=True), \
                 patch.object(guardian.subprocess, "Popen") as launch:
                guardian.guard(root)
            launch.assert_not_called()
            self.assertEqual(json.loads((root / "guardian.json").read_text())["state"], "needs_attention")

    def test_recovery_limit_prevents_endless_restarts(self):
        output = guardian.PROJECT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output, prefix="guard-test-") as directory:
            root = Path(directory)
            self.fixture(root)
            guardian.save(root / "recovery_attempts.json", {"train_baseline": 2})
            with patch.object(guardian, "alive", return_value=False), \
                 patch.object(guardian.subprocess, "Popen") as launch:
                guardian.guard(root)
            launch.assert_not_called()
            self.assertEqual(json.loads((root / "guardian.json").read_text())["reason"], "recovery limit reached")


if __name__ == "__main__":
    unittest.main()
