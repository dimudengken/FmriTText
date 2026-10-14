"""Experiment health snapshot and optional progress log; never changes training."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time


def process_alive(pid, experiment):
    try:
        directory = Path("/proc") / str(int(pid))
        state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (directory / "cmdline").read_bytes().decode(errors="replace")
        return state != "Z" and str(experiment) in command and any(name in command for name in
            ("run_cluster_ablation.py", "run_cluster_ablation_continue.py", "guard_cluster_ablation.py"))
    except (OSError, ValueError, TypeError):
        return False


def tail(path, size=16384):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - size))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def inspect(root):
    now = time.time()
    status = json.loads((root / "status.json").read_text())
    phase = status.get("phase", "unknown")
    progress = float(status.get("phase_started_unix", status.get("started_unix", now)))
    metrics, nonfinite = {}, False
    for arm in ("baseline", "functional", "random"):
        candidates = list((root / arm / "csv").glob("*/metrics.csv"))
        path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else root / arm / "csv/metrics/metrics.csv"
        latest = {"training": None, "validation": None}
        if path.exists():
            if phase == "train_" + arm:
                progress = max(progress, path.stat().st_mtime)
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    for field, label in (("train/loss", "training"), ("val/token_loss", "validation")):
                        raw = row.get(field)
                        if raw:
                            try:
                                value = float(raw)
                            except ValueError:
                                continue
                            finite = math.isfinite(value)
                            nonfinite |= not finite
                            latest[label] = {"epoch": row.get("epoch"), "step": row.get("step"),
                                             "loss": value if finite else None}
        metrics[arm] = latest
        if phase == "evaluate_" + arm:
            for path in (root / arm).glob("predictions_*.json"):
                progress = max(progress, path.stat().st_mtime)
            path = root / arm / "test_metrics.json"
            if path.exists():
                progress = max(progress, path.stat().st_mtime)
    log = Path(status.get("log", root / "campaign.log"))
    text = tail(log)
    errors = [line[-500:] for line in text.splitlines() if re.search(
        r"Traceback|CUDA error|out of memory|Segmentation fault|Fatal Python error|RuntimeError|ValueError", line, re.I)]
    summary = None
    if (root / "summary.json").exists():
        saved = json.loads((root / "summary.json").read_text())
        summary = {"comparison": saved.get("paired_comparisons", {}).get("functional_minus_baseline"),
                   "exploratory_single_seed": saved.get("exploratory_single_seed", True)}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    return {"version": 1, "checked_unix": now, "state": status.get("state", "unknown"),
            "phase": phase, "error": status.get("error"), "log": str(log),
            "supervisor_pid": status.get("pid"), "worker_pid": status.get("child_pid"),
            "supervisor_alive": process_alive(status.get("pid"), root),
            "worker_alive": process_alive(status.get("child_pid"), root),
            "progress_unix": progress, "progress_age_seconds": max(0, now - progress),
            "metrics": metrics, "nonfinite_loss": nonfinite, "fatal_patterns": errors,
            "gpu": gpu.stdout.strip(), "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-progress", action="store_true")
    parser.add_argument("--ensure-guardian", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    snapshot = inspect(root)
    if args.ensure_guardian:
        from guard_cluster_ablation import ensure_guardian
        snapshot["guardian"] = ensure_guardian(root)
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
    if args.write_progress:
        temporary = root / "monitor_snapshot.json.tmp"
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(root / "monitor_snapshot.json")
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +08:00")
        arm = snapshot["phase"].removeprefix("train_")
        metrics = snapshot["metrics"].get(arm, {})
        training = metrics.get("training") or {}
        validation = metrics.get("validation") or {}
        line = (f"{stamp} state={snapshot['state']} phase={snapshot['phase']} "
                f"epoch_index={training.get('epoch', '?')} step={training.get('step', '?')} "
                f"train_loss={training.get('loss', '?')} val_loss={validation.get('loss', '?')} "
                f"worker_alive={snapshot['worker_alive']} gpu={snapshot['gpu']}\n")
        with (root / "progress.log").open("a", encoding="utf-8") as stream:
            stream.write(line)
    print(encoded)


if __name__ == "__main__":
    main()
