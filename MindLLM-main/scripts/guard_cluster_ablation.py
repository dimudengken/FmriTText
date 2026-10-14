"""Watch the existing campaign; recover infrastructure failures without retuning."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
PHASES = [f"train_{arm}" for arm in ("baseline", "functional", "random")] + [
    f"evaluate_{arm}" for arm in ("baseline", "functional", "random")] + ["compare"]


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def alive(pid, root):
    try:
        proc = Path("/proc") / str(int(pid))
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (proc / "cmdline").read_bytes().decode(errors="replace")
        return state != "Z" and str(root) in command and any(
            name in command for name in ("run_cluster_ablation.py", "run_cluster_ablation_continue.py", "guard_cluster_ablation.py"))
    except (OSError, ValueError, TypeError):
        return False


def ensure_guardian(root):
    path = root / "guardian.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if alive(saved.get("pid"), root) or saved.get("state") in ("complete", "needs_attention", "budget_exhausted"):
            return saved
    status = json.loads((root / "status.json").read_text())
    if status["state"] == "complete":
        return {"state": "complete", "recoveries_total": 0}
    with (root / "guardian.log").open("a") as log:
        process = subprocess.Popen(["/usr/bin/python3", "-u", str(Path(__file__).resolve()),
            "--run-dir", str(root)], cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return {"pid": process.pid, "state": "starting"}


def event(root, kind, **details):
    with (root / "guard_events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time_unix": time.time(), "kind": kind, **details}) + "\n")


def progress_time(root, status):
    latest = status.get("phase_started_unix", status["started_unix"])
    phase = status.get("phase", "")
    arm = phase.split("_", 1)[-1]
    if phase.startswith("train_"):
        paths = (root / arm / "csv").glob("*/metrics.csv")
    elif phase.startswith("evaluate_"):
        paths = (root / arm).glob("predictions_*.json")
    else:
        paths = []
    for path in paths:
        latest = max(latest, path.stat().st_mtime)
    return latest


def terminate_own_worker(pid, root):
    if alive(pid, root) and os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(10):
            if not alive(pid, root):
                return
            time.sleep(1)
        if alive(pid, root):
            os.killpg(pid, signal.SIGKILL)


def recoverable(root, status):
    log = Path(status.get("log", root / "campaign.log"))
    text = log.read_text(errors="replace")[-12000:] if log.exists() else ""
    combined = str(status.get("error", "")) + text
    return bool(re.search(r"exit code -(?:6|7|9|11)|exit code (?:137|139)|CUDA error|out of memory|"
                          r"segmentation|Fatal Python error|killed|supervisor disappeared|stalled", combined, re.I))


def finished(root, phase, since):
    if phase.startswith("train_"):
        path = root / phase.removeprefix("train_") / "result.json"
    elif phase.startswith("evaluate_"):
        path = root / phase.removeprefix("evaluate_") / "evaluation.json"
    else:
        path = root / "summary.json"
    return path.exists() and path.stat().st_mtime >= since


def gpu_is_available():
    try:
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=10)
        memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10)
        return (processes.returncode == 0 and memory.returncode == 0
                and not any(line.strip().isdigit() for line in processes.stdout.splitlines())
                and all(int(line.strip()) < 1024 for line in memory.stdout.splitlines() if line.strip()))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def write_results(root):
    path = root / "summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    lines = ["# Cluster ablation results", "", "Single subject / seed 42; exploratory comparison.", "",
             "| Arm | Test CIDEr x100 | BLEU-4 | ROUGE-L | Shuffle drop x100 |",
             "|---|---:|---:|---:|---:|"]
    for arm in ("baseline", "functional", "random"):
        metrics = summary["metrics"][arm]
        real = metrics["real"]
        lines.append(f"| {arm} | {real['CIDEr_x100']:.3f} | {real['BLEU4']:.4f} | "
                     f"{real['ROUGE_L']:.4f} | {metrics['brain_dependence_shuffle_CIDEr_x100']:.3f} |")
    lines += ["", "## Paired image bootstrap", ""]
    for label, result in summary["paired_comparisons"].items():
        low, high = result["CI95_x100"]
        lines.append(f"- {label}: delta={result['delta_CIDEr_x100']:.3f}, CI95=[{low:.3f}, {high:.3f}]")
    lines += ["", "These intervals are conditional on this training seed and selected checkpoints.",
              "Training, validation, prediction files, failure logs and recovery records are retained.",
              "Inspect summary.json, metrics.csv, protocol.json, launch.json and guard_events.jsonl.", ""]
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    if protocol.get("parent_output"):
        previous = json.loads((Path(protocol["parent_output"]) / "summary.json").read_text())
        lines += ["", "## Continuation versus 20 epochs", "",
                  "Post-hoc extension of the same training runs; not an independent replication.", "",
                  "| Arm | Previous CIDEr x100 | Continued CIDEr x100 | Change |", "|---|---:|---:|---:|"]
        for arm in ("baseline", "functional", "random"):
            before = previous["metrics"][arm]["real"]["CIDEr_x100"]
            after = summary["metrics"][arm]["real"]["CIDEr_x100"]
            lines.append(f"| {arm} | {before:.3f} | {after:.3f} | {after - before:+.3f} |")
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def recover_train(root, arm, attempt):
    import torch
    from omegaconf import OmegaConf
    from main import main as train_mindllm
    cfg = OmegaConf.load(root / arm / "config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.resume = True
    cfg.trainer.logger.version = "continuation" if attempt == 0 else f"recovery_{attempt}"
    run = root / arm
    checkpoint = run / "coco-caption/checkpoints/last.ckpt"
    resumed = str(checkpoint) if checkpoint.exists() else None
    (run / f"recovery_config_{attempt}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    event(root, "training_resume", arm=arm, attempt=attempt, checkpoint=resumed,
          note="same optimizer, RNG callback, split, model and max_epochs; restart from base only if no checkpoint exists")
    train_mindllm(cfg)
    saved = torch.load(checkpoint, map_location="cpu")
    callbacks = [value for key, value in saved["callbacks"].items() if "ModelCheckpoint" in key]
    selected = callbacks[0]
    if not Path(selected["best_model_path"]).is_file():
        raise RuntimeError("best checkpoint missing after recovery")
    save(run / "result.json", {"arm": arm, "mode": "train", "recovery_attempt": attempt,
        "resumed_checkpoint": resumed, "best_checkpoint": selected["best_model_path"],
        "best_val_loss": float(selected["best_model_score"]), "epochs": int(saved["epoch"]) + 1,
        "global_step": int(saved["global_step"]), "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3})


def guard(root):
    root.relative_to(PROJECT / "outputs")
    with (root / "guardian.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        return guard_locked(root)


def guard_locked(root):
    protocol = json.loads((root / "protocol.json").read_text())
    initial = json.loads((root / "status.json").read_text())
    deadline = initial["started_unix"] + protocol["max_hours"] * 3600
    ledger_path = root / "recovery_attempts.json"
    attempts = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    info = {"pid": os.getpid(), "state": "watching", "recoveries_total": sum(attempts.values()),
            "interval_seconds": 60, "max_attempts_per_stage": 2, "stall_minutes": 30}
    save(root / "recovery_policy.json", {**info, "same_training_configuration": True,
        "preserve_original_logs": True, "deadline_unix": deadline,
        "unrecoverable_errors": "stop and notify; never change batch size, loss, split or learning rate"})
    while time.time() < deadline:
        status = json.loads((root / "status.json").read_text())
        info.update(checked_unix=time.time(), phase=status.get("phase"))
        save(root / "guardian.json", info)
        if status["state"] == "complete":
            write_results(root)
            info["state"] = "complete"
            save(root / "guardian.json", info)
            return
        if status["state"] == "running" and alive(status.get("pid"), root):
            if alive(status.get("child_pid"), root) and time.time() - progress_time(root, status) > 1800:
                event(root, "stalled_worker", phase=status["phase"], pid=status["child_pid"])
                info["last_stalled_phase"] = status["phase"]
                terminate_own_worker(status["child_pid"], root)
            time.sleep(60)
            continue
        if alive(status.get("child_pid"), root):
            time.sleep(60)
            continue
        phase = status.get("phase")
        if info.get("last_stalled_phase") == phase:
            status["error"] = str(status.get("error", "")) + "; stalled worker terminated by guardian"
        if phase not in PHASES:
            info.update(state="needs_attention", reason="unexpected preparation failure")
            save(root / "guardian.json", info)
            return
        if status["state"] != "failed":
            status["error"] = "supervisor disappeared"
        complete = set(status.get("completed", []))
        if finished(root, phase, status["phase_started_unix"]):
            complete.add(phase)
        elif not recoverable(root, status):
            info.update(state="needs_attention", reason=status.get("error"))
            save(root / "guardian.json", info)
            event(root, "unrecoverable", phase=phase, reason=status.get("error"))
            return
        if phase not in complete and attempts.get(phase, 0) >= 2:
            info.update(state="needs_attention", reason="recovery limit reached")
            save(root / "guardian.json", info)
            return
        if not gpu_is_available():
            info["state"] = "waiting_for_gpu"
            save(root / "guardian.json", info)
            time.sleep(60)
            continue
        info["state"] = "watching"
        event(root, "takeover", phase=phase, previous_supervisor=status.get("pid"))
        status.update(pid=os.getpid(), state="running", error=None)
        environment = os.environ.copy()
        java = Path("/mnt/data2/yjs2026/jdk/jdk-11.0.32.1+1")
        environment.update(JAVA_HOME=str(java), WANDB_MODE="disabled", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false", PYTHONFAULTHANDLER="1")
        environment["PATH"] = str(java / "bin") + os.pathsep + environment.get("PATH", "")
        for current in PHASES:
            if current in complete:
                continue
            for name, digest in protocol["source_sha256"].items():
                if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError("frozen experiment source changed: " + name)
            attempt = attempts.get(current, 0) + 1
            if attempt > 2:
                info.update(state="needs_attention", reason="recovery limit reached")
                save(root / "guardian.json", info)
                return
            # Only retries consume the limit. Later unstarted stages are normal runs.
            retry = current == phase
            if retry:
                attempts[current] = attempt
                save(ledger_path, attempts)
                info["recoveries_total"] = sum(attempts.values())
                info.pop("last_stalled_phase", None)
            mode, arm = (current.split("_", 1) + [None])[:2] if current != "compare" else ("compare", None)
            if mode == "evaluate":
                splits = {json.loads((root / name / "split.json").read_text())["sha256"]
                          for name in ("baseline", "functional", "random")}
                if len(splits) != 1:
                    raise RuntimeError("training arms used different splits")
            if mode == "train" and (root / arm / "config.yaml").exists():
                command = [str(PROJECT / ".venv-repaired/bin/python"), "-u", str(Path(__file__).resolve()),
                           "--run-dir", str(root), "--recover-train", arm, "--attempt", str(attempt)]
            else:
                command = [str(PROJECT / ".venv-repaired/bin/python"), "-u", str(PROJECT / "scripts/run_cluster_ablation.py"),
                           "--mode", mode, "--output", str(root), "--seed", str(protocol["seed"]),
                           "--init-checkpoint", protocol["initial_checkpoint"],
                           "--feature-bank", protocol["feature_bank"], "--epochs", str(protocol["max_epochs"]),
                           "--batch-size", str(protocol["batch_size"]),
                           "--accumulate", str(protocol["accumulate_grad_batches"]),
                           "--patience", str(protocol["early_stop_patience"])]
                if arm:
                    command += ["--arm", arm]
            command = ["taskset", "-c", "16-31", *command]
            log_path = root / f"{current}.guard_{attempt}.log"
            status.update(phase=current, phase_started_unix=time.time(), completed=list(complete), log=str(log_path))
            with log_path.open("a") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status["child_pid"] = process.pid
                save(root / "status.json", status)
                event(root, "stage_started", phase=current, retry=retry, attempt=attempt, pid=process.pid)
                while process.poll() is None and time.time() < deadline:
                    info.update(phase=current, checked_unix=time.time())
                    save(root / "guardian.json", info)
                    if time.time() - progress_time(root, status) > 1800:
                        info["last_stalled_phase"] = current
                        event(root, "stalled_worker", phase=current, pid=process.pid)
                        terminate_own_worker(process.pid, root)
                    time.sleep(30)
                if process.poll() is None:
                    terminate_own_worker(process.pid, root)
                code = process.wait()
            if code:
                status.update(state="failed", error=f"{current} failed with exit code {code}")
                save(root / "status.json", status)
                event(root, "stage_failed", phase=current, exit_code=code)
                break
            complete.add(current)
            status["completed"] = list(complete)
            save(root / "status.json", status)
        else:
            status.update(state="complete", finished_unix=time.time())
            save(root / "status.json", status)
        time.sleep(10)
    info["state"] = "budget_exhausted"
    save(root / "guardian.json", info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--recover-train", choices=("baseline", "functional", "random"))
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--ensure", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.ensure:
        print(json.dumps(ensure_guardian(root)))
    elif args.recover_train:
        recover_train(root, args.recover_train, args.attempt)
    else:
        try:
            guard(root)
        except BaseException as error:
            event(root, "guardian_error", error=str(error))
            save(root / "guardian.json", {"pid": os.getpid(), "state": "needs_attention", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
