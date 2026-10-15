"""Continue cluster arms until validation early stopping; retain stopped baseline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_cluster_ablation import file_hash, save_json


def prepare(args):
    from scripts.run_cluster_ablation_continue import prepare as clone
    clone(args)
    root, source = args.output.resolve(), args.source.resolve()
    # Reuse the already early-stopped baseline, including its test predictions.
    for name in ('result.json', 'evaluation.json', 'test_metrics.json', 'per_image_cider.npz',
                 'predictions_real.json', 'predictions_zero.json', 'predictions_shuffle.json'):
        shutil.copy2(source / 'baseline' / name, root / 'baseline' / name)
    result_path = root / 'baseline/result.json'
    result = json.loads(result_path.read_text())
    result['best_checkpoint'] = result['best_checkpoint'].replace(str(source), str(root))
    result['reused_from'] = str(source)
    result['stop_reason'] = 'validation_early_stop'
    save_json(result_path, result)
    protocol = json.loads((root / 'protocol.json').read_text())
    protocol.update(require_early_stop=True, train_arms=['functional', 'random'],
        baseline_policy='reuse previously early-stopped baseline and its fixed test outputs',
        epoch_limit_policy='1000 is a safety ceiling; refuse testing unless validation early stopping occurred',
        note='Post-hoc continuation after the 60-epoch results; original early-stop counters and optimizer preserved.')
    protocol['source_sha256']['scripts/run_cluster_until_early_stop.py'] = file_hash(Path(__file__).resolve())
    save_json(root / 'protocol.json', protocol)


def campaign(args):
    from scripts.guard_cluster_ablation import gpu_is_available, write_results, terminate_own_worker
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if (root / 'status.json').exists():
        raise ValueError('campaign already launched')
    if not gpu_is_available():
        raise RuntimeError('GPU occupied')
    status = {'pid': os.getpid(), 'state': 'running', 'started_unix': time.time(),
              'completed': ['train_baseline', 'evaluate_baseline']}
    deadline = time.monotonic() + protocol['max_hours'] * 3600
    phases = ['train_functional', 'train_random', 'evaluate_functional', 'evaluate_random', 'compare']
    try:
        for phase in phases:
            for name, expected in protocol['source_sha256'].items():
                if file_hash(PROJECT / name) != expected:
                    raise ValueError('frozen source changed: ' + name)
            mode, arm = phase.split('_', 1) if '_' in phase else (phase, None)
            if mode == 'train':
                command = [sys.executable, '-u', str(PROJECT / 'scripts/guard_cluster_ablation.py'),
                    '--run-dir', str(root), '--recover-train', arm, '--attempt', '0']
            else:
                for cluster_arm in ('functional', 'random'):
                    result = json.loads((root / cluster_arm / 'result.json').read_text())
                    if result.get('stop_reason') != 'validation_early_stop':
                        raise ValueError('testing blocked: clustering did not reach validation early stopping')
                if len({json.loads((root / a / 'split.json').read_text())['sha256'] for a in ('baseline','functional','random')}) != 1:
                    raise ValueError('split mismatch')
                command = [sys.executable, '-u', str(PROJECT / 'scripts/run_cluster_ablation.py'),
                    '--mode', mode, '--output', str(root), '--epochs', str(protocol['max_epochs']),
                    '--seed', str(protocol['seed']), '--patience', str(protocol['early_stop_patience']),
                    '--batch-size', str(protocol['batch_size']), '--accumulate', str(protocol['accumulate_grad_batches']),
                    '--init-checkpoint', protocol['initial_checkpoint'], '--feature-bank', protocol['feature_bank']]
                if arm:
                    command += ['--arm', arm]
            log_path = root / (phase + '.log')
            status.update(phase=phase, phase_started_unix=time.time(), command=command, log=str(log_path))
            save_json(root / 'status.json', status)
            with log_path.open('w') as log:
                worker = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status['child_pid'] = worker.pid
                save_json(root / 'status.json', status)
                try:
                    code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    terminate_own_worker(worker.pid, root)
                    worker.wait()
                    raise RuntimeError('time budget exhausted before validation early stopping; no final test')
            if code:
                raise RuntimeError(f'{phase} failed with exit code {code}')
            status['completed'].append(phase)
            save_json(root / 'status.json', status)
        write_results(root)
        status.update(state='complete', finished_unix=time.time())
    except BaseException as error:
        status.update(state='failed', error=str(error), finished_unix=time.time())
        raise
    finally:
        save_json(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare','campaign'], required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.source is None:
            parser.error('--source required')
        prepare(args)
    else:
        campaign(args)
