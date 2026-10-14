"""Continue a completed ablation in a separate directory, preserving optimizer/RNG."""
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
from scripts.run_cluster_ablation import ARMS, file_hash, save_json


def relocate(value, old, new):
    if isinstance(value, str):
        return value.replace(str(old), str(new))
    if isinstance(value, dict):
        return {relocate(k, old, new): relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, old, new) for v in value]
    return value


def prepare(args):
    import torch
    from omegaconf import OmegaConf
    source, root = args.source.resolve(), args.output.resolve()
    root.relative_to(PROJECT / 'outputs')
    source.relative_to(PROJECT / 'outputs')
    if root.exists():
        raise ValueError('continuation output already exists')
    if json.loads((source / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('source campaign is not complete')
    protocol = json.loads((source / 'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        if file_hash(PROJECT / name) != digest:
            raise ValueError('source changed: ' + name)
    root.mkdir(parents=True)
    for name in ('functional.pt', 'random.pt'):
        shutil.copy2(source / name, root / name)
    records = {}
    for arm in ARMS:
        old, new = source / arm, root / arm
        new.mkdir()
        original = old / 'coco-caption/checkpoints/last.ckpt'
        checkpoint = torch.load(original, map_location='cpu')
        if checkpoint['epoch'] + 1 >= args.epochs:
            raise ValueError('new epoch limit must exceed saved epoch')
        if not checkpoint.get('optimizer_states') or not checkpoint.get('rng_state'):
            raise ValueError('full optimizer/RNG state required')
        callbacks = checkpoint['callbacks']
        model_key = next(k for k in callbacks if 'ModelCheckpoint' in k)
        record = callbacks[model_key]
        # Keep the prior validation winner eligible and relocate callback bookkeeping.
        paths = {record['best_model_path'], record.get('kth_best_model_path', '')}
        paths.update(record.get('best_k_models', {}).keys())
        destination = new / 'coco-caption/checkpoints'
        destination.mkdir(parents=True)
        for path in paths:
            if path:
                candidate = Path(path)
                relative = candidate.relative_to(old)
                shutil.copy2(candidate, new / relative)
        callbacks[model_key] = relocate(record, old, new)
        torch.save(checkpoint, destination / 'last.ckpt')
        del checkpoint
        cfg = OmegaConf.create(relocate(OmegaConf.to_container(OmegaConf.load(old / 'config.yaml'), resolve=True), source, root))
        cfg.resume = True
        cfg.trainer.max_epochs = args.epochs
        cfg.trainer.logger.version = 'continuation'
        (new / 'config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        shutil.copy2(old / 'split.json', new / 'split.json')
        shutil.copytree(old / 'csv', new / 'csv')
        records[arm] = {'source_last_checkpoint': str(original),
                        'source_last_sha256': file_hash(original),
                        'source_result': json.loads((old / 'result.json').read_text()),
                        'resume_checkpoint': str(destination / 'last.ckpt'),
                        'resume_sha256': file_hash(destination / 'last.ckpt')}
    protocol.update(parent_output=str(source), max_epochs=args.epochs, max_hours=args.max_hours,
        continuation=records, policy='sequential continuation with bounded guardian recovery',
        posthoc=True, note='Epoch extension chosen after inspecting the 20-epoch results; same seed/split/settings, not independent replication.')
    for name in ('scripts/run_cluster_ablation_continue.py', 'scripts/guard_cluster_ablation.py'):
        protocol['source_sha256'][name] = file_hash(PROJECT / name)
    save_json(root / 'protocol.json', protocol)


def campaign(args):
    from scripts.guard_cluster_ablation import PHASES, gpu_is_available, write_results
    root = args.output.resolve()
    protocol = json.loads((root / 'protocol.json').read_text())
    if (root / 'status.json').exists():
        raise ValueError('continuation has already been launched')
    if not gpu_is_available():
        raise RuntimeError('GPU occupied; refusing to start another job')
    status = {'pid': os.getpid(), 'state': 'running', 'completed': [], 'started_unix': time.time()}
    deadline = time.monotonic() + protocol['max_hours'] * 3600
    try:
        for phase in PHASES:
            for name, digest in protocol['source_sha256'].items():
                if file_hash(PROJECT / name) != digest:
                    raise RuntimeError('frozen source changed: ' + name)
            mode, arm = phase.split('_', 1) if '_' in phase else (phase, None)
            if mode == 'train':
                command = [sys.executable, '-u', str(PROJECT / 'scripts/guard_cluster_ablation.py'),
                           '--run-dir', str(root), '--recover-train', arm, '--attempt', '0']
            else:
                command = [sys.executable, '-u', str(PROJECT / 'scripts/run_cluster_ablation.py'),
                    '--output', str(root), '--mode', mode, '--epochs', str(protocol['max_epochs']),
                    '--patience', str(protocol['early_stop_patience']), '--seed', str(protocol['seed']),
                    '--batch-size', str(protocol['batch_size']), '--accumulate', str(protocol['accumulate_grad_batches']),
                    '--init-checkpoint', protocol['initial_checkpoint'], '--feature-bank', protocol['feature_bank']]
                if arm:
                    command += ['--arm', arm]
            if mode == 'evaluate':
                if len({json.loads((root / a / 'split.json').read_text())['sha256'] for a in ARMS}) != 1:
                    raise RuntimeError('split mismatch')
            log_path = root / (phase + '.log')
            status.update(phase=phase, phase_started_unix=time.time(), command=command, log=str(log_path))
            save_json(root / 'status.json', status)
            with log_path.open('w') as log:
                worker = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status['child_pid'] = worker.pid
                save_json(root / 'status.json', status)
                try:
                    code = worker.wait(timeout=max(1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    from scripts.guard_cluster_ablation import terminate_own_worker
                    terminate_own_worker(worker.pid, root)
                    worker.wait()
                    raise RuntimeError('continuation time budget exhausted')
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
    parser.add_argument('--mode', choices=['prepare', 'campaign'], required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.source is None:
            parser.error('--source required for preparation')
        prepare(args)
    else:
        campaign(args)
