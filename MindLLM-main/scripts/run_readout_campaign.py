"""Queue the readout ablation after a dependency; hourly records, bounded recovery."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
ARMS = ('baseline', 'adaptive', 'mlp_control')
PHASES = [f'smoke_{a}' for a in ARMS] + [f'train_{a}' for a in ARMS] + [f'evaluate_{a}' for a in ARMS] + ['compare']


def environment():
    env = os.environ.copy()
    java = '/mnt/data2/yjs2026/jdk/jdk-11.0.32.1+1'
    env.update(JAVA_HOME=java, WANDB_MODE='disabled', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', PYTHONFAULTHANDLER='1')
    env['PATH'] = java + '/bin:' + env.get('PATH', '')
    return env


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def alive(pid, root):
    try:
        proc = Path('/proc') / str(int(pid))
        command = (proc / 'cmdline').read_bytes().decode(errors='replace')
        return (str(root) in command and 'run_readout_campaign.py' in command
                and (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z')
    except (OSError, ValueError, TypeError):
        return False


def terminate(pid, root):
    if alive(pid, root) and os.getpgid(pid) == pid:
        os.killpg(pid, signal.SIGTERM)
        for _ in range(10):
            if not alive(pid, root):
                return
            time.sleep(1)
        if alive(pid, root):
            os.killpg(pid, signal.SIGKILL)


def snapshot(root, status):
    metrics = {}
    for arm in ARMS:
        paths = list((root / arm / 'csv').glob('*/metrics.csv'))
        latest = {}
        if paths:
            with max(paths, key=lambda p: p.stat().st_mtime).open() as stream:
                for row in csv.DictReader(stream):
                    if row.get('train/loss'):
                        latest['training'] = {k: row.get(k) for k in ('epoch', 'step', 'train/loss')}
                    if row.get('val/token_loss'):
                        latest['validation'] = {k: row.get(k) for k in ('epoch', 'step', 'val/token_loss')}
        metrics[arm] = latest
    return {'checked_unix': time.time(), 'state': status.get('state'), 'phase': status.get('phase'),
            'error': status.get('error'), 'metrics': metrics, 'completed': status.get('completed', [])}


def hourly(root, status, force=False):
    path = root / 'hourly_snapshot.json'
    if not force and path.exists() and time.time() - path.stat().st_mtime < 3600:
        return
    value = snapshot(root, status)
    if status.get('state') == 'waiting':
        dependency = Path(status['after'])
        inspector = dependency.parents[1] / 'scripts/inspect_cluster_ablation.py'
        try:
            checked = subprocess.run(['/usr/bin/python3', str(inspector), '--run-dir', str(dependency),
                '--write-progress', '--ensure-guardian'], capture_output=True, text=True, timeout=30)
            value['dependency'] = json.loads(checked.stdout) if checked.returncode == 0 else {'error': checked.stderr[-1000:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            value['dependency'] = {'error': str(error)}
    save(path, value)
    with (root / 'hourly_progress.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')


def configuration(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(PROJECT / 'config'), version_base=None):
        cfg = compose(config_name='readout_' + args.arm)
    OmegaConf.set_struct(cfg, False)
    run = args.output / (f'smoke_{args.arm}' if args.mode == 'smoke' else args.arm)
    cfg.output_dir, cfg.checkpoint = str(run), str(PROJECT / 'ckpts/MindLLM/mindllm-base.ckpt')
    cfg.seed, cfg.lr, cfg.stage, cfg.resume = 42, 1e-4, 'fit', args.resume
    cfg.test_after_fit, cfg.lm_checkpointing = False, 'off'
    cfg.data.split_seed, cfg.data.num_workers = 42, 0
    cfg.data.mixup, cfg.data.group_by_coco = False, True
    cfg.data.batch_size = 4
    cfg.trainer.accumulate_grad_batches = 8
    cfg.trainer.max_epochs, cfg.trainer.enable_progress_bar = args.epochs, False
    cfg.trainer.num_sanity_val_steps = 2
    if args.mode == 'smoke':
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches, cfg.trainer.limit_val_batches = 8, 2
    return cfg, run


def train(args):
    import torch
    from train_readout import preflight
    from main import main as train_mindllm
    cfg, run = configuration(args)
    cfg = preflight(cfg)
    train_mindllm(cfg)
    checkpoint = torch.load(run / 'coco-caption/checkpoints/last.ckpt', map_location='cpu')
    selected = next(v for k, v in checkpoint['callbacks'].items() if 'ModelCheckpoint' in k)
    save(run / 'result.json', {'arm': args.arm, 'epochs': int(checkpoint['epoch']) + 1,
        'global_step': int(checkpoint['global_step']), 'best_checkpoint': selected['best_model_path'],
        'best_val_loss': float(selected['best_model_score']),
        'max_memory_gib': torch.cuda.max_memory_allocated() / 1024**3})


def evaluate(args):
    from omegaconf import OmegaConf
    from train_readout import audit_checkpoint
    import scripts.run_cluster_ablation as experiment
    run = args.output / args.arm
    best = json.loads((run / 'result.json').read_text())['best_checkpoint']
    audit_checkpoint(best, args.arm, allow_baseline=False)
    experiment.configuration = lambda _: (OmegaConf.load(run / 'config.yaml'), run)
    args.seed = 42
    experiment.evaluate(args)


def compare(args):
    import numpy as np
    path = args.output / 'summary.json'
    evaluations = [json.loads((args.output / a / 'evaluation.json').read_text()) for a in ARMS]
    if any(e['keys'] != evaluations[0]['keys'] or e['references'] != evaluations[0]['references'] for e in evaluations[1:]):
        raise ValueError('test keys/references differ between arms')
    summary = {'exploratory_single_seed': True, 'metrics': {}, 'paired_comparisons': {}}
    for arm in ARMS:
        metrics = json.loads((args.output / arm / 'test_metrics.json').read_text())
        per_image = np.load(args.output / arm / 'per_image_cider.npz')
        metrics['brain_dependence_shuffle_CIDEr_x100'] = float((per_image['real'] - per_image['shuffle']).mean() * 100)
        summary['metrics'][arm] = metrics
    scores = {a: np.load(args.output / a / 'per_image_cider.npz')['real'] for a in ARMS}
    rng = np.random.default_rng(42)
    summary['paired_comparisons'] = {}
    for first, second in [('adaptive', 'baseline'), ('adaptive', 'mlp_control'), ('mlp_control', 'baseline')]:
        delta = scores[first] - scores[second]
        boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(2000)]
        summary['paired_comparisons'][first + '_minus_' + second] = {
            'delta_CIDEr_x100': float(delta.mean() * 100), 'CI95_x100': (np.quantile(boot, [.025, .975]) * 100).tolist()}
    save(path, summary)
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm', 'real_CIDEr_x100', 'BLEU4', 'ROUGE_L', 'shuffle_drop_CIDEr_x100'])
        for arm in ARMS:
            metrics = summary['metrics'][arm]
            real = metrics['real']
            writer.writerow([arm, real['CIDEr_x100'], real['BLEU4'], real['ROUGE_L'], metrics['brain_dependence_shuffle_CIDEr_x100']])
    lines = ['# Adaptive reread experiment', '', 'S1, seed 42; 60-epoch cap, validation early stopping.', '',
        '| Arm | CIDEr x100 | BLEU4 | ROUGE-L | Epochs |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        real = summary['metrics'][arm]['real']
        epochs = json.loads((args.output / arm / 'result.json').read_text())['epochs']
        lines.append(f"| {arm} | {real['CIDEr_x100']:.3f} | {real['BLEU4']:.4f} | {real['ROUGE_L']:.4f} | {epochs} |")
    lines += ['', 'Paired image bootstrap is conditional on this single seed and selected checkpoints.', '',
              json.dumps(summary['paired_comparisons'], indent=2)]
    (args.output / 'RESULTS.md').write_text('\n'.join(lines))


def progress_time(root, phase, since):
    mode, arm = phase.split('_', 1) if '_' in phase else (phase, None)
    paths = list((root / arm / 'csv').glob('*/metrics.csv')) if mode == 'train' else (
        list((root / arm).glob('predictions_*.json')) if mode == 'evaluate' else [])
    return max([since] + [p.stat().st_mtime for p in paths])


def campaign(args):
    root = args.output.resolve()
    root.relative_to(PROJECT / 'outputs')
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        campaign_locked(args, root)


def campaign_locked(args, root):
    protocol_path = root / 'protocol.json'
    if not protocol_path.exists():
        # Stdlib-only queue preparation; no GPU work before dependency completion.
        from scripts.run_cluster_ablation import file_hash
        files = ['scripts/run_readout_campaign.py', 'scripts/run_cluster_ablation.py', 'train_readout.py',
            'main.py', 'src/modules/adaptive_neuro_informed_attn.py', 'src/modules/neuro_informed_attn.py',
            'src/models/mindllm.py', 'src/dataset.py', 'src/dataset_cluster_ablation.py', 'src/utils/callbacks.py']
        files += [str(p.relative_to(PROJECT)) for p in (PROJECT / 'config').rglob('*.yaml')]
        save(protocol_path, {'source_sha256': {n: file_hash(PROJECT / n) for n in files},
            'after': str(args.after), 'arms': ARMS, 'max_epochs': args.epochs, 'max_hours': args.max_hours,
            'seed': 42, 'batch_size': 4, 'accumulate_grad_batches': 8, 'learning_rate': 1e-4,
            'early_stop_patience': 5, 'checkpoint': str(PROJECT / 'ckpts/MindLLM/mindllm-base.ckpt'),
            'checkpoint_sha256': file_hash(PROJECT / 'ckpts/MindLLM/mindllm-base.ckpt'),
            'selection': 'min val/token_loss', 'no_lora_or_auxiliary_losses': True,
            'single_seed_exploratory': True, 'hourly_records': True, 'retry_limit_per_stage': 2})
    protocol = json.loads(protocol_path.read_text())
    if protocol['max_epochs'] != args.epochs or protocol['after'] != str(args.after) or protocol.get('max_hours', args.max_hours) != args.max_hours:
        raise ValueError('queue configuration changed')
    path = root / 'status.json'
    status = json.loads(path.read_text()) if path.exists() else {'state': 'waiting', 'phase': 'wait_dependency',
        'after': str(args.after), 'completed': [], 'retries': {}}
    if status['state'] in ('complete', 'needs_attention', 'budget_exhausted'):
        return
    status['pid'] = os.getpid()
    save(path, status)
    while status['state'] == 'waiting':
        hourly(root, status)
        try:
            dependency = json.loads((args.after / 'status.json').read_text())
        except (OSError, ValueError):
            time.sleep(60)
            continue
        if dependency['state'] == 'complete':
            # Snapshot upstream results before starting a new GPU job.
            for name in ('summary.json', 'RESULTS.md', 'metrics.csv'):
                content = (args.after / name).read_bytes()
                (root / ('prior_' + name)).write_bytes(content)
            status['state'] = 'running'
            status['started_unix'] = time.time()
            save(path, status)
            hourly(root, status, True)
            break
        time.sleep(60)
    deadline = status['started_unix'] + args.max_hours * 3600
    from scripts.guard_cluster_ablation import gpu_is_available
    for phase in PHASES:
        if phase in status['completed']:
            continue
        mode, arm = phase.split('_', 1) if '_' in phase else (phase, None)
        marker = (root / ('smoke_' + arm if mode == 'smoke' else arm) / ('evaluation.json' if mode == 'evaluate' else 'result.json')) if arm else root / 'RESULTS.md'
        # A previous supervisor may have disconnected while its worker continued.
        while alive(status.get('child_pid'), root):
            hourly(root, status)
            time.sleep(60)
        if marker.exists():
            status['completed'].append(phase)
            save(path, status)
            continue
        while True:
            if time.time() >= deadline:
                status.update(state='budget_exhausted', error='48-hour active experiment budget exhausted')
                save(path, status)
                hourly(root, status, True)
                return
            for name, expected in protocol['source_sha256'].items():
                if digest(PROJECT / name) != expected:
                    raise ValueError('frozen source changed: ' + name)
            if not gpu_is_available():
                hourly(root, status)
                time.sleep(60)
                continue
            if mode == 'evaluate' and len({json.loads((root / a / 'split.json').read_text())['sha256'] for a in ARMS}) != 1:
                raise ValueError('split mismatch')
            count = status['retries'].get(phase, 0)
            command = ['taskset', '-c', '16-31', str(PROJECT / '.venv-repaired/bin/python'), '-u',
                str(Path(__file__).resolve()), '--mode', mode, '--output', str(root), '--epochs', str(args.epochs)]
            if arm:
                command += ['--arm', arm]
            if mode in ('train', 'smoke') and (root / ('smoke_' + arm if mode == 'smoke' else arm) / 'readout_manifest.json').exists():
                command += ['--resume']
            log_path = root / f'{phase}.attempt_{count}.log'
            status.update(state='running', phase=phase, phase_started_unix=time.time(), log=str(log_path), error=None)
            with log_path.open('a') as log:
                worker = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment())
                status['child_pid'] = worker.pid
                save(path, status)
                stalled = False
                while worker.poll() is None:
                    hourly(root, status)
                    if time.time() >= deadline or time.time() - progress_time(root, phase, status['phase_started_unix']) > 1800:
                        stalled = True
                        terminate(worker.pid, root)
                        break
                    time.sleep(30)
                code = worker.wait()
            if code == 0:
                status['completed'].append(phase)
                save(path, status)
                hourly(root, status, True)
                break
            text = log_path.read_text(errors='replace')[-12000:]
            recoverable = stalled or code in (-6, -7, -9, -11, 137, 139) or any(s in text for s in ('CUDA error', 'out of memory', 'Fatal Python error'))
            last = root / ('smoke_' + arm if mode == 'smoke' else arm or '') / 'coco-caption/checkpoints/last.ckpt'
            if not recoverable or count >= 2 or (mode in ('train', 'smoke') and not last.exists()):
                status.update(state='needs_attention', error=f'{phase} exit {code}; automatic recovery unavailable')
                save(path, status)
                hourly(root, status, True)
                return
            status['retries'][phase] = count + 1
            save(path, status)
    status.update(state='complete', finished_unix=time.time())
    save(path, status)
    hourly(root, status, True)


def inspect_queue(root):
    status = json.loads((root / 'status.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    if status['state'] in ('waiting', 'running') and not alive(status.get('pid'), root):
        with (root / 'campaign.log').open('a') as log:
            process = subprocess.Popen(['/usr/bin/python3', '-u', str(Path(__file__).resolve()),
                '--mode', 'campaign', '--output', str(root), '--after', protocol['after'],
                '--epochs', str(protocol['max_epochs']), '--max-hours', str(protocol['max_hours'])],
                cwd=PROJECT, env=environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['restarted_pid'] = process.pid
    hourly(root, status, True)
    print((root / 'hourly_snapshot.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['campaign', 'smoke', 'train', 'evaluate', 'compare', 'inspect'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--max-hours', type=float, default=48)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == 'campaign':
        if args.after is None:
            parser.error('--after required')
        args.after = args.after.resolve()
    elif args.mode not in ('compare', 'inspect') and not args.arm:
        parser.error('--arm required')
    try:
        if args.mode == 'campaign':
            campaign(args)
        elif args.mode == 'inspect':
            inspect_queue(args.output)
        elif args.mode in ('smoke', 'train'):
            train(args)
        elif args.mode == 'evaluate':
            evaluate(args)
        else:
            compare(args)
    except BaseException as error:
        if args.mode == 'campaign' and (args.output / 'status.json').exists():
            status = json.loads((args.output / 'status.json').read_text())
            status.update(state='needs_attention', error=str(error))
            save(args.output / 'status.json', status)
            hourly(args.output, status, True)
        raise
