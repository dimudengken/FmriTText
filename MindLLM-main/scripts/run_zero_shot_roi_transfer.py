"""Source-only GMM functional groups, anatomical label transfer, target inference."""
import argparse
import csv
from contextlib import ExitStack
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
from scripts.run_cluster_ablation import save_json, file_hash
ARMS = ('baseline', 'functional')
PHASES = ['prepare'] + [f'smoke_{a}' for a in ARMS] + [f'train_{a}' for a in ARMS] + [f'evaluate_{a}' for a in ARMS] + ['compare']


def metadata(subject):
    import nibabel as nib
    import numpy as np
    folder = PROJECT / f'data/roi/subj{subject:02d}'
    image = nib.load(folder / 'nsdgeneral.nii.gz')
    mask = image.get_fdata().astype('int64') > 0
    indices = np.column_stack(np.where(mask))
    coords = nib.affines.apply_affine(image.affine, indices)
    # NSD native volumes are centered, RAS-oriented here. Do not call this MNI.
    if nib.aff2axcodes(image.affine) != ('R', 'A', 'S'):
        raise ValueError('unexpected anatomical axis convention')
    midpoint = nib.affines.apply_affine(image.affine, (np.array(mask.shape)-1)/2)[0]
    rois = np.column_stack([nib.load(folder/(name+'.nii.gz')).get_fdata()[mask].astype('int64')
                           for name in ('HCP_MMP1','Kastner2015','prf-visualrois')])
    return {'coords': coords, 'hemisphere': (coords[:,0] >= midpoint).astype('int64'),
            'rois': rois, 'mask_shape': list(mask.shape), 'flat_indices': np.flatnonzero(mask.ravel())}


def prepare(args):
    import h5py
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from src.roi_cluster_transfer import frozen_split, transfer_labels
    from src.voxel_clustering import by_coco, training_responses, encoding_features
    root = args.output
    source_ids = {}
    with ExitStack() as stack:
        handles = {sid: stack.enter_context(h5py.File(PROJECT/f'data/nsd0/subject_{sid}.h5','r')) for sid in args.sources}
        for sid, handle in handles.items():
            source_ids[f'subject_{sid}'] = {part: sorted(by_coco(handle[part])) for part in ('train','test')}
        # Target test IMAGE IDs only for leakage exclusion. No target train group or responses.
        with h5py.File(PROJECT/f'data/nsd0/subject_{args.target}.h5','r') as handle:
            target_test = sorted(by_coco(handle['test']))
        partitions, excluded_test, excluded_val = frozen_split(source_ids, target_test, 900, args.seed)
        partitions[f'subject_{args.target}'] = {'train': [], 'val': [], 'test': target_test}
        manifest = {'sources': args.sources, 'target': args.target, 'subjects': partitions,
                    'excluded_test_ids': excluded_test, 'source_validation_ids': excluded_val}
        manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        bank = torch.load(args.feature_bank, map_location='cpu', weights_only=True)
        if bank.get('modality', 'image') != 'image':
            raise ValueError('requires an image CLIP bank')
        ids = list(map(int, bank['coco_ids'].tolist()))
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate feature bank image IDs')
        index = {image:i for i,image in enumerate(ids)}
        clip = bank['embeddings'].float().numpy()
        entries, features, anatomy = {}, [], []
        for sid, handle in handles.items():
            subject = f'subject_{sid}'
            allowed = set(partitions[subject]['train'])
            excluded = set(by_coco(handle['train'])) - allowed
            fit_ids, responses = training_responses(handle['train'], excluded, index)
            vectors = encoding_features(clip[[index[i] for i in fit_ids]], responses, 100.)
            del responses
            meta = metadata(sid)
            if len(vectors) != len(meta['coords']):
                raise ValueError('voxel order/width mismatch')
            entries[subject] = {'fit_coco_ids': fit_ids, 'mask_shape': meta['mask_shape'],
                'flat_indices': torch.as_tensor(meta['flat_indices'].copy())}
            features.append(vectors)
            anatomy.append(meta)
            print('source_features', subject, 'images', len(fit_ids), 'voxels', len(vectors), flush=True)
    all_features = np.concatenate(features)
    pca = PCA(n_components=64, svd_solver='randomized', random_state=args.seed)
    reduced = pca.fit_transform(all_features)
    gmm = GaussianMixture(n_components=128, covariance_type='diag', reg_covar=1e-3,
                          n_init=2, max_iter=200, random_state=args.seed, verbose=1)
    assigned = gmm.fit_predict(reduced)
    if not gmm.converged_:
        raise RuntimeError('GMM did not converge; do not launch training')
    offset = 0
    for sid, vectors, meta in zip(args.sources, features, anatomy):
        labels = assigned[offset:offset+len(vectors)].copy()
        offset += len(vectors)
        entries[f'subject_{sid}']['cluster_ids'] = torch.as_tensor(labels).long()
        meta['labels'] = labels
    target_meta = metadata(args.target)
    target_labels, diagnostics = transfer_labels(anatomy, target_meta, 128, neighbors=5)
    entries[f'subject_{args.target}'] = {'fit_coco_ids': [], 'mask_shape':target_meta['mask_shape'],
        'flat_indices':torch.as_tensor(target_meta['flat_indices'].copy()),
        'cluster_ids':torch.as_tensor(target_labels).long(), 'assignment_method':'ROI_native_position_transfer'}
    output = {'version':1, 'n_clusters':128, 'method':'source_gmm_anatomical_transfer',
        'subjects':entries, 'source_subjects':args.sources, 'target_subject':args.target,
        'seed':args.seed, 'excluded_coco_ids':sorted(set(excluded_test)|set(excluded_val)),
        'feature_model_id':bank.get('model_id'), 'split_sha256':manifest['sha256'],
        'pca_mean':torch.as_tensor(pca.mean_), 'pca_components':torch.as_tensor(pca.components_),
        'gmm_means':torch.as_tensor(gmm.means_), 'gmm_covariances':torch.as_tensor(gmm.covariances_),
        'gmm_weights':torch.as_tensor(gmm.weights_)}
    torch.save(output, root/'functional.pt')
    rng = np.random.default_rng(args.seed)
    for entry in output['subjects'].values():
        labels = entry['cluster_ids'].numpy().copy()
        rng.shuffle(labels)
        entry['cluster_ids'] = torch.as_tensor(labels).long()
    output['method'] = 'random_size_matched_control'
    torch.save(output, root/'random.pt')
    summary = {'target_voxels':len(target_labels), 'target_response_arrays_used_for_fit':0,
        'space':'ROI-local standardized native RAS; approximate, not registered MNI',
        'levels':{name:int((diagnostics['mapping_level']==i).sum()) for i,name in enumerate(('HCP','Kastner','pRF','hemisphere_fallback'))},
        'mean_vote_confidence':float(diagnostics['vote_confidence'].mean()),
        'mean_standardized_distance':float(diagnostics['nearest_standardized_distance'].mean()),
        'distance_quantiles':np.quantile(diagnostics['nearest_standardized_distance'],[.5,.9,.99]).tolist(),
        'target_occupied_clusters':int(len(np.unique(target_labels))),
        'gmm_converged':bool(gmm.converged_), 'gmm_iterations':int(gmm.n_iter_),
        'source_global_occupied_clusters':int(len(np.unique(assigned))),
        'source_counts':{key:{s:len(v) for s,v in value.items()} for key,value in partitions.items()}}
    np.savez_compressed(root/'transfer_diagnostics.npz', **diagnostics)
    save_json(root/'transfer_report.json', summary)
    save_json(root/'manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)


def configuration(args):
    from omegaconf import OmegaConf
    from scripts.run_cluster_ablation import configuration as original
    args.init_checkpoint = Path('NO_BRAIN_CHECKPOINT_SOURCE_ONLY')
    args.batch_size, args.accumulate, args.patience = 4, 8, 5
    cfg, run = original(args)
    cfg.checkpoint = None  # strict unseen-subject test: no official brain checkpoint
    cfg.data._target_ = 'src.dataset_zero_shot.SourceOnlyDataModule'
    cfg.data.subjects = [args.target] if args.mode == 'evaluate' else args.sources
    cfg.data.manifest_path = str(args.output/'manifest.json')
    cfg.data.target_mode = args.mode == 'evaluate'
    cfg.data.split_val = False
    cfg.trainer.max_steps = args.max_steps
    cfg.resume = (run/'coco-caption/checkpoints/last.ckpt').exists()
    cfg.trainer.logger.version = f'attempt_{args.attempt}'
    if args.mode == 'smoke':
        cfg.trainer.max_steps = 1
    # Keep the source training config separate from target inference config.
    (run/('eval_config.yaml' if args.mode == 'evaluate' else 'config.yaml')).write_text(OmegaConf.to_yaml(cfg, resolve=True))
    return cfg, run


def train(args):
    import torch
    from main import main as original_main
    from src.voxel_clustering import audit_cluster_split
    cfg, run = configuration(args)
    manifest = json.loads((args.output/'manifest.json').read_text())
    if args.target in cfg.data.subjects:
        raise ValueError('target subject leaked into training')
    if args.arm != 'baseline':
        bank = torch.load(cfg.encoder.cluster_path, map_location='cpu', weights_only=True)
        if bank['subjects'][f'subject_{args.target}']['fit_coco_ids']:
            raise ValueError('target functional fitting forbidden')
        excluded = set(bank['excluded_coco_ids'])
        for sid in args.sources:
            fit = set(bank['subjects'][f'subject_{sid}']['fit_coco_ids'])
            allowed = set(manifest['subjects'][f'subject_{sid}']['train'])
            if not fit or not fit <= allowed or fit & excluded:
                raise ValueError('invalid source fitting provenance')
    original_main(cfg)
    result = {'arm':args.arm, 'mode':args.mode, 'source_only':True,
              'max_memory_gib':torch.cuda.max_memory_allocated()/1024**3}
    if args.mode == 'train':
        checkpoint = torch.load(run/'coco-caption/checkpoints/last.ckpt', map_location='cpu')
        selected = next(v for k,v in checkpoint['callbacks'].items() if 'ModelCheckpoint' in k)
        result.update(best_checkpoint=selected['best_model_path'], best_val_loss=float(selected['best_model_score']),
            epochs=int(checkpoint['epoch'])+1, global_step=int(checkpoint['global_step']))
    save_json(run/'result.json', result)


def evaluate(args):
    from omegaconf import OmegaConf
    import scripts.run_cluster_ablation as original
    run = args.output/args.arm
    cfg = OmegaConf.load(run/'config.yaml')
    cfg.data.target_mode, cfg.data.subjects = True, [args.target]
    cfg.data.split_audit_path = str(run/'target_split.json')
    original.configuration = lambda _: (cfg, run)
    original.evaluate(args)


def compare(args):
    import numpy as np
    root = args.output
    evaluations = [json.loads((root / arm / 'evaluation.json').read_text()) for arm in ARMS]
    if any(e['keys'] != evaluations[0]['keys'] or e['references'] != evaluations[0]['references'] for e in evaluations[1:]):
        raise ValueError('target evaluation keys/references differ')
    summary = {'exploratory_single_seed': True, 'arms': list(ARMS), 'metrics': {}, 'paired_comparisons': {}}
    scores = {}
    for arm in ARMS:
        metrics = json.loads((root / arm / 'test_metrics.json').read_text())
        per_image = dict(np.load(root / arm / 'per_image_cider.npz'))
        scores[arm] = per_image['real']
        metrics['brain_dependence_shuffle_CIDEr_x100'] = float((per_image['real'] - per_image['shuffle']).mean() * 100)
        summary['metrics'][arm] = metrics
    delta = scores['functional'] - scores['baseline']
    rng = np.random.default_rng(args.seed)
    bootstrap = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(2000)]
    summary['paired_comparisons']['functional_minus_baseline'] = {
        'delta_CIDEr_x100': float(delta.mean() * 100), 'CI95_x100': (np.quantile(bootstrap, [.025, .975]) * 100).tolist()}
    save_json(root / 'summary.json', summary)
    with (root / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['arm','real_CIDEr_x100','BLEU4','ROUGE_L','shuffle_drop_CIDEr_x100'])
        for arm in ARMS:
            metrics = summary['metrics'][arm]
            real = metrics['real']
            writer.writerow([arm,real['CIDEr_x100'],real['BLEU4'],real['ROUGE_L'],metrics['brain_dependence_shuffle_CIDEr_x100']])
    write_report(root)


def write_report(root):
    summary = json.loads((root / 'summary.json').read_text())
    lines = ['# Zero-shot ROI cluster transfer results', '',
             'Source-only S2/S5/S7 training, target S1 inference; seed 42. Random arm skipped at user request.', '',
             '| Arm | CIDEr x100 | BLEU4 | ROUGE-L | Shuffle drop x100 |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        metrics = summary['metrics'][arm]
        real = metrics['real']
        lines.append(f"| {arm} | {real['CIDEr_x100']:.3f} | {real['BLEU4']:.4f} | {real['ROUGE_L']:.4f} | {metrics['brain_dependence_shuffle_CIDEr_x100']:.3f} |")
    result = summary['paired_comparisons']['functional_minus_baseline']
    lines += ['', f"Functional minus baseline: {result['delta_CIDEr_x100']:.3f}; CI95={result['CI95_x100']}", '',
        'Image bootstrap is conditional on this seed/checkpoints. ROI-local native-space transfer is approximate, not MNI registration.',
        'This experiment uses source ridge functional vectors plus GMM, not official UBE voxel embeddings.']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def alive(pid, root):
    try:
        p=Path('/proc')/str(int(pid))
        return str(root) in (p/'cmdline').read_bytes().decode(errors='replace') and 'run_zero_shot_roi_transfer.py' in (p/'cmdline').read_bytes().decode(errors='replace') and (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,ValueError,TypeError):
        return False


def terminate(pid, root):
    if alive(pid,root) and os.getpgid(pid)==pid:
        os.killpg(pid,signal.SIGTERM)
        for _ in range(10):
            if not alive(pid,root): return
            time.sleep(1)
        if alive(pid,root): os.killpg(pid,signal.SIGKILL)


def environment():
    env=os.environ.copy()
    java='/mnt/data2/yjs2026/jdk/jdk-11.0.32.1+1'
    env.update(JAVA_HOME=java, OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONFAULTHANDLER='1',TOKENIZERS_PARALLELISM='false')
    env['PATH']=java+'/bin:'+env.get('PATH','')
    return env


def progress(root, phase, started):
    mode,arm=phase.split('_',1) if '_' in phase else (phase,None)
    files=list((root/arm/'csv').glob('*/metrics.csv')) if mode=='train' else list((root/arm).glob('predictions_*.json')) if mode=='evaluate' else []
    # Preparation/first imports can be slow; its log is a valid heartbeat.
    return max([started]+[p.stat().st_mtime for p in files])


def campaign(args):
    root=args.output
    root.relative_to(PROJECT/'outputs')
    root.mkdir(parents=True,exist_ok=True)
    with (root/'campaign.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return
        return campaign_locked(args)


def campaign_locked(args):
    from scripts.guard_cluster_ablation import gpu_is_available
    root=args.output
    if not (root/'protocol.json').exists():
        names=['scripts/run_zero_shot_roi_transfer.py','src/roi_cluster_transfer.py','src/dataset_zero_shot.py',
            'scripts/run_cluster_ablation.py','main.py','src/dataset.py','src/models/mindllm.py',
            'src/modules/neuro_informed_attn.py','src/modules/clustered_neuro_informed_attn.py','src/utils/callbacks.py']
        names += [str(p.relative_to(PROJECT)) for p in (PROJECT/'config').rglob('*.yaml')]
        save_json(root/'protocol.json', {'sources':args.sources,'target':args.target,'seed':args.seed,
            'initialization':'encoder/projector random; same seed, frozen pretrained Vicuna; NO official brain checkpoint',
            'task':'coco-caption','arms':list(ARMS),'max_epochs':args.epochs,'max_steps':args.max_steps,'max_hours':args.max_hours,
            'selection':'source validation token loss only; patience 5', 'lr':1e-4,'batch_size':4,'accumulate':8,
            'target_fit_samples':0,'target_gradients':False,'gmm':'PCA64 + diagonal GMM128; source-only',
            'transfer':'shared ROI + hemisphere + ROI-normalized native RAS nearest neighbors; approximate, not MNI',
            'feature_bank':str(args.feature_bank),'feature_sha256':file_hash(args.feature_bank),
            'source_sha256':{name:file_hash(PROJECT/name) for name in names}})
    protocol=json.loads((root/'protocol.json').read_text())
    identity={k:getattr(args,k) for k in ('sources','target','seed','max_steps','max_hours')}
    if any(protocol[k]!=v for k,v in identity.items()) or protocol['max_epochs']!=args.epochs:
        raise ValueError('experiment configuration changed')
    path=root/'status.json'
    status=json.loads(path.read_text()) if path.exists() else {'state':'running','completed':[],'started_unix':time.time(),'retries':{}}
    if status['state'] in ('complete','needs_attention','budget_exhausted'): return
    status['pid']=os.getpid()
    save_json(path,status)
    deadline=status['started_unix']+args.max_hours*3600
    for phase in PHASES:
        if phase in status['completed']: continue
        while alive(status.get('child_pid'),root): time.sleep(60)
        mode,arm=phase.split('_',1) if '_' in phase else (phase,None)
        marker=root/'manifest.json' if phase=='prepare' else root/'summary.json' if phase=='compare' else root/(('smoke_'+arm) if mode=='smoke' else arm)/('evaluation.json' if mode=='evaluate' else 'result.json')
        if marker.exists():
            status['completed'].append(phase);save_json(path,status);continue
        while True:
            if time.time()>=deadline:
                status.update(state='budget_exhausted',error='active experiment time budget exhausted');save_json(path,status);return
            for name,expected in protocol['source_sha256'].items():
                if file_hash(PROJECT/name)!=expected: raise ValueError('frozen source changed: '+name)
            if not gpu_is_available():
                status.update(state='waiting_for_gpu',phase=phase);save_json(path,status);time.sleep(60);continue
            count=status['retries'].get(phase,0)
            command=['taskset','-c','16-31',str(PROJECT/'.venv-repaired/bin/python'),'-u',str(Path(__file__).resolve()),
                '--mode',mode,'--output',str(root),'--sources',*map(str,args.sources),'--target',str(args.target),
                '--feature-bank',str(args.feature_bank),'--epochs',str(args.epochs),'--max-steps',str(args.max_steps),
                '--seed',str(args.seed),'--attempt',str(count)]
            if arm: command+=['--arm',arm]
            log_path=root/f'{phase}.attempt_{count}.log'
            status.update(state='running',phase=phase,phase_started_unix=time.time(),log=str(log_path),error=None)
            with log_path.open('a') as log:
                worker=subprocess.Popen(command,cwd=PROJECT,env=environment(),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status['child_pid']=worker.pid;save_json(path,status)
                stalled=False
                while worker.poll() is None:
                    latest=progress(root,phase,status['phase_started_unix'])
                    if phase=='prepare': latest=max(latest,log_path.stat().st_mtime)
                    if time.time()>=deadline or time.time()-latest>1800:
                        stalled=True;terminate(worker.pid,root);break
                    time.sleep(30)
                code=worker.wait()
            if code==0:
                if not marker.exists(): raise RuntimeError('stage missing completion artifact: '+phase)
                status['completed'].append(phase);save_json(path,status);break
            text=log_path.read_text(errors='replace')[-10000:]
            recoverable=stalled or code in (-6,-7,-9,-11,137,139) or any(s in text for s in ('CUDA error','out of memory','Fatal Python error'))
            if not recoverable or count>=2:
                status.update(state='needs_attention',error=f'{phase} exit {code}');save_json(path,status);return
            status['retries'][phase]=count+1;save_json(path,status)
    write_report(root)
    status.update(state='complete',finished_unix=time.time());save_json(path,status)


def inspect(args):
    import scripts.inspect_cluster_ablation as inspector
    inspector.process_alive=alive
    value=inspector.inspect(args.output)
    path=args.output/'status.json'
    status=json.loads(path.read_text())
    if args.ensure and status['state'] in ('running','waiting_for_gpu') and not alive(status.get('pid'),args.output):
        with (args.output/'campaign.log').open('a') as log:
            command=['/usr/bin/python3','-u',str(Path(__file__).resolve()),'--mode','campaign','--output',str(args.output),
                '--sources',*map(str,args.sources),'--target',str(args.target),'--feature-bank',str(args.feature_bank),
                '--epochs',str(args.epochs),'--max-steps',str(args.max_steps),'--seed',str(args.seed),'--max-hours',str(args.max_hours)]
            child=subprocess.Popen(command,cwd=PROJECT,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        value['restarted_pid']=child.pid
    save_json(args.output/'monitor_snapshot.json',value)
    with (args.output/'progress.log').open('a') as stream: stream.write(json.dumps(value)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['campaign','prepare','smoke','train','evaluate','compare','status'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sources',type=int,nargs='+',default=[2,5,7])
    parser.add_argument('--target',type=int,default=1)
    parser.add_argument('--feature-bank',type=Path,default=PROJECT/'data/clip_features/coco_clip_vitl14_s1-7.pt')
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--max-steps',type=int,default=15000)
    parser.add_argument('--max-hours',type=float,default=48)
    parser.add_argument('--attempt',type=int,default=0)
    parser.add_argument('--ensure',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve();args.feature_bank=args.feature_bank.resolve()
    if args.target in args.sources or len(set(args.sources))!=len(args.sources) or any(s not in range(1,9) for s in args.sources+[args.target]):
        parser.error('distinct source subjects and a held-out target in 1..8 required')
    if args.mode in ('smoke','train','evaluate') and args.arm is None: parser.error('--arm required')
    try:
        if args.mode=='campaign': campaign(args)
        elif args.mode=='prepare': prepare(args)
        elif args.mode in ('smoke','train'): train(args)
        elif args.mode=='evaluate': evaluate(args)
        elif args.mode=='status': inspect(args)
        else: compare(args)
    except BaseException as error:
        if args.mode=='campaign' and (args.output/'status.json').exists():
            status=json.loads((args.output/'status.json').read_text());status.update(state='needs_attention',error=str(error));save_json(args.output/'status.json',status)
        raise
