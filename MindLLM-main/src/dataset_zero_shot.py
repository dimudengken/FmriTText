"""Image-disjoint source-only training and untouched target-test inference."""
import json
from pathlib import Path
from src.dataset import fMRIInstructionDataModule


class SourceOnlyDataModule(fMRIInstructionDataModule):
    def __init__(self, manifest_path, target_mode=False, split_audit_path=None, **kwargs):
        manifest = json.loads(Path(manifest_path).read_text())
        expected = [manifest['target']] if target_mode else manifest['sources']
        if list(kwargs['subjects']) != expected:
            raise ValueError('subjects disagree with frozen zero-shot manifest')
        kwargs['split_val'] = False
        if target_mode:
            # Bypass the base constructor, which opens the target train group.
            import h5py
            from lightning import LightningDataModule
            from src.dataset import DataCollator
            LightningDataModule.__init__(self)
            if kwargs.get('nsd_version', 0) != 0 or kwargs.get('whole_brain', False):
                raise ValueError('zero-shot protocol requires NSD0 visual voxels')
            self.tokenizer = kwargs['tokenizer']
            self.subjects = expected
            self.nsd_version, self.whole_brain = 0, False
            self.task, self.exclude = kwargs.get('task'), kwargs.get('exclude')
            self.num_workers, self.batch_size = kwargs.get('num_workers', 0), kwargs.get('batch_size', 4)
            self.train_samples, self.split_val = -1, False
            self.mixup = False
            self.group_by = 'coco' if kwargs.get('group_by_coco', True) else 'sample'
            self.collate_fn = DataCollator(self.tokenizer)
            self._files = {f'subject_{sid}': h5py.File(f'data/nsd0/subject_{sid}.h5', 'r') for sid in expected}
        else:
            super().__init__(**kwargs)
        self.target_mode = target_mode
        self.manifest = manifest
        # No target training/validation sample identifiers or response arrays are read.
        self.file = {split: {f'subject_{sid}': {} for sid in expected} for split in ('train','val','test')}
        for sid in expected:
            subject = f'subject_{sid}'
            handle = self._files[subject]
            partitions = manifest['subjects'][subject]
            if target_mode:
                allowed = set(partitions['test'])
                self.file['test'][subject] = {key: group for key, group in handle['test'].items()
                                             if int(group['coco_id'][()]) in allowed}
            else:
                train, val = set(partitions['train']), set(partitions['val'])
                if train & val:
                    raise ValueError('source train/validation overlap')
                for key, group in handle['train'].items():
                    image_id = int(group['coco_id'][()])
                    if image_id in train:
                        self.file['train'][subject][key] = group
                    elif image_id in val:
                        self.file['val'][subject][key] = group
        if split_audit_path:
            Path(split_audit_path).write_text(json.dumps({'sha256': manifest['sha256'],
                                                         'subjects': manifest['subjects']}, indent=2))

    def setup(self, stage):
        if self.target_mode != (stage in ('test','predict')):
            raise ValueError('source/target data role does not match trainer stage')
        super().setup(stage)
        if stage in ('fit','validate'):
            self.val_data = [self.get_data_by_source(self.file['val'], source, split='val', generate_prompt=True)
                             for source in self.instruction_dataset.datasets]
