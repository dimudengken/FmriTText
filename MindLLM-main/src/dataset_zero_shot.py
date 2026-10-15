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
