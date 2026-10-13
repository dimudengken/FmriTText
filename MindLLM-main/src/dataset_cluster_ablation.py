"""COCO caption ablation with validation carved out of training, never test."""

import hashlib
import json
from pathlib import Path

from src.dataset import fMRIInstructionDataModule


class ClusterAblationDataModule(fMRIInstructionDataModule):
    def __init__(self, split_audit_path=None, **kwargs):
        if kwargs.get("nsd_version", 0) != 0 or not kwargs.get("split_val", False):
            raise ValueError("ablation requires NSD0 and split_val=true")
        super().__init__(**kwargs)
        self.split_audit_path = split_audit_path
        self.split_audit = {}
        for subject in self.file["train"]:
            ids = {split: sorted({int(group[key]["coco_id"][()]) for key in group})
                   for split in ("train", "val", "test")
                   for group in [self.file[split][subject]]}
            if any(set(ids[a]) & set(ids[b]) for a, b in
                   (("train", "val"), ("train", "test"), ("val", "test"))):
                raise ValueError(f"overlapping ablation splits: {subject}")
            self.split_audit[subject] = ids
        encoded = json.dumps(self.split_audit, sort_keys=True).encode()
        self.split_sha256 = hashlib.sha256(encoded).hexdigest()
        if split_audit_path:
            path = Path(split_audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"sha256": self.split_sha256,
                                      "subjects": self.split_audit}, indent=2))

    def setup(self, stage):
        super().setup(stage)
        if stage in ("fit", "validate"):
            self.val_data = [
                self.get_data_by_source(self.file["val"], source,
                                        split="val", generate_prompt=True)
                for source in self.instruction_dataset.datasets
            ]
