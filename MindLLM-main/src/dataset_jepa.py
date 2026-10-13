"""Paired NSD repeats for a standalone cross-repeat experiment.

The split is global across subjects, so every presentation of a COCO stimulus
lands on the same side of the train/validation boundary. Only stimuli with at
least two distinct presentations in one subject produce pairs.
"""

import logging
import random

import torch
from torch.utils.data import DataLoader, Dataset

from src.dataset_contrastive import (
    BrainContrastiveDataModule,
    SubjectBatchSampler,
    SubjectCocoBatchSampler,
)

logger = logging.getLogger(__name__)


class CrossRepeatDataset(Dataset):
    def __init__(self, groups, entries, random_pair):
        self.groups = groups
        self.entries = entries
        self.random_pair = random_pair

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        subject, keys, coco_id = self.entries[index]
        if self.random_pair:
            first, second = random.sample(keys, 2)
        else:
            first, second = keys[:2]
        group = self.groups[subject]
        return {
            "context_voxels": torch.as_tensor(
                group[first]["visual_voxels"][()], dtype=torch.float32
            ),
            "target_voxels": torch.as_tensor(
                group[second]["visual_voxels"][()], dtype=torch.float32
            ),
            "coco_id": coco_id,
            "subject": subject,
        }


def collate_cross_repeat(batch):
    return {
        "context_voxels": torch.stack([item["context_voxels"] for item in batch]),
        "target_voxels": torch.stack([item["target_voxels"] for item in batch]),
        "coco_id": torch.tensor([item["coco_id"] for item in batch], dtype=torch.long),
        "subject": [item["subject"] for item in batch],
    }


class CrossRepeatDataModule(BrainContrastiveDataModule):
    """Globally split stimuli, then pair repeats within each subject."""

    def __init__(self, val_fraction=0.1, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must lie in (0, 1)")
        self.val_fraction = val_fraction

    def _split(self, group, subject):
        # One split shared by all subjects prevents the same image from being a
        # training example in one subject and a validation example in another.
        if not hasattr(self, "_heldout_ids"):
            all_train_ids = {
                int(handle["train"][key]["coco_id"][()])
                for handle in self.files.values()
                for key in handle["train"]
            }
            test_overlap = all_train_ids & self.test_coco_ids
            if test_overlap:
                raise ValueError(
                    "JEPA train/test stimuli overlap; first coco_ids="
                    f"{sorted(test_overlap)[:10]}"
                )
            ids = sorted(all_train_ids)
            random.Random(self.split_seed).shuffle(ids)
            n_val = max(1, round(len(ids) * self.val_fraction))
            self._heldout_ids = set(ids[:n_val])
            logger.info(
                "[cross-repeat-data] global train/val split: %d/%d stimuli",
                len(ids) - n_val, n_val,
            )
        train, val = {}, {}
        for key in group:
            coco_id = int(group[key]["coco_id"][()])
            (val if coco_id in self._heldout_ids else train)[key] = group[key]
        return train, val

    def setup(self, stage=None):
        if getattr(self, "_pair_ready", False):
            return
        super().setup(stage)

        train_entries, val_entries = [], []
        for subject, group in self.train_groups.items():
            train_entries.extend(
                (subject, tuple(keys), coco_id)
                for coco_id, keys in self._by_coco(group).items()
                if len(keys) >= 2
            )
        for subject, group in self.val_groups.items():
            val_entries.extend(
                (subject, tuple(keys), coco_id)
                for coco_id, keys in self._by_coco(group).items()
                if len(keys) >= 2
            )
        if not train_entries or not val_entries:
            raise ValueError("cross-repeat training requires paired train and validation stimuli")

        self.train_dataset = CrossRepeatDataset(self.train_groups, train_entries, True)
        self.val_dataset = CrossRepeatDataset(self.val_groups, val_entries, False)
        self.coco_groups = [[index] for index in range(len(train_entries))]
        self.coco_subjects = [subject for subject, _, _ in train_entries]
        self.val_subjects = [subject for subject, _, _ in val_entries]
        self.val_coco_ids = self._heldout_ids
        self.train_batches = len(
            SubjectCocoBatchSampler(
                self.coco_groups,
                self.coco_subjects,
                self.stimuli_per_batch,
                drop_last=self.drop_last,
            )
        )
        logger.info(
            "[cross-repeat-data] train_pairs=%d val_pairs=%d "
            "batches_per_epoch=%d subjects=%d",
            len(train_entries), len(val_entries), self.train_batches, len(self.files),
        )
        self._pair_ready = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=SubjectCocoBatchSampler(
                self.coco_groups,
                self.coco_subjects,
                self.stimuli_per_batch,
                shuffle=True,
                drop_last=self.drop_last,
            ),
            num_workers=self.num_workers,
            collate_fn=collate_cross_repeat,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=SubjectBatchSampler(self.val_batch_size, self.val_subjects),
            num_workers=self.num_workers,
            collate_fn=collate_cross_repeat,
        )
