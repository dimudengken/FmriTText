"""Data path for the standalone contrastive stage.

This does not go through `fMRIInstructionDataModule`: that one exists to tokenize
instruction-following chats and would force a Vicuna tokenizer plus a built
instruction dataset into a stage that only ever needs voxels and a stimulus id.
"""

import logging
import os

import h5py
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from src.dataset import train_val_split_by_coco_id

logger = logging.getLogger(__name__)


class BrainContrastiveDataset(Dataset):
    """Rows of fMRI repeats, each carrying the stimulus id its target is keyed by.

    An entry holds either one `sample_key` (training: one row per repeat) or a tuple
    of them (validation: one row per stimulus, averaged over its repeats, which is
    the alignment the retrieval metric expects).
    """

    def __init__(self, groups, entries):
        self.groups = groups
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        subject, sample_keys, coco_id = self.entries[index]
        group = self.groups[subject]
        if isinstance(sample_keys, str):
            voxels = group[sample_keys]["visual_voxels"][()]
        else:
            voxels = np.stack(
                [group[key]["visual_voxels"][()] for key in sample_keys]
            ).mean(0)
        return {
            "voxels": torch.as_tensor(voxels, dtype=torch.float32),
            "coco_id": coco_id,
            "subject": subject,
        }


def collate_brain_contrastive(batch):
    return {
        "voxels": torch.stack([item["voxels"] for item in batch]),
        "coco_id": torch.tensor([item["coco_id"] for item in batch], dtype=torch.long),
        "subject": [item["subject"] for item in batch],
    }


def _group_indices_by_subject(group_subjects):
    grouped = {}
    for index, subject in enumerate(group_subjects):
        grouped.setdefault(subject, []).append(index)
    return grouped


class SubjectCocoBatchSampler(Sampler):
    """Batches whole stimuli, and never mixes subjects inside a batch.

    Whole stimuli land together so every repeat of an image acts as a mutual
    positive instead of being pushed apart as a false negative. Subjects stay apart
    because two things forbid mixing them: `safe_get_source` asserts a batch holds
    one subject, and each subject's voxels have a different width, so stacking rows
    from two subjects is not even a valid tensor. The price is that a batch's
    negatives all come from one subject; cross-subject positives are simply absent.
    """

    def __init__(
        self, coco_groups, coco_subjects, stimuli_per_batch, shuffle=True, drop_last=True
    ):
        self.coco_groups = coco_groups
        self.by_subject = _group_indices_by_subject(coco_subjects)
        self.stimuli_per_batch = stimuli_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last

    def _chunks(self, rows):
        limit = len(rows)
        if self.drop_last:
            limit = (limit // self.stimuli_per_batch) * self.stimuli_per_batch
        for start in range(0, limit, self.stimuli_per_batch):
            batch = []
            for i in rows[start : start + self.stimuli_per_batch]:
                batch.extend(self.coco_groups[i])
            yield batch

    def __iter__(self):
        subjects = list(self.by_subject)
        np.random.shuffle(subjects)
        batches = []
        for subject in subjects:
            rows = np.array(self.by_subject[subject])
            np.random.shuffle(rows)
            batches.extend(self._chunks(rows))
        np.random.shuffle(batches)
        yield from batches

    def __len__(self):
        total = 0
        for rows in self.by_subject.values():
            n = len(rows)
            total += n // self.stimuli_per_batch if self.drop_last else -(-n // self.stimuli_per_batch)
        return total


class SubjectBatchSampler(Sampler):
    """Fixed-order, single-subject batches, for validation.

    Validation still concatenates every subject's rows before ranking, so the
    candidate bank is every val stimulus of every subject. Rows of one image
    belonging to different subjects share a target embedding, which ties rather
    than loses under `retrieval_metrics`' strict comparison.
    """

    def __init__(self, batch_size, subjects):
        self.batch_size = batch_size
        self.by_subject = _group_indices_by_subject(subjects)

    def __iter__(self):
        for rows in self.by_subject.values():
            for start in range(0, len(rows), self.batch_size):
                yield rows[start : start + self.batch_size]

    def __len__(self):
        return sum(
            -(-len(rows) // self.batch_size) for rows in self.by_subject.values()
        )


class BrainContrastiveDataModule(LightningDataModule):
    def __init__(
        self,
        subjects=(1,),
        h5_dir="data/nsd0",
        val_coco=900,
        split_seed=42,
        stimuli_per_batch=16,
        val_batch_size=32,
        num_workers=0,
        drop_last=True,
        audit_test_ids=False,
    ):
        super().__init__()
        self.subjects = list(subjects)
        self.h5_dir = h5_dir
        self.val_coco = val_coco
        self.split_seed = split_seed
        self.stimuli_per_batch = stimuli_per_batch
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.audit_test_ids = audit_test_ids
        self._ready = False
        self.test_coco_ids = set()

    def _split(self, group, subject):
        if self.val_coco <= 0:
            return group, {}
        return train_val_split_by_coco_id(
            group, self.val_coco, subject, seed=self.split_seed
        )

    @staticmethod
    def _by_coco(group):
        grouped = {}
        for key in group:
            grouped.setdefault(int(group[key]["coco_id"][()]), []).append(key)
        return grouped

    def setup(self, stage=None):
        # The trainer calls setup once for validate and again for fit, and reopening
        # the h5 handles would invalidate the datasets built the first time.
        if self._ready:
            return
        self._ready = True

        self.files = {
            f"subject_{subject}": h5py.File(
                os.path.join(self.h5_dir, f"subject_{subject}.h5"), "r"
            )
            for subject in self.subjects
        }
        if self.audit_test_ids:
            self.test_coco_ids = {
                int(handle["test"][key]["coco_id"][()])
                for handle in self.files.values() if "test" in handle
                for key in handle["test"]
            }
        self.train_groups, self.val_groups = {}, {}
        train_entries, val_entries = [], []
        coco_groups, coco_subjects, val_subjects = [], [], []

        for subject, handle in self.files.items():
            train, val = self._split(handle["train"], subject)
            self.train_groups[subject] = train
            self.val_groups[subject] = val

            train_by_coco = self._by_coco(train)
            for coco_id, keys in train_by_coco.items():
                coco_groups.append(
                    list(range(len(train_entries), len(train_entries) + len(keys)))
                )
                coco_subjects.append(subject)
                train_entries.extend((subject, key, coco_id) for key in keys)

            val_by_coco = self._by_coco(val)
            for coco_id, keys in val_by_coco.items():
                val_entries.append((subject, tuple(keys), coco_id))
                val_subjects.append(subject)

            repeats = np.array([len(keys) for keys in train_by_coco.values()])
            widths = {len(keys) for keys in train_by_coco.values()}
            logger.info(
                f"[stage1-data] {subject}: train_samples={len(train)} "
                f"train_stimuli={len(train_by_coco)} repeats_per_stimulus "
                f"min/max/mean={repeats.min()}/{repeats.max()}/{repeats.mean():.2f} "
                f"val_stimuli={len(val_by_coco)}"
            )
            if len(widths) != 1:
                logger.warning(
                    f"[stage1-data] {subject}: repeats_per_stimulus is not uniform "
                    f"({sorted(widths)}); whole stimuli are still kept together, so "
                    f"batch sizes here are not constant"
                )

        self.train_dataset = BrainContrastiveDataset(self.train_groups, train_entries)
        self.val_dataset = BrainContrastiveDataset(self.val_groups, val_entries)
        self.val_coco_ids = {coco_id for _, _, coco_id in val_entries}
        self.coco_groups = coco_groups
        self.coco_subjects = coco_subjects
        self.val_subjects = val_subjects
        self.train_batches = len(
            SubjectCocoBatchSampler(
                coco_groups, coco_subjects, self.stimuli_per_batch, drop_last=self.drop_last
            )
        )
        logger.info(
            f"[stage1-data] total train_samples={len(train_entries)} "
            f"train_stimuli={len(coco_groups)} subjects={len(self.files)} "
            f"batches_per_epoch={self.train_batches} "
            f"val_rows={len(val_entries)}"
        )

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
            collate_fn=collate_brain_contrastive,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_sampler=SubjectBatchSampler(self.val_batch_size, self.val_subjects),
            num_workers=self.num_workers,
            collate_fn=collate_brain_contrastive,
        )
