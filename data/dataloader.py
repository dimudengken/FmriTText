"""单被试 batch 的 DataLoader（Key/Value 编码器需要单被试）。

协议（被试切分）：
- 训练：S1-7，每个被试全量 trial = train(非 shared1000) ∪ new_test(shared1000)。
- 测试：S8 留出，new_test（1000 张 shared1000 图 × 3 遍 ≈ 3000 trial），不 shuffle。

batch 为【单被试】——Key/Value 编码器的解剖 Key 逐被试不同、体素长度也不同，
混合被试 batch 无法堆叠。等权：batch 采样器按被试分组、打乱 batch 顺序。

内存：betas 走 hdf5 懒加载句柄；images 句柄共享。num_workers 必须 = 0（h5py 不可 pickle）。
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset

from .nsd_dataset import build_subject_dataset
from . import preprocessing


def collate_batch(batch):
    """单被试 batch：体素同长，可堆叠。"""
    out = {
        "voxels": torch.stack([b["voxels"] for b in batch]),  # (B, n_voxels)
        "captions": [b["caption"] for b in batch],
        "nsd_idx": torch.tensor([b["nsd_idx"] for b in batch], dtype=torch.long),
        "subj": batch[0]["subj"],  # int，单被试 batch
    }
    if "image" in batch[0]:
        out["image"] = torch.stack([b["image"] for b in batch])  # (B, 3, 224, 224)
    return out


class SubjectBatchSampler(Sampler):
    """把样本按被试分组，形成单被试 batch，再打乱 batch 顺序（各被试等权）。"""

    def __init__(self, dataset_lengths, batch_size, shuffle=True, drop_last=True, seed=42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        g = torch.Generator().manual_seed(seed)

        offsets = np.cumsum([0] + list(dataset_lengths))
        batches = []
        for s, length in enumerate(dataset_lengths):
            start = offsets[s]
            idx = list(range(start, start + length))
            if shuffle:
                idx = torch.randperm(length, generator=g).tolist()
                idx = [start + i for i in idx]
            for b in range(0, len(idx), batch_size):
                chunk = idx[b:b + batch_size]
                if drop_last and len(chunk) < batch_size:
                    break
                batches.append(chunk)

        if shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        self.batches = batches

    def __iter__(self):
        for chunk in self.batches:
            yield chunk

    def __len__(self):
        return len(self.batches)


def build_train_loader(data_path, subj_list, captions_by_nsd_idx, batch_size,
                       return_image=True, num_workers=0, seed=42, collate_fn=None,
                       splits=("train", "new_test")):
    """S1-7 训练 DataLoader（单被试 batch）。

    collate_fn 可替换默认的 collate_batch（如 SFT 需把 caption 分词并 padding）。
    splits 默认 train∪new_test（含 shared1000）；BIT-LLM 协议用 ("train",) 剔除 shared1000，
    S8 shared1000 才成为干净 held-out。
    """
    images = preprocessing.load_images_handle(data_path)
    datasets = [
        build_subject_dataset(data_path, s, list(splits), captions_by_nsd_idx,
                              return_image=return_image, images=images)
        for s in subj_list
    ]
    concat = ConcatDataset(datasets)
    sampler = SubjectBatchSampler([len(ds) for ds in datasets], batch_size,
                                  shuffle=True, drop_last=True, seed=seed)
    return DataLoader(concat, batch_sampler=sampler, collate_fn=collate_fn or collate_batch,
                      num_workers=num_workers)


def build_train_val_loaders(data_path, subj_list, captions_by_nsd_idx, batch_size,
                            val_holdout=0.05, return_image=True, num_workers=0,
                            seed=42, collate_fn=None, splits=("train", "new_test")):
    """S1-7 训练 + 每被试随机划 val_holdout 的验证 DataLoader（均单被试 batch）。

    与 build_train_loader 同构，但每被试从 splits 指定的 trial 池随机划出 val_holdout
    比例做独立验证集（Subset，每被试独立种子，可复现），S8 全程不参与。
    验证不 shuffle、drop_last=False（不丢样本）。返回 (train_loader, val_loader)。
    splits 默认 train∪new_test；BIT-LLM 协议用 ("train",) 剔除 shared1000。
    """
    images = preprocessing.load_images_handle(data_path)
    train_dss, val_dss = [], []
    for s in subj_list:
        ds = build_subject_dataset(data_path, s, list(splits),
                                   captions_by_nsd_idx, return_image=return_image,
                                   images=images)
        n = len(ds)
        n_val = int(n * val_holdout)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed + s))
        val_dss.append(Subset(ds, perm[:n_val].tolist()))
        train_dss.append(Subset(ds, perm[n_val:].tolist()))

    train_concat = ConcatDataset(train_dss)
    val_concat = ConcatDataset(val_dss)
    train_sampler = SubjectBatchSampler([len(ds) for ds in train_dss], batch_size,
                                        shuffle=True, drop_last=True, seed=seed)
    val_sampler = SubjectBatchSampler([len(ds) for ds in val_dss], batch_size,
                                      shuffle=False, drop_last=False, seed=seed)
    train_loader = DataLoader(train_concat, batch_sampler=train_sampler,
                              collate_fn=collate_fn or collate_batch, num_workers=num_workers)
    val_loader = DataLoader(val_concat, batch_sampler=val_sampler,
                            collate_fn=collate_fn or collate_batch, num_workers=num_workers)
    return train_loader, val_loader


def build_finetune_loader(data_path, subj, captions_by_nsd_idx, batch_size,
                          split="new_test", return_image=True, num_workers=0,
                          seed=42, collate_fn=None, shuffle=True, drop_last=True):
    """单被试 S8 微调 DataLoader（默认 shared1000=new_test，即 MindEye2 的 ~750 图协议）。

    S8 只有这一个被试，无需 SubjectBatchSampler 的被试等权；直接 DataLoader + shuffle。
    shuffle/drop_last 可覆盖：如 --contrastive 的 gallery 构建用 shuffle=False, drop_last=False
    保证覆盖全部 trial（否则两个 pass 的 shuffle 状态不同，drop_last 丢的 trial 不同 → 漏图）。
    """
    images = preprocessing.load_images_handle(data_path)
    ds = build_subject_dataset(data_path, subj, split, captions_by_nsd_idx,
                               return_image=return_image, images=images)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn or collate_batch,
                      num_workers=num_workers, drop_last=drop_last,
                      generator=torch.Generator().manual_seed(seed))


def build_test_loader(data_path, subj, captions_by_nsd_idx, batch_size,
                      return_image=True, num_workers=0, split="new_test"):
    """S8 留出被试的 DataLoader（不 shuffle）。

    split 默认 new_test（shared1000，与 S8 微调同源）；传 "train" 可测 S8 的 unique 图
    （从未进过 S8 微调 → held-out 泛化检查，判断 CIDEr 是记忆还是真泛化）。
    """
    images = preprocessing.load_images_handle(data_path)
    ds = build_subject_dataset(data_path, subj, split, captions_by_nsd_idx,
                               return_image=return_image, images=images)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_batch, num_workers=num_workers, drop_last=False)
