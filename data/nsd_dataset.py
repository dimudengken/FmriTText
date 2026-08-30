"""NSD PyTorch Dataset：按 trial 产出 (voxels, image, caption)。

依赖 preprocessing 预构建的 trial 索引与 caption 索引；体素默认走 hdf5 懒加载句柄
（不整读，省内存），图像走共享 hdf5 句柄懒加载。I2T 训练用 caption，Stage 1 双对齐额外用 image。
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import preprocessing


class NSDDataset(Dataset):
    """单个被试的 NSD 数据集。

    Args:
        data_path: 含 betas_*.hdf5 / coco_images_*.hdf5 的根目录。
        subj: 被试编号 1~8。
        image_idx: (n,) int64，每个 trial 对应的 COCO 图像索引（nsdId）。
        voxel_idx: (n,) int64，每个 trial 对应的体素行号。
        captions_by_nsd_idx: list[list[str]]，图像索引→caption 列表。
        betas: 可选。numpy 数组（整读）或 h5py 句柄；None = 懒加载句柄（默认）。
        images: hdf5 句柄（preprocessing.load_images_handle 的结果），可跨被试共享。
        return_image: 是否产出图像张量（Stage 1 需要，SFT/GRPO 可关掉省 I/O）。
        caption_strategy: 'first' 取第一条；'random' 随机取一条；'all' 返回全部。
    """

    def __init__(self, data_path, subj, image_idx, voxel_idx,
                 captions_by_nsd_idx, betas=None, images=None,
                 return_image=True, caption_strategy="random"):
        self.data_path = data_path
        self.subj = subj
        self.image_idx = np.asarray(image_idx, dtype=np.int64)
        self.voxel_idx = np.asarray(voxel_idx, dtype=np.int64)
        self.captions_by_nsd_idx = captions_by_nsd_idx
        self.return_image = return_image
        self.caption_strategy = caption_strategy

        assert len(self.image_idx) == len(self.voxel_idx), \
            "image_idx 与 voxel_idx 长度不一致"

        # betas：numpy 数组（整读）/ h5py 句柄 / None(懒加载句柄，默认)
        self._betas_file = None
        if betas is not None:
            self.betas = betas
        else:
            self._betas_file = h5py.File(
                f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5", "r")
            self.betas = self._betas_file["betas"]

        self.images = images if images is not None else preprocessing.load_images_handle(
            data_path)

    def __len__(self):
        return len(self.image_idx)

    def _sample_caption(self, nsd_idx):
        caps = self.captions_by_nsd_idx[nsd_idx]
        if not caps:
            return ""
        if self.caption_strategy == "first":
            return caps[0]
        if self.caption_strategy == "all":
            return caps
        return caps[np.random.randint(len(caps))]

    def __getitem__(self, idx):
        nsd_idx = int(self.image_idx[idx])
        vox = torch.from_numpy(np.asarray(self.betas[self.voxel_idx[idx]])).float()  # (n_voxels,)

        item = {
            "voxels": vox,
            "caption": self._sample_caption(nsd_idx),
            "nsd_idx": nsd_idx,
            "subj": self.subj,
        }

        if self.return_image:
            # MindEye2 打包的 coco_images_224_float16.hdf5 已是 [0,1] 的 float16，
            # 不能再 /255（会变成近黑图，CLIP 嵌入全挤一起，对比学习无从学起）。
            img = self.images[nsd_idx]  # (3, 224, 224) float16, [0,1]
            item["image"] = torch.from_numpy(img.astype(np.float32))

        return item

    def close(self):
        if self._betas_file is not None:
            self._betas_file.close()
            self._betas_file = None


def build_full_trial_index(data_path, subj, splits, num_sessions=None):
    """合并多个 split 的 trial 索引，返回 (image_idx, voxel_idx)。

    训练被试的全量数据 = ['train'(非 shared1000) + 'new_test'(shared1000)]，
    两者 trial 互斥，concat 即该被试的全部 trial。
    """
    image_idx, voxel_idx = [], []
    for sp in splits:
        ii, vi = preprocessing.build_trial_index_from_wds(
            data_path, subj, sp, num_sessions=num_sessions)
        image_idx.append(ii)
        voxel_idx.append(vi)
    return np.concatenate(image_idx), np.concatenate(voxel_idx)


def build_subject_dataset(data_path, subj, split, captions_by_nsd_idx,
                          num_sessions=None, return_image=True, betas=None, images=None):
    """便捷构造：从 webdataset 现场建索引并返回 NSDDataset。

    split ∈ {'train','test','new_test'} 或它们的 list。训练被试应传
    ['train','new_test'] 拿全量 trial；测试被试传 'new_test'（shared1000）。
    """
    splits = [split] if isinstance(split, str) else list(split)
    image_idx, voxel_idx = build_full_trial_index(data_path, subj, splits, num_sessions)
    return NSDDataset(
        data_path, subj, image_idx, voxel_idx, captions_by_nsd_idx,
        betas=betas, images=images, return_image=return_image,
    )
