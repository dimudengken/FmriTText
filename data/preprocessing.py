"""体素归一化 + trial 索引 / caption 索引构建。

数据格式（MindEye2 打包的 pscotti/mindeyev2）：
- 体素: {data_path}/betas_all_subj0{s}_fp32_renorm.hdf5 → key 'betas'
    shape (n_trials, n_voxels)，float32；n_trials = 750 * 会话数（见下方 nsessions）
- 图像: {data_path}/coco_images_224_float16.hdf5 → key 'images'
    shape (73000, 3, 224, 224)，float16，**已归一化到 [0,1]**，按 nsdId 0~72999 排序
    （注意：不要再 /255，否则 CLIP 嵌入会坍缩）
- trial: {data_path}/wds/subj0{s}/{split}/*.tar → behav.npy
    shape (1, 17)；behav[0, 0] = cocoidx 图像索引（nsdId 0~72999）
                  behav[0, 5] = global_trial 体素行号（betas 的行，0~n_trials-1）
- 训练会话数（各被试不同）: nsessions_allsubj = [40, 40, 32, 30, 40, 32, 40, 30]
"""
import os

import h5py
import numpy as np


def zscore(betas, axis=0):
    """z-score 归一化（默认沿 trial 维度算均值/标准差）。

    betas: (n_trials, n_voxels) 或 (n_trials, 1, n_voxels)，返回同形状。
    """
    mean = betas.mean(axis=axis, keepdims=True)
    std = betas.std(axis=axis, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (betas - mean) / std


def build_trial_index_from_wds(data_path, subj, split, num_sessions=None):
    """从 webdataset tar 提取 trial → (image_idx, voxel_idx) 索引。

    返回 (image_idx, voxel_idx) 两个 int64 numpy 数组，长度 = 该被试该 split 的 trial 数。
    这是 MindEye2 训练循环里 behav0[:,0,0] / behav0[:,0,5] 的等价物（此处无 batching，
    behav 是单个样本的原始 2D (1,17)）。提前物化可避免训练时反复流式解 tar。
    """
    import webdataset as wds

    nsessions_allsubj = [40, 40, 32, 30, 40, 32, 40, 30]
    sessions = {
        "train": num_sessions if num_sessions else nsessions_allsubj[subj - 1],
        "test": 1,
        "new_test": 1,
    }[split]

    url = f"{data_path}/wds/subj0{subj}/{split}/" + "{0.." + f"{sessions - 1}" + "}.tar"
    ds = (
        wds.WebDataset(url, resampled=False, nodesplitter=lambda urls: urls)
        .decode("torch")
        .rename(behav="behav.npy")
        .to_tuple("behav")
    )

    image_idx, voxel_idx = [], []
    for (behav,) in ds:
        # behav 原始形状 (1, 17)：列0=cocoidx(图像索引)，列5=global_trial(体素行号)。
        # 用 np.asarray 兼容 numpy 数组 / torch tensor，[:, i] 统一取成 (1,) 一维。
        behav = np.asarray(behav)
        image_idx.append(behav[:, 0].astype(np.int64))
        voxel_idx.append(behav[:, 5].astype(np.int64))
    return np.concatenate(image_idx), np.concatenate(voxel_idx)


def build_caption_index(nsd_stim_info_csv, coco_annotations_json):
    """构建 图像索引 → list[str] 的 caption 映射。

    参数:
        nsd_stim_info_csv: NSD 的 stim info（含 73k 图像的顺序索引 ↔ COCO image id）。
        coco_annotations_json: str 或 list[str]，COCO2017 的 captions json。
            NSD 的 73k 图像同时来自 train2017 + val2017 两个 split，所以应传
            [captions_train2017.json, captions_val2017.json] 合并两者。

    返回:
        captions_by_nsd_idx: list[list[str]]，captions_by_nsd_idx[i] = 第 i 张 NSD 图像的 caption 列表。

    注意: NSD 图像索引是 MindEye2 内部 0~72999 的顺序，与 COCO image id 不同，
    必须先通过 nsd_stim_info 对齐。该映射需 NSD 授权文件，此处留接口。
    """
    import json

    import pandas as pd

    # NSD 顺序索引 → cocoId
    stim = pd.read_csv(nsd_stim_info_csv)
    nsd_idx_to_coco_id = dict(zip(stim["nsdId"].values, stim["cocoId"].values))

    # cocoId → captions（合并 train + val，image_id 跨 split 全局唯一）
    if isinstance(coco_annotations_json, (str, os.PathLike)):
        coco_annotations_json = [coco_annotations_json]
    coco_id_to_captions = {}
    for path in coco_annotations_json:
        with open(path) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            coco_id_to_captions.setdefault(ann["image_id"], []).append(ann["caption"])

    n = max(nsd_idx_to_coco_id) + 1
    captions_by_nsd_idx = [None] * n
    for nsd_idx, coco_id in nsd_idx_to_coco_id.items():
        captions_by_nsd_idx[nsd_idx] = coco_id_to_captions.get(coco_id, [])
    return captions_by_nsd_idx


def build_caption_map(data_path):
    """从 MindEye2 数据根目录组装 caption 索引（nsd_stim_info + COCO annotations）。"""
    stim_info = os.path.join(data_path, "nsd_stim_info_merged.csv")
    annotations_dir = os.path.join(data_path, "annotations")
    caps = [
        os.path.join(annotations_dir, "captions_train2017.json"),
        os.path.join(annotations_dir, "captions_val2017.json"),
    ]
    for p in [stim_info] + caps:
        assert os.path.exists(p), f"缺少文件: {p}"
    return build_caption_index(stim_info, caps)


def load_betas(data_path, subj, zscore_voxels=False):
    """加载单被试体素，返回 (n_trials, n_voxels) 的 float32 numpy 数组。

    注意：MindEye2 的 _fp32_renorm betas 已逐体素 z-score 过，默认不再二次标准化
    （zscore_voxels=False）。全量读入约 1.1~1.9GB/被试，内存吃紧时改用 NSDDataset
    的懒加载句柄（不整读）。
    """
    path = f"{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5"
    with h5py.File(path, "r") as f:
        betas = f["betas"][:]
    if zscore_voxels:
        betas = zscore(betas, axis=0)
    return betas.astype(np.float32)


def load_images_handle(data_path):
    """返回 images hdf5 的懒加载句柄（73k 张图像太大，不整体进内存）。"""
    return h5py.File(f"{data_path}/coco_images_224_float16.hdf5", "r")["images"]
