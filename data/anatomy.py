"""从 brain_region_masks.hdf5 提取分区特征并缓存，供 Key/Value 编码器使用。

brain_region_masks.hdf5（MindEye2 HF 仓库）逐被试给出 7 个布尔掩膜：
V1 / V2 / V3 / V4 / early_vis / higher_vis / nsd_general，shape 均 = (该被试 betas 体素数,)，
与 betas 逐体素对齐（已验证 subj01=15724 等）。本模块抽取 6 个有信息量的掩膜
（去掉恒真的 nsd_general），存成 (n_voxels, 6) 的 int8 特征矩阵，供编码器做 Key。

用法（先下好 brain_region_masks.hdf5）：
  python data/anatomy.py --brain_masks /path/to/brain_region_masks.hdf5 \
      --out_dir data/anatomy_cache --data_path /path/to/MindEyeV2-main/data
"""
import argparse
import os

import h5py
import numpy as np

REGION_NAMES = ["V1", "V2", "V3", "V4", "early_vis", "higher_vis"]


def build_anatomy_cache(brain_masks_path, out_dir, data_path=None, subj_list=range(1, 9)):
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(brain_masks_path, "r") as f:
        for s in subj_list:
            g = f[f"subj0{s}"]
            n_voxels = g["V1"].shape[0]
            masks = np.stack(
                [g[name][:].astype(np.int8) for name in REGION_NAMES], axis=1
            )  # (n_voxels, 6)

            if data_path is not None:
                with h5py.File(f"{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5", "r") as fb:
                    n_betas = fb["betas"].shape[1]
                assert n_voxels == n_betas, f"subj0{s}: masks {n_voxels} != betas {n_betas}"

            np.savez_compressed(
                os.path.join(out_dir, f"subj0{s}_anatomy.npz"), region_mask=masks
            )
            print(f"subj0{s}: {n_voxels} voxels, region_mask {masks.shape}")


def main():
    ap = argparse.ArgumentParser(description="从 brain_region_masks.hdf5 提取分区特征")
    ap.add_argument("--brain_masks", required=True, help="brain_region_masks.hdf5 路径")
    ap.add_argument("--out_dir", default="data/anatomy_cache")
    ap.add_argument("--data_path", default=None, help="MindEye2 数据根目录，用于校验体素数（可选）")
    ap.add_argument("--subj", type=int, nargs="+", default=list(range(1, 9)))
    args = ap.parse_args()

    build_anatomy_cache(args.brain_masks, args.out_dir, args.data_path, args.subj)


if __name__ == "__main__":
    main()
