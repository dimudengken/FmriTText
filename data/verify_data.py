"""冒烟测试：核对 MindEye2 数据目录与数据层代码的假设是否一致。

只做只读检查，不改数据、不下载。跑通后训练时 data_path 的写法和
preprocessing.py 里的文件名假设就都可信了。

用法（在服务器上、fMRI-LM2T-XS 仓库根目录）：
  python data/verify_data.py --data_path /path/to/MindEye2V2-main/data --subj 1

可选 caption 检查（给了 NSD stim info csv 和 COCO captions json 才跑）：
  python data/verify_data.py --data_path ... \
      --nsd_stim_info /path/to/nsd_stim_info_merged.csv \
      --coco_captions /path/to/captions_train2017.json /path/to/captions_val2017.json

注意：本脚本【不】全量加载 betas/images（会 OOM），只读 shape/dtype/首行切片。
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import preprocessing  # noqa: E402


def check(label, fn):
    try:
        fn()
        print(f"  [OK]   {label}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--nsd_stim_info", default=None)
    ap.add_argument("--coco_captions", nargs="+", default=None,
                    help="COCO captions json，可传多个（train2017 + val2017）")
    args = ap.parse_args()

    dp = args.data_path.rstrip("/")
    s = args.subj
    print(f"data_path = {dp}\nsubj = {s}\n")

    meta = {}  # 跨检查共享：n_trials / n_voxels / shape

    # 1. 目录是否存在
    print("== 目录 ==")
    check("data_path 存在", lambda: os.path.isdir(dp) or (_ for _ in ()).throw(FileNotFoundError(dp)))
    for p in ["wds", f"wds/subj0{s}"]:
        check(f"{p} 存在", lambda p=p: os.path.isdir(os.path.join(dp, p))
              or (_ for _ in ()).throw(FileNotFoundError(os.path.join(dp, p))))

    # 2. betas（只读元信息 + 首行切片，不全量加载）
    print("\n== 体素 betas ==")
    betas_path = f"{dp}/betas_all_subj0{s}_fp32_renorm.hdf5"
    check(f"betas 文件存在 {os.path.basename(betas_path)}",
          lambda: os.path.isfile(betas_path) or (_ for _ in ()).throw(FileNotFoundError(betas_path)))

    def load_b_meta():
        with h5py.File(betas_path, "r") as f:
            keys = list(f.keys())
            print(f"        hdf5 keys = {keys}")
            if "betas" not in f:
                raise KeyError(f"未找到 'betas' key，实际为 {keys}")
            ds = f["betas"]
            meta["shape"] = ds.shape
            meta["dtype"] = ds.dtype
            meta["n_trials"] = ds.shape[0]
            meta["n_voxels"] = ds.shape[-1]
            print(f"        betas shape = {ds.shape}  dtype = {ds.dtype}")
            print(f"        n_trials = {ds.shape[0]}, n_voxels = {ds.shape[-1]}")
            row = np.asarray(ds[0])
            print(f"        首行 shape = {row.shape}, 值域 ≈ [{row.min():.2f}, {row.max():.2f}]")
            if ds.shape[0] > 0 and ds.ndim == 2 and ds.shape[1] < 100:
                raise AssertionError(f"shape 可疑：第 2 维只有 {ds.shape[1]}，不像体素")
    check("betas 元信息", load_b_meta)

    # 3. 图像 hdf5（只读 shape，不加载）
    print("\n== 图像 coco_images ==")
    img_path = f"{dp}/coco_images_224_float16.hdf5"
    check(f"images 文件存在 {os.path.basename(img_path)}",
          lambda: os.path.isfile(img_path) or (_ for _ in ()).throw(FileNotFoundError(img_path)))

    def load_img_meta():
        with h5py.File(img_path, "r") as f:
            keys = list(f.keys())
            print(f"        hdf5 keys = {keys}")
            if "images" not in f:
                raise KeyError(f"未找到 'images' key，实际为 {keys}")
            ds = f["images"]
            print(f"        images shape = {ds.shape}  dtype = {ds.dtype}")
    check("images 元信息", load_img_meta)

    # 4. trial 索引（从 wds tar 现场建）
    print("\n== trial 索引 (wds tar → behav.npy) ==")
    for split in ["train", "test", "new_test"]:
        def build(split=split):
            ii, vi = preprocessing.build_trial_index_from_wds(dp, s, split)
            print(f"        {split}: {len(ii)} trials, "
                  f"image_idx∈[{ii.min()},{ii.max()}], voxel_idx∈[{vi.min()},{vi.max()}]")
            if "n_trials" in meta and vi.max() >= meta["n_trials"]:
                raise AssertionError(
                    f"voxel_idx 越界: max {vi.max()} >= betas n_trials {meta['n_trials']}")
        check(f"build_trial_index_from_wds({split})", build)

    # 5. caption 索引（可选）
    if args.nsd_stim_info and args.coco_captions:
        print("\n== caption 索引 ==")
        def build_cap():
            caps = preprocessing.build_caption_index(args.nsd_stim_info, args.coco_captions)
            non_empty = sum(1 for c in caps if c)
            print(f"        captions_by_nsd_idx 长度 = {len(caps)}, 非空 = {non_empty} / {len(caps)}")
            if non_empty == 0:
                raise AssertionError("没有任何 caption 映射成功，检查 nsdId/cocoId 对齐")
        check("build_caption_index", build_cap)
    else:
        print("\n== caption 索引 ==   (跳过：未提供 --nsd_stim_info / --coco_captions)")

    print("\n完成。")


if __name__ == "__main__":
    main()
