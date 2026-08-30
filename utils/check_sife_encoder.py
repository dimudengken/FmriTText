"""SIFE 消融诊断：encoder 是否真的动了（lr_enc=1e-5 冻结怀疑）。

SIFE 训练里 encoder 只有 lr_enc=1e-5。若 200 步内 encoder 权重几乎没动
（‖enc_sife − enc_stage1‖ ≈ 0），则对抗正则名存实亡，zero_shot ≈ stage2 零样本，
α 消融三组 loss 相同自然无区分度。同时比各组 α 之间 encoder 距离（应≈0）。

判读：
- rel 距离 ~1e-4 以下 = encoder 基本没动 → SIFE 对抗没施加在编码器上；
- 三组 α 相互距离 ≈ 0 但 vs stage1 都非零 = α 无区分度；
- vs stage1 非零且 α 间差异大 = α 有区分度，可继续。

用法（服务器，fmri 环境，项目根目录）：
  python utils/check_sife_encoder.py \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --sife_ckpts checkpoints/stage_sife/alpha1.0/stage_sife.pt \
                   checkpoints/stage_sife/alpha0.5/stage_sife.pt \
                   checkpoints/stage_sife/alpha0.3/stage_sife.pt
"""
import argparse
import torch


def enc_weights(path):
    st = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in st:
        st = st["model"]
    return {k.replace("encoder.", ""): v for k, v in st.items() if k.startswith("encoder.")}


def dist(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0, float("nan"), 0
    l2 = sum(((a[k] - b[k]) ** 2).sum().item() for k in keys) ** 0.5
    norm = sum(a[k].numel() for k in keys) ** 0.5
    return l2, l2 / norm, len(keys)


def main():
    ap = argparse.ArgumentParser(description="SIFE 消融 encoder 移动诊断")
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--sife_ckpts", nargs="+", required=True)
    args = ap.parse_args()

    ref = enc_weights(args.stage1)
    n_param = sum(v.numel() for v in ref.values())
    print(f"stage1 encoder: {n_param:,} params ({len(ref)} tensors)")

    encs = {p: enc_weights(p) for p in args.sife_ckpts}
    for p, e in encs.items():
        l2, rel, n = dist(e, ref)
        if n == 0:
            print(f"[{p}]\n  ** 无 encoder 键（0 tensors），该 ckpt 可能只存 projector/cross_attn）**")
            continue
        print(f"[{p}]\n  vs stage1  L2={l2:.4f} rel={rel:.6f} ({n} tensors)")

    names = list(encs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            l2, rel, n = dist(encs[names[i]], encs[names[j]])
            if n == 0:
                print(f"[{names[i]}] vs [{names[j]}]: 无共同 encoder 键")
                continue
            print(f"[{names[i]}] vs [{names[j]}]: L2={l2:.4f} rel={rel:.6f} ({n})")


if __name__ == "__main__":
    main()
