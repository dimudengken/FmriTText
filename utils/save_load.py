"""checkpoint 读写共享逻辑。

各阶段 ckpt 都只存非 llm 的 BrainLLM 状态（encoder/ridge/projector/cross_attn），
加载时 llm.* 缺失由 strict=False 放行（BrainLLM 里 LLM 本身冻结，不占 ckpt）。
"""
import torch


def load_into(model, path, prefix=None, strict=False):
    """把 {path} 里满足 prefix 前缀的键加载进 model。返回加载的键数。

    prefix: 如 "encoder."、"ridge."、"cross_attn."；None 表示全部键。
    多份 ckpt 的加载顺序：encoder → ridge → projector+cross_attn（stage2）。

    兼容嵌套训练 ckpt（save_train_ckpt 写的 {"model": state_dict}）：直接传 last.pt
    时自动解包，否则会静默加载 0 个张量、模型保持随机初始化——这是既往 eval 结果
    被污染的最可能来源（S1 评估 cos(real,zero)=0.84 与 diag_stage1 的 0.05 矛盾）。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    keys = {k: v for k, v in ckpt.items() if prefix is None or k.startswith(prefix)}
    model.load_state_dict(keys, strict=strict)
    return len(keys)


def save(state_dict, path):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(ckpt, path)
    return path


def _train_state(model):
    """训练 ckpt 只存非 llm 冻结权重 + LoRA（llm 基座冻结，续训时从预训练路径重新加载）。"""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if not k.startswith("llm.") or "lora_" in k}


def save_train_ckpt(model, opt, epoch, global_step, path, extra_state=None):
    """滚动保存训练进度（模型+优化器+步数），覆盖写，供 --resume 续训。"""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {"epoch": epoch, "global_step": global_step,
            "model": _train_state(model), "optimizer": opt.state_dict()}
    if extra_state:
        ckpt["extra"] = {k: v.detach().cpu() for k, v in extra_state.items()}
    torch.save(ckpt, path)
    return path


def load_train_ckpt(model, opt, path, extra_module=None):
    """从训练 ckpt 恢复模型/优化器/进度；返回 (epoch, global_step)。

    优化器状态恢复后按当前设备重放（save 时存的是 GPU 张量，load 时在 CPU）。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"], strict=False)
    if "optimizer" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer"])
            dev = next(model.parameters()).device
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(dev)
        except (ValueError, RuntimeError) as e:
            # 架构变化（如 aux_head 增删）→ 参数组不匹配，优化器状态作废，
            # 模型权重照常加载，优化器从零起步（动量丢失可接受）。
            print(f"[warn] optimizer state skipped: {e}", flush=True)
    if extra_module is not None and "extra" in ckpt:
        extra_module.load_state_dict(ckpt["extra"], strict=False)
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)
