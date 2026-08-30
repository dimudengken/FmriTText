"""完整 fMRI-to-LLM 模型：encoder → ridge → projector → cross-attn → frozen LLM。

数据流：
  原生体素(B, n_voxels) → KeyValueEncoder → (B, 128, 1024)
    → SubjectRidge(每被试 1024→1024) → (B, 128, 1024)
    → Projector(1024→3072 + RMSNorm) → (B, 128, 3072)
    → Cross-Attn 注入 Llama 第 {3,7,11,15} 层（门控残差 α=0.08, 温度 1.3）
    → Frozen Llama → logits

LLM 冻结；训练时只更新 encoder + ridge + projector + cross-attn adapter。
cross-attn 用 forward hook 挂在目标层输出处，brain tokens 通过 holder 传递。
"""
import torch
from torch import nn
from transformers import AutoModelForCausalLM

from .fmri_encoder import KeyValueEncoder
from .ridge import SubjectRidge
from .projector import Projector
from .cross_attention import CrossAttentionStack


class BrainLLM(nn.Module):
    def __init__(self, llm_path, n_subjects=8, encoder_kwargs=None,
                 hidden_dim=3072, n_heads=32, adapter_layers=(3, 7, 11, 15),
                 gate_init=0.08, temperature=1.3, torch_dtype=torch.bfloat16,
                 proj_mode="rmsnorm", o_proj_init=0.0, gate_mode="scalar"):
        super().__init__()
        self.encoder = KeyValueEncoder(**(encoder_kwargs or {}))
        self.ridge = SubjectRidge(n_subjects=n_subjects, dim=1024)
        self.projector = Projector(in_dim=1024, out_dim=hidden_dim, mode=proj_mode)

        self.llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch_dtype)
        self._dtype = torch_dtype
        for p in self.llm.parameters():
            p.requires_grad = False

        self.cross_attn = CrossAttentionStack(
            dim=hidden_dim, n_heads=n_heads, layer_indices=adapter_layers,
            gate_init=gate_init, temperature=temperature,
            o_proj_init=o_proj_init, gate_mode=gate_mode,
        ).to(torch_dtype)

        self._brain_tokens = None
        self._enc_sem = None
        self._register_hooks()

    def _register_hooks(self):
        for i, layer in enumerate(self.llm.model.layers):
            if i in self.cross_attn.layer_indices:
                adapter = self.cross_attn[i]

                def make_hook(adapter):
                    def hook(module, args, kwargs, output):
                        brain = self._brain_tokens
                        if brain is None:
                            return output
                        # transformers >= 4.57 层直接返回裸 hidden_states；
                        # 旧版返回 (hidden_states, past_key_value, ...) tuple
                        is_tuple = isinstance(output, tuple)
                        hidden_states = output[0] if is_tuple else output
                        hidden_states = adapter(hidden_states, brain)
                        if is_tuple:
                            return (hidden_states,) + tuple(output[1:])
                        return hidden_states
                    return hook

                layer.register_forward_hook(make_hook(adapter), with_kwargs=True)

    def _brain_forward(self, voxels, subj, use_ridge=True):
        x = self.encoder(voxels, subj)   # (B, 128, 1024)
        if use_ridge:                    # 零样本臂跳过 ridge（SIFE 主体不变特征）
            x = self.ridge(x, subj)
        self._enc_sem = x.mean(dim=1)    # post-ridge pooled (B, 1024)，stage1 head 的输入（#4 辅助对齐用）
        x = self.projector(x)            # (B, 128, hidden_dim)
        return x.to(self._dtype)         # 用存的 dtype；LLM 包了 peft 后未必有 .dtype

    def forward(self, voxels, subj, input_ids, attention_mask=None, labels=None,
                use_ridge=True, return_brain_sem=False):
        # 注意：不要给 self.llm 开 gradient checkpointing——重算阶段 hook 会再次
        # 读取 _brain_tokens（此处已置 None），要么报"backward a second time"，
        # 要么把 brain tokens 当常量吞掉 projector 的梯度。
        brain_tokens = self._brain_forward(voxels, subj, use_ridge=use_ridge)
        self._brain_tokens = brain_tokens
        try:
            out = self.llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        finally:
            self._brain_tokens = None
        if return_brain_sem:
            # 投影器输出的 128 token 语义汇总 + post-ridge pooled 编码器特征
            # （后者是 stage1 ContrastiveHead 的输入，供 #4 语义继承 MSE 对齐）
            return out, brain_tokens.mean(dim=1), self._enc_sem
        return out

    @torch.no_grad()
    def generate(self, voxels, subj, input_ids, attention_mask=None,
                 max_new_tokens=128, use_ridge=True, **gen_kwargs):
        self._brain_tokens = self._brain_forward(voxels, subj, use_ridge=use_ridge)
        try:
            return self.llm.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_new_tokens=max_new_tokens, **gen_kwargs)
        finally:
            self._brain_tokens = None
