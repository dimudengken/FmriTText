"""持久交叉注意力适配器（BIT-LLM）。

fMRI token 作为 KV memory 贯穿生成全程，每层可重新检索神经证据。
门控残差 H' = H + α * CrossAttn(Q=Norm(H)WQ, K=BWK, V=BWV)，α 初始 0.08；
attention logit 温度 T=1.3（不除以 sqrt(head_dim)，用温度代替缩放）。
"""
import torch
from torch import nn


class CrossAttentionAdapter(nn.Module):
    """单个 gated cross-attention 适配器。"""

    def __init__(self, dim=3072, n_heads=32, gate_init=0.08, temperature=1.3,
                 o_proj_init=0.0, gate_mode="scalar"):
        super().__init__()
        assert dim % n_heads == 0, f"{dim} % {n_heads} != 0"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.temperature = temperature
        self.gate_mode = gate_mode
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        # 默认零初始化输出投影：初始输出严格为 0（纯残差），LLM 不被随机噪声扰动。
        # 否则随机 o_proj 注入噪声抬高 loss，优化器最省事的降 loss 方式是压零脑
        # 路径输出（"关掉"它），接口增益永久钉在 ~0。零初始化迫使 cross-attn
        # 从干净起点学"用脑信息降 loss"（同 ResNet/LoRA 的输出零初始化技巧）。
        # o_proj_init>0（实验开关）：极小非零高斯初始化（如 1e-3），给训练早期
        # projector 一条梯度回流通道（否则须等 o_proj 长出非零权重梯度才回流）。
        # 权衡：过大随机噪声诱发优化器压零脑路径。
        if o_proj_init and o_proj_init > 0:
            nn.init.normal_(self.o_proj.weight, std=o_proj_init)
        else:
            nn.init.zeros_(self.o_proj.weight)
        if gate_mode == "per_token":
            # 逐脑 token value gate：sigmoid(Linear(brain)) → (B, N, 1)，0~1。
            # 每个脑 token 自己的门控值，替换全局标量 gate。训练时 stage2 读
            # last_gate 加 (g-0.5)² 正则防坍缩到 0/1（0=丢弃脑信号，1=全保留）。
            self.gate_net = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        else:
            self.gate = nn.Parameter(torch.tensor(gate_init))
        self.last_gate = None  # 前向时存本 batch 的 per-token gate，供训练脚本读去加正则

    def forward(self, hidden_states, brain_tokens):
        # hidden_states: (B, L, dim)  文本隐藏态
        # brain_tokens:  (B, N, dim)  fMRI token（projector 输出）
        B, L, D = hidden_states.shape
        N = brain_tokens.shape[1]
        self.last_gate = None

        q = self.q_proj(self.norm(hidden_states))  # (B, L, D)
        k = self.k_proj(brain_tokens)              # (B, N, D)
        v = self.v_proj(brain_tokens)              # (B, N, D)

        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, L, hd)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B, nh, N, hd)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if self.gate_mode == "per_token":
            g = self.gate_net(brain_tokens)        # (B, N, 1)
            self.last_gate = g
            v = v * g.unsqueeze(1)                 # (B, nh, N, hd) * (B, 1, N, 1)

        attn = (q @ k.transpose(-2, -1)) / self.temperature  # (B, nh, L, N)
        attn = attn.softmax(-1)
        out = attn @ v                                      # (B, nh, L, hd)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)

        if self.gate_mode == "per_token":
            return hidden_states + out
        return hidden_states + self.gate * out


class CrossAttentionStack(nn.Module):
    """多个 adapter 的容器，按层索引取用。"""

    def __init__(self, dim=3072, n_heads=32, layer_indices=(3, 7, 11, 15),
                 gate_init=0.08, temperature=1.3, o_proj_init=0.0, gate_mode="scalar"):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.adapters = nn.ModuleDict({
            str(i): CrossAttentionAdapter(dim, n_heads, gate_init, temperature,
                                          o_proj_init=o_proj_init, gate_mode=gate_mode)
            for i in layer_indices
        })

    def __getitem__(self, layer_idx):
        return self.adapters[str(layer_idx)]
