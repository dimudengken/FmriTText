"""Opt-in, checkpoint-compatible two-pass voxel readout for MindLLM.

The first pass is the original unscaled spatial attention. A low-rank query
update depends on that sample's first readout; the second pass reuses voxel K/V.
No LLM adapters, auxiliary objectives, or changes to the token head are needed.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.modules.neuro_informed_attn import (
    NeuroscienceInformedAttention,
    NeuroscienceInformedAttentionLayer,
)


class AdaptiveReadoutAttention(nn.Module):
    def __init__(self, query_embeddings, mode="adaptive", adapter_rank=16,
                 gate_init=0.1, query_chunk_size=128, checkpoint_second_pass=True):
        super().__init__()
        if mode not in ("adaptive", "mlp_control"):
            raise ValueError("mode must be adaptive or mlp_control")
        if adapter_rank < 1 or query_chunk_size < 1:
            raise ValueError("adapter_rank and query_chunk_size must be positive")
        if not math.isfinite(gate_init) or not 0 < abs(gate_init) < 1:
            raise ValueError("gate_init must be finite, nonzero, and inside (-1, 1)")
        self.query_embeddings = query_embeddings  # retain the baseline checkpoint key
        self.size, self.rank = query_embeddings.shape[1:]
        self.mode = mode
        self.query_chunk_size = query_chunk_size
        self.checkpoint_second_pass = checkpoint_second_pass
        self.register_buffer("variant_id", torch.tensor(1 if mode == "adaptive" else 2))
        self.context_norm = nn.LayerNorm(self.size)
        self.gate = nn.Parameter(torch.tensor(math.atanh(gate_init)))
        if mode == "adaptive":
            self.context_projection = nn.Linear(self.size, adapter_rank)
            self.query_factors = nn.Parameter(torch.empty(self.size, adapter_rank))
            self.key_factors = nn.Parameter(torch.empty(adapter_rank, self.rank))
            nn.init.xavier_uniform_(self.query_factors)
            nn.init.xavier_uniform_(self.key_factors)
            # Delta Q starts at zero while a nonzero gate lets this projection
            # receive a gradient on step one. Do NOT zero the gate as well.
            nn.init.zeros_(self.context_projection.weight)
            nn.init.zeros_(self.context_projection.bias)
        else:
            self.mlp_down = nn.Linear(self.size, adapter_rank)
            self.mlp_up = nn.Linear(adapter_rank, self.size)
            nn.init.zeros_(self.mlp_up.weight)
            nn.init.zeros_(self.mlp_up.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "variant_id")
        if saved is not None and not torch.equal(saved.cpu(), self.variant_id.cpu()):
            error_msgs.append(prefix + "variant_id: checkpoint uses a different readout mode")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _correction(query, delta_query, keys, values):
        # Subtract the SAME chunked first-pass computation to make delta=0
        # exactly zero even under autocast or a different GEMM chunk size.
        # Do not scale logits: preserve MindLLM's original attention convention.
        logits = query @ keys.transpose(1, 2)
        delta_logits = delta_query @ keys.transpose(1, 2)
        weights0 = logits.softmax(-1)
        weights1 = (logits + delta_logits).softmax(-1)
        return ((weights1 - weights0) @ values.unsqueeze(-1)).squeeze(-1)

    def forward(self, values, keys=None, context=None):
        if values.ndim != 2 or keys is None or keys.shape != (
                values.size(0), values.size(1), self.rank):
            raise ValueError("expected values [B,N] and keys [B,N,rank]")
        first = NeuroscienceInformedAttentionLayer.forward(self, values, keys)
        features = self.context_norm(first)
        if self.mode == "mlp_control":
            correction = self.mlp_up(torch.nn.functional.gelu(self.mlp_down(features)))
        else:
            coefficients = torch.tanh(self.context_projection(features))
            chunks = []
            for start in range(0, self.size, self.query_chunk_size):
                end = min(start + self.query_chunk_size, self.size)
                delta_query = (coefficients[:, None, :] * self.query_factors[start:end]) @ self.key_factors
                args = (self.query_embeddings[:, start:end], delta_query, keys, values)
                if self.checkpoint_second_pass and self.training and torch.is_grad_enabled():
                    chunk = checkpoint(self._correction, *args, use_reentrant=False)
                else:
                    chunk = self._correction(*args)
                chunks.append(chunk)
            correction = torch.cat(chunks, dim=1)
        return first + self.gate.tanh() * correction


class AdaptiveNeuroscienceInformedAttention(NeuroscienceInformedAttention):
    def __init__(self, mode="adaptive", adapter_rank=16, gate_init=0.1,
                 query_chunk_size=128, checkpoint_second_pass=True, **kwargs):
        super().__init__(**kwargs)
        # Adding a module must not change projector initialization or data RNG.
        # Base construction is unchanged; isolate only the new CPU parameters.
        with torch.random.fork_rng(devices=[]):
            self.neuro_informed_attn = AdaptiveReadoutAttention(
                self.neuro_informed_attn.query_embeddings, mode=mode,
                adapter_rank=adapter_rank, gate_init=gate_init,
                query_chunk_size=query_chunk_size,
                checkpoint_second_pass=checkpoint_second_pass,
            )
