"""Cluster-restricted attention with the original MindLLM MLP/token head.

Each functional cluster owns an equal number of attention queries. Subsequent
MLP layers can mix clusters: the final LLM tokens are NOT one-token-per-cluster.
Parameter names and shapes match the original encoder for checkpoint transfer.
"""

import logging

import torch
from torch import nn

from src.modules.neuro_informed_attn import (
    NeuroscienceInformedAttention,
    NeuroscienceInformedAttentionLayer,
)

logger = logging.getLogger(__name__)


class ClusterRestrictedAttention(nn.Module):
    def __init__(self, query_embeddings, assignments, n_clusters):
        super().__init__()
        self.query_embeddings = query_embeddings
        self.size = query_embeddings.size(1)
        self.rank = query_embeddings.size(2)
        self.n_clusters = n_clusters
        if n_clusters < 1 or self.size % n_clusters:
            raise ValueError("hidden_dim must be divisible by n_clusters")
        self.queries_per_cluster = self.size // n_clusters
        self.subjects = set(assignments)
        for subject, labels in assignments.items():
            if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
                raise ValueError(f"{subject}: cluster_ids must be a nonempty int64 vector")
            if labels.min() < 0 or labels.max() >= n_clusters:
                raise ValueError(f"{subject}: cluster_ids outside [0, n_clusters)")
            self.register_buffer(f"cluster_ids_{subject}", labels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Buffers saved in a clustered checkpoint must describe the configured
        # bank. A baseline checkpoint has no such keys and can initialize queries.
        for name, value in self.named_buffers(recurse=False):
            saved = state_dict.get(prefix + name)
            if saved is not None and not torch.equal(saved.cpu(), value.cpu()):
                error_msgs.append(f"{prefix}{name}: checkpoint cluster bank differs from configured bank")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, values, keys, subject):
        if subject not in self.subjects:
            raise ValueError(f"cluster bank has no assignments for {subject}")
        labels = getattr(self, f"cluster_ids_{subject}")
        if values.ndim != 2 or values.size(1) != labels.numel():
            raise ValueError(f"{subject}: voxel width differs from cluster bank")
        if keys.shape != (values.size(0), labels.numel(), self.rank):
            raise ValueError("voxel keys must have shape [batch, voxels, rank]")
        if self.n_clusters == 1:
            # Exact baseline ablation, including its floating-point reduction.
            return NeuroscienceInformedAttentionLayer.forward(self, values, keys)
        query = self.query_embeddings[0].reshape(self.n_clusters, self.queries_per_cluster, self.rank)
        # Keep the baseline's unscaled query/key dot product. Only the set of
        # visible voxels changes. Segment reductions also avoid padding every
        # cluster to the size of the largest cluster (which can be very uneven).
        logits = torch.einsum("vqr,bvr->bvq", query[labels], keys).float()
        index = labels[None, :, None].expand_as(logits)
        shape = (values.size(0), self.n_clusters, self.queries_per_cluster)
        maxima = logits.new_full(shape, -torch.inf)
        maxima.scatter_reduce_(1, index, logits.detach(), reduce="amax", include_self=True)
        unnormalized = (logits - maxima.gather(1, index)).exp()
        denominator = logits.new_zeros(shape).scatter_add(1, index, unnormalized)
        weights = unnormalized / denominator.gather(1, index)
        pooled = logits.new_zeros(shape).scatter_add(
            1, index, weights * values.float().unsqueeze(-1)
        )
        # Absent clusters have no scatter sources and remain exactly zero.
        return pooled.reshape(values.size(0), self.size)


class ClusteredNeuroscienceInformedAttention(NeuroscienceInformedAttention):
    def __init__(self, cluster_path, n_clusters=128, **kwargs):
        if not cluster_path:
            raise ValueError("cluster_path is required; build a training-only cluster bank first")
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if kwargs.get("pe_method", "gauss") not in ("gauss", "coord"):
            raise ValueError("clustered encoder supports gauss and coord positional encoding")
        bank = torch.load(cluster_path, map_location="cpu", weights_only=True)
        if bank.get("version") != 1 or bank.get("n_clusters") != n_clusters or not bank.get("subjects"):
            raise ValueError("cluster bank version, n_clusters, or subjects are invalid")
        if kwargs.get("hidden_dim", 1024) % n_clusters:
            raise ValueError("hidden_dim must be divisible by n_clusters")
        super().__init__(**kwargs)
        self.cluster_path = str(cluster_path)
        self.cluster_method = bank.get("method")
        assignments = {}
        for subject, entry in bank["subjects"].items():
            if not hasattr(self, f"visual_mask_{subject}"):
                raise ValueError(f"unsupported subject in cluster bank: {subject}")
            mask = self.get_mask([subject])
            expected = torch.where(mask.reshape(-1))[0].cpu()
            if (list(mask.shape) != entry["mask_shape"]
                    or not torch.equal(expected, entry["flat_indices"].cpu())
                    or expected.numel() != entry["cluster_ids"].numel()):
                raise ValueError(f"{subject}: cluster bank voxel order/mask differs from encoder")
            assignments[subject] = entry["cluster_ids"]
        self.neuro_informed_attn = ClusterRestrictedAttention(
            self.neuro_informed_attn.query_embeddings, assignments, n_clusters
        )
        logger.info("[clustered-encoder] method=%s clusters=%d queries_per_cluster=%d bank=%s",
                    self.cluster_method, n_clusters,
                    self.neuro_informed_attn.queries_per_cluster, cluster_path)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for subject in self.neuro_informed_attn.subjects:
            name = f"visual_mask_{subject}"
            saved = state_dict.get(prefix + name)
            if saved is not None and not torch.equal(saved.cpu(), getattr(self, name).cpu()):
                error_msgs.append(f"{prefix}{name}: checkpoint mask differs from cluster bank mask")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, voxels, **kwargs):
        subjects = kwargs["subject"]
        if not isinstance(subjects, (list, tuple)) or not subjects or len(set(subjects)) != 1:
            raise ValueError("a batch must contain one subject, supplied as a list")
        subject = subjects[0]
        if subject not in self.neuro_informed_attn.subjects:
            raise ValueError(f"cluster bank has no assignments for {subject}")
        mask = self.get_mask(list(subjects))
        batch_size = voxels.size(0)
        if voxels.dim() > 2:
            if tuple(voxels.shape[1:]) != tuple(mask.shape):
                raise ValueError("volume input shape differs from the subject mask")
            voxels = voxels.masked_select(mask.unsqueeze(0)).view(batch_size, -1)
        coords = torch.stack(torch.where(mask)).transpose(0, 1)
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        if self.pe_method == "coord":
            coords = self.coords_transform((coords.float() + 0.5) / 50.0)
        else:
            coords = self.coords_transform(self.coords_encoding(coords))
        roi_embeds = []
        for roi_name, roi_embeddings in self.roi_embeds.items():
            roi = getattr(self, f"roi_{roi_name}_{subject}")
            roi_embeds.append(roi_embeddings[roi[mask]])
        roi_embeds = torch.cat(roi_embeds, dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.region_feature_project(torch.cat([coords, roi_embeds], dim=-1))
        x = self.neuro_informed_attn(voxels, keys, subject)
        x = self.neuro_informed_attn_post(x)
        residual = x
        for res_block in range(self.n_mlp_layers):
            x = self.mlp[res_block](x)
            x = x + residual
            residual = x
        x = self.head(x.reshape(batch_size, -1))
        return x.reshape(batch_size, self.n_fmri_tokens, -1)
