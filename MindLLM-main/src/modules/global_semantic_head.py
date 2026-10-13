import torch.nn.functional as F
from torch import nn


class GlobalSemanticHead(nn.Module):
    """Map the fMRI brain tokens onto the frozen semantic target space.

    Pools the `[B, L, D]` brain tokens into a single `[B, out_dim]` embedding and
    L2-normalizes it, so it can be compared with the precomputed target bank by a
    plain cosine/dot product.
    """

    def __init__(self, brain_dim, out_dim, hidden_dim=None, pooling="mean"):
        super().__init__()
        assert pooling == "mean", f"unsupported pooling: {pooling}"
        self.pooling = pooling

        layers = []
        in_dim = brain_dim
        if hidden_dim is not None:
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, out_dim)]
        self.proj = nn.Sequential(*layers)

    def forward(self, brain_tokens):
        # brain_tokens: [B, L, D]
        x = brain_tokens.mean(dim=1)
        return F.normalize(self.proj(x), dim=-1)
