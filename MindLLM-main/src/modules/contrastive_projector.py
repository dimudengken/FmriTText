import torch.nn as nn
import torch.nn.functional as F


class BrainContrastiveProjector(nn.Module):
    """Pool the brain tokens and project them into the frozen target's embedding space.

    Pooling is a plain mean because the encoder emits no token mask: all tokens are
    learned queries attending over every voxel, so every token is always valid and a
    masked mean would be the identical operation.
    """

    def __init__(self, brain_dim: int = 1024, projection_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(brain_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, brain_tokens):
        """[B, L, brain_dim] -> [B, projection_dim], L2-normalized."""
        return F.normalize(self.proj(brain_tokens.mean(dim=1)), dim=-1)
