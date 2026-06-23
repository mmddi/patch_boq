from typing import Optional, Tuple

import torch
import torch.nn as nn


class DinoV2BlockAdapter(nn.Module):
    """MLP adapter for DINOv2 block tokens.

    The adapter keeps the token-flow interface `[B, N, C] -> [B, N, C]`.
    `num_prefix_tokens` is preserved for interface compatibility so the module
    can be used with or without CLS/prefix tokens.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        scale: float = 0.1,
        zero_init: bool = True,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}.")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be > 0, got {bottleneck_dim}.")
        if num_prefix_tokens < 0:
            raise ValueError(f"num_prefix_tokens must be >= 0, got {num_prefix_tokens}.")

        self.dim = int(dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.scale = float(scale)
        self.zero_init = bool(zero_init)
        self.num_prefix_tokens = int(num_prefix_tokens)

        # MLP adapter internals.
        self.down_proj = nn.Linear(self.dim, self.bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(self.bottleneck_dim, self.dim)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.zero_init:
            nn.init.zeros_(self.up_proj.weight)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
        else:
            nn.init.trunc_normal_(self.up_proj.weight, std=1e-3)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        del spatial_shape
        if x.ndim != 3:
            raise ValueError(f"Expected `[B, N, C]` tokens, got shape {tuple(x.shape)}.")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected token dim {self.dim}, got {x.shape[-1]}.")

        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.scale
