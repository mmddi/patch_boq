import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ViewSpecificSpatialAdapter(nn.Module):
    """Lightweight residual branch for ViT tokens.

    Args:
        dim: Token embedding dimension.
        bottleneck_dim: Hidden dimension used by the adapter bottleneck.
        scale: Residual scale applied after the output projection.
        zero_init: When True, initialize the output projection close to zero.
        num_prefix_tokens: Number of non-spatial tokens kept untouched at the
            beginning of the token sequence. For DINOv2 this is usually `1`
            for the CLS token.

    Inputs:
        x: Tensor with shape `[B, N, dim]`.
        spatial_shape: Optional `(H, W)` patch grid. If omitted, the adapter
            falls back to square-grid inference from the number of patch tokens.

    Returns:
        A residual tensor with the same shape `[B, N, dim]`. It is intended to
        be applied as `x = x + adapter(x)`.
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
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be > 0, got {bottleneck_dim}.")

        self.dim = int(dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.scale = float(scale)
        self.zero_init = bool(zero_init)
        self.num_prefix_tokens = int(num_prefix_tokens)

        self.norm = nn.LayerNorm(self.dim)
        self.down_proj = nn.Linear(self.dim, self.bottleneck_dim)
        self.depthwise_conv = nn.Conv2d(
            self.bottleneck_dim,
            self.bottleneck_dim,
            kernel_size=3,
            padding=1,
            groups=self.bottleneck_dim,
        )
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

    @staticmethod
    def _resolve_spatial_shape(
        num_patch_tokens: int,
        spatial_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if spatial_shape is not None:
            height, width = int(spatial_shape[0]), int(spatial_shape[1])
            if height <= 0 or width <= 0:
                raise ValueError(f"spatial_shape must be positive, got {spatial_shape}.")
            if height * width != num_patch_tokens:
                raise ValueError(
                    "spatial_shape does not match the number of patch tokens: "
                    f"{spatial_shape} -> {height * width} vs {num_patch_tokens}."
                )
            return height, width

        inferred_height = int(math.sqrt(num_patch_tokens))
        if inferred_height * inferred_height != num_patch_tokens:
            raise ValueError(
                "Failed to infer a square patch grid from the number of patch tokens: "
                f"{num_patch_tokens}. Pass a valid spatial_shape or use square inputs."
            )
        return inferred_height, inferred_height

    def forward(self, x: torch.Tensor, spatial_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected `[B, N, C]` tokens, got shape {tuple(x.shape)}.")

        batch_size, num_tokens, _ = x.shape
        if num_tokens <= self.num_prefix_tokens:
            return torch.zeros_like(x)

        patch_tokens = x[:, self.num_prefix_tokens :]
        patch_tokens = self.norm(patch_tokens)
        patch_tokens = self.down_proj(patch_tokens)

        height, width = self._resolve_spatial_shape(
            num_patch_tokens=patch_tokens.size(1),
            spatial_shape=spatial_shape,
        )
        patch_tokens = patch_tokens.transpose(1, 2).contiguous().reshape(
            batch_size,
            self.bottleneck_dim,
            height,
            width,
        )
        patch_tokens = self.depthwise_conv(patch_tokens)
        patch_tokens = self.act(patch_tokens)
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2).contiguous()
        patch_tokens = self.up_proj(patch_tokens) * self.scale

        if self.num_prefix_tokens == 0:
            return patch_tokens

        prefix_delta = torch.zeros(
            batch_size,
            self.num_prefix_tokens,
            self.dim,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat((prefix_delta, patch_tokens), dim=1)

