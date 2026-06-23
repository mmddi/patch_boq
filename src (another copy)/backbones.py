# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# https://github.com/amaralibey/Bag-of-Queries
#
# See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import importlib
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torchvision

try:
    from src.dinov2_block_adapter import DinoV2BlockAdapter
except ModuleNotFoundError:
    from dinov2_block_adapter import DinoV2BlockAdapter


class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        pretrained=True,
        unfreeze_n_blocks=2,
        reshape_output=True,
        use_block_adapter=False,
        adapter_num_prefix_tokens=None,
        adapter_bottleneck_dim=128,
        adapter_scale=0.1,
        adapter_zero_init=True,
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        self.freeze_backbone = False
        self.use_block_adapter = bool(use_block_adapter)
        self.supports_freeze_backbone_forward = True
        self.adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_zero_init = bool(adapter_zero_init)
        
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"                             
                
        self.dino = self._load_dino_model(self.backbone_name, pretrained=pretrained)
        self.out_channels = self.dino.embed_dim
        self.adapter_num_prefix_tokens = (
            self.num_prefix_tokens if adapter_num_prefix_tokens is None else int(adapter_num_prefix_tokens)
        )
        if self.adapter_num_prefix_tokens < 0:
            raise ValueError(
                f"adapter_num_prefix_tokens must be >= 0, got {self.adapter_num_prefix_tokens}."
            )
        self.block_adapters = nn.ModuleList(
            self._make_block_adapter() for _ in range(self.num_blocks)
        ) if self.use_block_adapter else nn.ModuleList()
        self.set_trainable_layers(freeze_backbone=False, unfreeze_n_blocks=unfreeze_n_blocks)

    @staticmethod
    def _load_dino_model(backbone_name, pretrained=True):
        local_repo = Path(__file__).resolve().parents[1] / "dinov2_src"
        local_hubconf = local_repo / "hubconf.py"
        if local_hubconf.exists():
            return torch.hub.load(
                str(local_repo),
                backbone_name,
                source="local",
                pretrained=pretrained,
            )

        try:
            dinov2_backbones = importlib.import_module("dinov2.hub.backbones")
            builder = getattr(dinov2_backbones, backbone_name)
            return builder(pretrained=pretrained)
        except (ImportError, AttributeError):
            return torch.hub.load(
                "facebookresearch/dinov2",
                backbone_name,
                pretrained=pretrained,
            )
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches

    @property
    def num_blocks(self):
        return len(self.dino.blocks)

    @property
    def num_prefix_tokens(self):
        return 1 + int(getattr(self.dino, "num_register_tokens", 0))

    def _make_block_adapter(self):
        return DinoV2BlockAdapter(
            dim=self.out_channels,
            bottleneck_dim=self.adapter_bottleneck_dim,
            scale=self.adapter_scale,
            zero_init=self.adapter_zero_init,
            num_prefix_tokens=self.adapter_num_prefix_tokens,
        )

    def get_adapter_state_keys(self):
        if not self.use_block_adapter:
            return []

        state_keys = []
        for block_idx, adapter in enumerate(self.block_adapters):
            for key in adapter.state_dict().keys():
                state_keys.append(f"block_adapters.{block_idx}.{key}")
        return state_keys

    def get_missing_state_dict_keys_to_ignore(self):
        return self.get_adapter_state_keys()

    def set_trainable_layers(self, freeze_backbone=None, unfreeze_n_blocks=None):
        if unfreeze_n_blocks is not None:
            if not isinstance(unfreeze_n_blocks, int) or unfreeze_n_blocks < 0 or unfreeze_n_blocks > self.num_blocks:
                raise ValueError(
                    f"unfreeze_n_blocks must be an integer in [0, {self.num_blocks}], "
                    f"got {unfreeze_n_blocks!r}."
                )
            self.unfreeze_n_blocks = int(unfreeze_n_blocks)

        if freeze_backbone is not None:
            self.freeze_backbone = bool(freeze_backbone)

        for param in self.dino.parameters():
            param.requires_grad = False

        trainable_blocks = 0 if self.freeze_backbone else min(self.unfreeze_n_blocks, self.num_blocks)
        if trainable_blocks > 0:
            for block in self.dino.blocks[-trainable_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        if self.use_block_adapter:
            for param in self.block_adapters.parameters():
                param.requires_grad = True

        return trainable_blocks

    def get_trainable_block_start(self, force_freeze=False):
        trainable_blocks = 0 if (self.freeze_backbone or force_freeze) else min(self.unfreeze_n_blocks, self.num_blocks)
        return self.num_blocks - trainable_blocks

    def prepare_tokens(self, x):
        return self.dino.prepare_tokens_with_masks(x)

    def tokens_to_feature_map(self, tokens, batch_size, height, width):
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        if not self.reshape_output:
            return patch_tokens

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input size {(height, width)} is not divisible by patch_size={self.patch_size}."
            )
        _, _, channels = patch_tokens.shape
        return patch_tokens.permute(0, 2, 1).reshape(
            batch_size,
            channels,
            height // self.patch_size,
            width // self.patch_size,
        )

    @staticmethod
    @contextmanager
    def _temporarily_freeze_block_params(block):
        """Freeze DINO block params for one forward while keeping adapter grads alive."""
        trainable_params = [param for param in block.parameters() if param.requires_grad]
        if not trainable_params:
            yield
            return

        try:
            for param in trainable_params:
                param.requires_grad_(False)
            yield
        finally:
            for param in trainable_params:
                param.requires_grad_(True)

    @staticmethod
    def _drop_add_residual_stochastic_depth(x, residual_func, sample_drop_ratio):
        batch_size = x.shape[0]
        sample_subset_size = max(int(batch_size * (1 - sample_drop_ratio)), 1)
        brange = torch.randperm(batch_size, device=x.device)[:sample_subset_size]
        residual = residual_func(x[brange])

        x_flat = x.flatten(1)
        residual_flat = residual.flatten(1)
        residual_scale_factor = batch_size / sample_subset_size
        x_plus_residual = torch.index_add(
            x_flat,
            0,
            brange,
            residual_flat.to(dtype=x.dtype),
            alpha=residual_scale_factor,
        )
        return x_plus_residual.view_as(x)

    def _forward_block_with_parallel_adapter(self, block, tokens, adapter, spatial_shape):
        """Parallel residual branch: attn(norm1(x)) + adapter(norm1(x))."""

        def attn_plus_adapter_residual_func(x):
            x_norm = block.norm1(x)
            attn_out = block.attn(x_norm)
            adapter_out = adapter(x_norm, spatial_shape=spatial_shape)
            return block.ls1(attn_out + adapter_out)

        def ffn_residual_func(x):
            return block.ls2(block.mlp(block.norm2(x)))

        sample_drop_ratio = float(getattr(block, "sample_drop_ratio", 0.0))
        if block.training and sample_drop_ratio > 0.1:
            tokens = self._drop_add_residual_stochastic_depth(
                tokens,
                residual_func=attn_plus_adapter_residual_func,
                sample_drop_ratio=sample_drop_ratio,
            )
            tokens = self._drop_add_residual_stochastic_depth(
                tokens,
                residual_func=ffn_residual_func,
                sample_drop_ratio=sample_drop_ratio,
            )
            return tokens

        if block.training and sample_drop_ratio > 0.0:
            tokens = tokens + block.drop_path1(attn_plus_adapter_residual_func(tokens))
            tokens = tokens + block.drop_path1(ffn_residual_func(tokens))
            return tokens

        tokens = tokens + attn_plus_adapter_residual_func(tokens)
        tokens = tokens + ffn_residual_func(tokens)
        return tokens

    @staticmethod
    def _module_has_trainable_params(module):
        if module is None:
            return False
        return any(param.requires_grad for param in module.parameters())

    def _get_first_graph_block(self, adapter_slots, trainable_block_start):
        first_graph_block = trainable_block_start
        if adapter_slots is None:
            return first_graph_block

        for block_idx, adapter in enumerate(adapter_slots):
            if self._module_has_trainable_params(adapter):
                return min(first_graph_block, block_idx)
        return first_graph_block

    def _forward_tokens_with_block_adapters(
        self,
        tokens,
        spatial_shape,
        adapter_slots: Optional[Sequence[nn.Module]] = None,
        freeze_backbone: bool = False,
    ):
        if adapter_slots is not None and len(adapter_slots) != self.num_blocks:
            raise ValueError(
                f"adapter_slots must match num_blocks={self.num_blocks}, got {len(adapter_slots)}."
            )

        was_dino_training = self.dino.training
        if freeze_backbone:
            self.dino.eval()

        trainable_block_start = self.get_trainable_block_start(force_freeze=freeze_backbone)
        first_graph_block = self._get_first_graph_block(adapter_slots, trainable_block_start)

        try:
            for block_idx, blk in enumerate(self.dino.blocks):
                adapter = adapter_slots[block_idx] if adapter_slots is not None else None
                freeze_block = freeze_backbone or block_idx < trainable_block_start

                if block_idx < first_graph_block:
                    with torch.no_grad():
                        if adapter is None:
                            tokens = blk(tokens)
                        else:
                            tokens = self._forward_block_with_parallel_adapter(
                                blk,
                                tokens,
                                adapter,
                                spatial_shape=spatial_shape,
                            )
                    continue

                block_context = self._temporarily_freeze_block_params(blk) if freeze_block else nullcontext()
                with block_context:
                    if adapter is None:
                        tokens = blk(tokens)
                    else:
                        tokens = self._forward_block_with_parallel_adapter(
                            blk,
                            tokens,
                            adapter,
                            spatial_shape=spatial_shape,
                        )
        finally:
            if freeze_backbone and was_dino_training:
                self.dino.train()

        return tokens

    def _count_params(self, module):
        if module is None:
            return 0, 0
        params = list(module.parameters())
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        return total, trainable

    def format_adapter_status_lines(self):
        lines = [
            f"use_block_adapter={self.use_block_adapter}",
            f"adapter_num_prefix_tokens={self.adapter_num_prefix_tokens}",
            f"adapter_bottleneck_dim={self.adapter_bottleneck_dim}",
            f"adapter_scale={self.adapter_scale}",
            f"adapter_zero_init={self.adapter_zero_init}",
        ]
        if not self.use_block_adapter:
            return lines

        for block_idx in range(self.num_blocks):
            lines.append(f"block[{block_idx}]: adapter=on")
        return lines

    def format_trainable_summary_lines(self):
        dino_total, dino_trainable = self._count_params(self.dino)
        adapter_total, adapter_trainable = self._count_params(self.block_adapters)
        overall_total, overall_trainable = self._count_params(self)
        return [
            f"dino: total={dino_total:,}, trainable={dino_trainable:,}",
            f"block_adapters: total={adapter_total:,}, trainable={adapter_trainable:,}",
            f"overall: total={overall_total:,}, trainable={overall_trainable:,}",
        ]
    
    def forward_features(
        self,
        x,
        freeze_backbone: bool = False,
        adapter_slots: Optional[Sequence[nn.Module]] = None,
    ):
        B, _, H, W = x.shape
        tokens = self.prepare_tokens(x)
        spatial_shape = (H // self.patch_size, W // self.patch_size)
        tokens = self._forward_tokens_with_block_adapters(
            tokens,
            spatial_shape=spatial_shape,
            adapter_slots=adapter_slots,
            freeze_backbone=freeze_backbone,
        )
        return self.tokens_to_feature_map(tokens, batch_size=B, height=H, width=W)

    def forward(self, x, freeze_backbone=False):
        adapter_slots = self.block_adapters if self.use_block_adapter else None
        return self.forward_features(
            x,
            freeze_backbone=freeze_backbone,
            adapter_slots=adapter_slots,
        )
    
    
class ResNet(nn.Module):
    AVAILABLE_MODELS = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
        freeze_layers=None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block
        self.freeze_layers = pretrained if freeze_layers is None else bool(freeze_layers)

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Load the model
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)

        # Create backbone with only the necessary layers
        self.net = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            *([] if crop_last_block else [resnet.layer4]),
        )

        # Handle trainable/frozen layers
        nb_layers = len(self.net)
        assert (
            isinstance(unfreeze_n_blocks, int) and 0 <= unfreeze_n_blocks <= nb_layers
        ), f"unfreeze_n_blocks must be an integer between 0 and {nb_layers} (inclusive)"

        if self.freeze_layers:
            # Freeze required layers
            for layer in self.net[:nb_layers - unfreeze_n_blocks]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks is ignored when freeze_layers=False. Setting it to 0.")
                self.unfreeze_n_blocks = 0

        # Output channels
        if backbone_name in ["resnet18", "resnet34"]:
            self.out_channels = resnet.layer3[-1].conv2.out_channels
        else:
            self.out_channels = resnet.layer3[-1].conv3.out_channels

    def forward(self, x):
        return self.net(x)
