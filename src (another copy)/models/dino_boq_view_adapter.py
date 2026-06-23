from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones import DinoV2

try:
    from src.dinov2_block_adapter import DinoV2BlockAdapter
except ModuleNotFoundError:
    from dinov2_block_adapter import DinoV2BlockAdapter


class DinoBoQViewSpecificAdapterEncoder(nn.Module):
    """Shared DINOv2 backbone + dual view-specific block adapters + shared BoQ encoder."""

    _LEGACY_SPATIAL_ADAPTER_SUFFIXES = (
        "norm.weight",
        "norm.bias",
        "depthwise_conv.weight",
        "depthwise_conv.bias",
    )

    def __init__(
        self,
        backbone: DinoV2,
        aggregator: nn.Module,
        normalize_descriptor: bool = True,
        use_view_specific_adapter: bool = True,
        adapter_type: str = "spatial",
        adapter_layers: Optional[Sequence[int]] = None,
        adapter_bottleneck_dim: int = 128,
        adapter_scale: float = 0.1,
        adapter_zero_init: bool = True,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: Optional[int] = None,
        freeze_aggregation: bool = False,
        train_ground_adapter: bool = True,
        train_sat_adapter: bool = True,
    ):
        super().__init__()
        if not isinstance(backbone, DinoV2):
            raise TypeError(
                "DinoBoQViewSpecificAdapterEncoder expects a `src.backbones.DinoV2` backbone "
                f"for block-wise adapter insertion, got {type(backbone).__name__}."
            )

        self.backbone = backbone
        self.aggregator = aggregator
        self.normalize_descriptor = bool(normalize_descriptor)
        self.use_view_specific_adapter = bool(use_view_specific_adapter)
        # Kept for CLI/checkpoint compatibility. The implementation now always
        # uses DinoV2BlockAdapter-style residual branches internally.
        self.adapter_type = adapter_type
        self.adapter_bottleneck_dim = int(adapter_bottleneck_dim)
        self.adapter_scale = float(adapter_scale)
        self.adapter_zero_init = bool(adapter_zero_init)
        self.train_ground_adapter = bool(train_ground_adapter)
        self.train_sat_adapter = bool(train_sat_adapter)
        self.num_backbone_blocks = self.backbone.num_blocks

        self.adapter_layers = tuple(self._resolve_adapter_layers(adapter_layers))
        self._adapter_layer_set = set(self.adapter_layers)

        self.ground_adapters = self._build_adapter_slots(view_name="ground")
        self.satellite_adapters = self._build_adapter_slots(view_name="satellite")

        self.configure_trainable_modules(
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            freeze_aggregation=freeze_aggregation,
            train_ground_adapter=train_ground_adapter,
            train_sat_adapter=train_sat_adapter,
        )

    @property
    def output_dim(self) -> int:
        return self.aggregator.proj_c.out_channels * self.aggregator.fc.out_features

    @property
    def shared_backbone(self) -> DinoV2:
        return self.backbone

    @property
    def shared_aggregator(self) -> nn.Module:
        return self.aggregator

    def _resolve_adapter_layers(self, adapter_layers: Optional[Sequence[int]]) -> List[int]:
        if not self.use_view_specific_adapter:
            return []

        if adapter_layers is None:
            adapter_layers = list(range(self.num_backbone_blocks))

        resolved_layers = sorted({int(layer_idx) for layer_idx in adapter_layers})
        invalid_layers = [layer_idx for layer_idx in resolved_layers if layer_idx < 0 or layer_idx >= self.num_backbone_blocks]
        if invalid_layers:
            raise ValueError(
                f"adapter_layers must be in [0, {self.num_backbone_blocks - 1}], got {invalid_layers}."
            )
        return resolved_layers

    def _make_adapter(self) -> DinoV2BlockAdapter:
        return DinoV2BlockAdapter(
            dim=self.backbone.out_channels,
            bottleneck_dim=self.adapter_bottleneck_dim,
            scale=self.adapter_scale,
            zero_init=self.adapter_zero_init,
            num_prefix_tokens=self.backbone.num_prefix_tokens,
        )

    def _build_adapter_slots(self, view_name: str) -> List[Optional[DinoV2BlockAdapter]]:
        slots: List[Optional[DinoV2BlockAdapter]] = []
        for layer_idx in range(self.num_backbone_blocks):
            adapter = self._make_adapter() if layer_idx in self._adapter_layer_set else None
            slots.append(adapter)
            if adapter is not None:
                # Preserve legacy module names for checkpoint compatibility.
                self.add_module(f"{view_name}_adapter_{layer_idx}", adapter)
        return slots

    @staticmethod
    def _set_module_trainable(module: Optional[nn.Module], trainable: bool):
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = bool(trainable)

    def _iter_real_adapters(self, adapter_slots: Sequence[Optional[nn.Module]]) -> Iterable[nn.Module]:
        for adapter in adapter_slots:
            if adapter is not None:
                yield adapter

    def _set_adapter_slots_trainable(self, adapter_slots: Sequence[Optional[nn.Module]], trainable: bool):
        for adapter in self._iter_real_adapters(adapter_slots):
            self._set_module_trainable(adapter, trainable)

    def configure_trainable_modules(
        self,
        freeze_backbone: bool,
        unfreeze_last_n_blocks: Optional[int] = None,
        freeze_aggregation: bool = False,
        train_ground_adapter: bool = True,
        train_sat_adapter: bool = True,
    ):
        self.train_ground_adapter = bool(train_ground_adapter)
        self.train_sat_adapter = bool(train_sat_adapter)

        self.backbone.set_trainable_layers(
            freeze_backbone=freeze_backbone,
            unfreeze_n_blocks=unfreeze_last_n_blocks,
        )
        self._set_module_trainable(self.aggregator, not freeze_aggregation)
        self._set_adapter_slots_trainable(self.ground_adapters, self.train_ground_adapter)
        self._set_adapter_slots_trainable(self.satellite_adapters, self.train_sat_adapter)

    def _count_params(self, modules) -> Tuple[int, int]:
        if modules is None:
            return 0, 0

        if isinstance(modules, (list, tuple)):
            parameters = []
            seen_param_ids = set()
            for module in modules:
                if module is None:
                    continue
                for param in module.parameters():
                    if id(param) in seen_param_ids:
                        continue
                    seen_param_ids.add(id(param))
                    parameters.append(param)
        else:
            parameters = list(modules.parameters())

        total = sum(param.numel() for param in parameters)
        trainable = sum(param.numel() for param in parameters if param.requires_grad)
        return total, trainable

    @staticmethod
    def _format_param_count_line(
        name: str,
        total: int,
        trainable: int,
        module_type: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> str:
        parts = [f"{name}: {module_type}" if module_type else name]
        parts.append(f"total={total:,}, trainable={trainable:,}")
        if extra:
            parts.append(extra)
        return " | ".join(parts)

    @staticmethod
    def _module_has_trainable_params(module: Optional[nn.Module]) -> bool:
        if module is None:
            return False
        return any(param.requires_grad for param in module.parameters())

    def _format_block_adapter_status(self, adapter: Optional[nn.Module]) -> str:
        if adapter is None:
            return "off"
        return "on(trainable)" if self._module_has_trainable_params(adapter) else "on(frozen)"

    def get_trainable_param_summary(self) -> Dict[str, Dict[str, int]]:
        backbone_total, backbone_trainable = self._count_params(self.backbone)
        aggregation_total, aggregation_trainable = self._count_params(self.aggregator)
        ground_total, ground_trainable = self._count_params(self.ground_adapters)
        sat_total, sat_trainable = self._count_params(self.satellite_adapters)
        overall_total, overall_trainable = self._count_params(self)
        return {
            "backbone": {"total": backbone_total, "trainable": backbone_trainable},
            "aggregation": {"total": aggregation_total, "trainable": aggregation_trainable},
            "ground_adapters": {"total": ground_total, "trainable": ground_trainable},
            "satellite_adapters": {"total": sat_total, "trainable": sat_trainable},
            "overall": {"total": overall_total, "trainable": overall_trainable},
        }

    def format_adapter_status_lines(self) -> List[str]:
        lines = [
            f"use_view_specific_adapter={self.use_view_specific_adapter}",
            f"adapter_type={self.adapter_type} (compatibility_only)",
            "adapter_impl=dual_dinov2_block_adapter",
            f"adapter_layers={list(self.adapter_layers)}",
            f"adapter_bottleneck_dim={self.adapter_bottleneck_dim}",
            f"adapter_scale={self.adapter_scale}",
            f"adapter_zero_init={self.adapter_zero_init}",
        ]
        for layer_idx in range(self.num_backbone_blocks):
            lines.append(
                "block[{idx}]: ground={ground}, satellite={satellite}".format(
                    idx=layer_idx,
                    ground=self._format_block_adapter_status(self.ground_adapters[layer_idx]),
                    satellite=self._format_block_adapter_status(self.satellite_adapters[layer_idx]),
                )
            )
        return lines

    def format_trainable_summary_lines(self) -> List[str]:
        summary = self.get_trainable_param_summary()
        return [
            (
                f"{module_name}: total={values['total']:,}, "
                f"trainable={values['trainable']:,}"
            )
            for module_name, values in summary.items()
        ]

    def format_module_summary_lines(self) -> List[str]:
        encoder_total, encoder_trainable = self._count_params(self)
        backbone_total, backbone_trainable = self._count_params(self.backbone)
        dino_total, dino_trainable = self._count_params(self.backbone.dino)
        aggregator_total, aggregator_trainable = self._count_params(self.aggregator)
        ground_total, ground_trainable = self._count_params(self.ground_adapters)
        sat_total, sat_trainable = self._count_params(self.satellite_adapters)

        lines = [
            self._format_param_count_line(
                "image_encoder",
                encoder_total,
                encoder_trainable,
                module_type=self.__class__.__name__,
                extra=f"normalize_descriptor={self.normalize_descriptor}",
            ),
            self._format_param_count_line(
                "backbone",
                backbone_total,
                backbone_trainable,
                module_type=self.backbone.__class__.__name__,
                extra=(
                    f"blocks={self.num_backbone_blocks}, embed_dim={self.backbone.out_channels}, "
                    f"patch_size={self.backbone.patch_size}, prefix_tokens={self.backbone.num_prefix_tokens}"
                ),
            ),
            self._format_param_count_line(
                "backbone.dino",
                dino_total,
                dino_trainable,
                module_type=self.backbone.dino.__class__.__name__,
            ),
            self._format_param_count_line(
                "aggregator",
                aggregator_total,
                aggregator_trainable,
                module_type=self.aggregator.__class__.__name__,
            ),
        ]

        for child_name, child_module in self.aggregator.named_children():
            child_total, child_trainable = self._count_params(child_module)
            lines.append(
                self._format_param_count_line(
                    f"aggregator.{child_name}",
                    child_total,
                    child_trainable,
                    module_type=child_module.__class__.__name__,
                )
            )

        lines.append(
            self._format_param_count_line(
                "ground_adapters",
                ground_total,
                ground_trainable,
                extra=f"enabled={len(self.adapter_layers)}/{self.num_backbone_blocks}",
            )
        )
        for layer_idx in self.adapter_layers:
            adapter = self.ground_adapters[layer_idx]
            adapter_total, adapter_trainable = self._count_params(adapter)
            lines.append(
                self._format_param_count_line(
                    f"ground_adapter_{layer_idx}",
                    adapter_total,
                    adapter_trainable,
                    module_type=adapter.__class__.__name__,
                )
            )

        lines.append(
            self._format_param_count_line(
                "satellite_adapters",
                sat_total,
                sat_trainable,
                extra=f"enabled={len(self.adapter_layers)}/{self.num_backbone_blocks}",
            )
        )
        for layer_idx in self.adapter_layers:
            adapter = self.satellite_adapters[layer_idx]
            adapter_total, adapter_trainable = self._count_params(adapter)
            lines.append(
                self._format_param_count_line(
                    f"satellite_adapter_{layer_idx}",
                    adapter_total,
                    adapter_trainable,
                    module_type=adapter.__class__.__name__,
                )
            )

        return lines

    def get_adapter_state_keys(self, view_type: str) -> List[str]:
        adapter_slots = self._get_adapter_slots(view_type)
        state_keys = []
        for layer_idx, adapter in enumerate(adapter_slots):
            if adapter is None:
                continue
            for key in adapter.state_dict().keys():
                state_keys.append(f"{view_type}_adapter_{layer_idx}.{key}")
        return state_keys

    def _get_adapter_slots(self, view_type: str) -> List[Optional[DinoV2BlockAdapter]]:
        if view_type == "ground":
            return self.ground_adapters
        if view_type == "satellite":
            return self.satellite_adapters
        raise ValueError(f"Unsupported view_type={view_type!r}. Expected 'ground' or 'satellite'.")

    def _forward_backbone_with_adapters(
        self,
        images: torch.Tensor,
        adapter_slots: Sequence[Optional[DinoV2BlockAdapter]],
        freeze_backbone: bool = False,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape `[B, C, H, W]`, got {tuple(images.shape)}.")

        return self.backbone.forward_features(
            images,
            freeze_backbone=freeze_backbone,
            adapter_slots=adapter_slots,
        )

    def encode_view(
        self,
        images: torch.Tensor,
        view_type: str,
        freeze_backbone: bool = False,
        return_debug: bool = False,
    ):
        adapter_slots = self._get_adapter_slots(view_type)
        features = self._forward_backbone_with_adapters(
            images,
            adapter_slots=adapter_slots,
            freeze_backbone=freeze_backbone,
        )
        descriptors, _ = self.aggregator(features)
        if self.normalize_descriptor:
            descriptors = F.normalize(descriptors, p=2, dim=-1)

        if return_debug:
            debug_shapes = {
                "backbone_out": tuple(features.shape),
                "after_adapter": tuple(features.shape),
                "descriptor": tuple(descriptors.shape),
            }
            return descriptors, debug_shapes
        return descriptors


    

    def encode_view_with_patch_map(
        self,
        images: torch.Tensor,
        view_type: str,
        freeze_backbone: bool = False,
    ):
        """
        Evaluation-only encoding.

        Returns:
            embedding:
                Global BoQ descriptor, shape [B, D].
            patch_map:
                Spatial feature map before global BoQ aggregation,
                shape [B, C, Hf, Wf].
        """
        adapter_slots = self._get_adapter_slots(view_type)

        # Shared DINOv2 + view-specific adapters.
        features = self._forward_backbone_with_adapters(
            images,
            adapter_slots=adapter_slots,
            freeze_backbone=freeze_backbone,
        )

        # Original global BoQ descriptor.
        embedding, _ = self.aggregator(features)
        if self.normalize_descriptor:
            embedding = F.normalize(embedding, p=2, dim=-1)

        # Use the trained BoQ channel projection as the local feature space.
        # For the current configuration:
        # [B, 768, Hf, Wf] -> [B, 384, Hf, Wf].
        patch_map = self.aggregator.proj_c(features)
        patch_map = F.normalize(patch_map, p=2, dim=1)

        return {
            "embedding": embedding,
            "patch_map": patch_map,
        }




    def encode_ground(self, images: torch.Tensor, freeze_backbone: bool = False, return_debug: bool = False):
        return self.encode_view(
            images,
            view_type="ground",
            freeze_backbone=freeze_backbone,
            return_debug=return_debug,
        )

    def encode_satellite(self, images: torch.Tensor, freeze_backbone: bool = False, return_debug: bool = False):
        return self.encode_view(
            images,
            view_type="satellite",
            freeze_backbone=freeze_backbone,
            return_debug=return_debug,
        )

    def forward(
        self,
        images: torch.Tensor,
        view_type: str = "ground",
        freeze_backbone: bool = False,
        return_debug: bool = False,
    ):
        return self.encode_view(
            images,
            view_type=view_type,
            freeze_backbone=freeze_backbone,
            return_debug=return_debug,
        )

    @classmethod
    def _is_legacy_spatial_adapter_key(cls, local_key: str) -> bool:
        if not (local_key.startswith("ground_adapter_") or local_key.startswith("satellite_adapter_")):
            return False
        return any(local_key.endswith(suffix) for suffix in cls._LEGACY_SPATIAL_ADAPTER_SUFFIXES)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Older checkpoints used ViewSpecificSpatialAdapter and therefore may
        # still contain conv/norm weights under the same adapter prefixes. The
        # shared down/up projection keys remain compatible and are left intact.
        legacy_keys = []
        for key in list(state_dict.keys()):
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            if self._is_legacy_spatial_adapter_key(local_key):
                legacy_keys.append(key)

        for key in legacy_keys:
            state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )





