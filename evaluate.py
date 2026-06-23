from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from train import (
    HyperParams,
    build_aggregator,
    build_backbone,
    build_backbone_from_name,
)
from src.dataloaders.datasets_ws_kitti360 import (
    KITTI360BaseDataset,
    configure_kitti360_options,

)

from src.dataloaders.datasets_ws_nuscenes import (
    NuScenesBaseDataset,
    configure_nuscenes_options,
)
from src.evaluate import test as evaluate_kitti360
from src.models.dino_boq_view_adapter import DinoBoQViewSpecificAdapterEncoder
from src.models.kitti360_boq_triplet import Kitti360BoQTripletModel


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot read hparams file {path}: PyYAML is not installed. "
            "Either install PyYAML or pass the model architecture options explicitly."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=yaml.FullLoader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in hparams file {path}, got {type(data).__name__}.")
    return data


def _infer_hparams_path(checkpoint_path: Path) -> Path | None:
    checkpoint_path = checkpoint_path.resolve()
    for parent in checkpoint_path.parents:
        candidate = parent / "hparams.yaml"
        if candidate.exists():
            return candidate
    return None


def _apply_mapping(hparams: HyperParams, values: dict):
    for key, value in values.items():
        if hasattr(hparams, key):
            setattr(hparams, key, value)


def _apply_cli_overrides(hparams: HyperParams, args: argparse.Namespace):
    overrides = [
        # ============================================================
        # basic / model
        # ============================================================
        ("aggregator", "aggregator_type"),
        ("device", "device"),
        ("infer_batch_size", "kitti360_infer_batch_size"),
        ("num_workers", "num_workers"),

        # ============================================================
        # dataset switch
        # ============================================================
        ("dataset_name", "dataset_name"),

        # ============================================================
        # KITTI360
        # ============================================================
        ("kitti360_path", "kitti360_path"),
        ("kitti360_camnames", "kitti360_camnames"),
        ("kitti360_maptype", "kitti360_maptype"),
        ("kitti360_db_crop", "kitti360_db_crop_size"),
        ("kitti360_val_pos_th", "kitti360_val_positive_dist_threshold"),

        # ============================================================
        # nuScenes
        # ============================================================
        ("nuscenes_path", "nuscenes_path"),
        ("nuscenes_train_version", "nuscenes_train_version"),
        ("nuscenes_val_version", "nuscenes_val_version"),
        ("nuscenes_locations", "nuscenes_locations"),
        ("nuscenes_camnames", "nuscenes_camnames"),
        ("nuscenes_maptype", "nuscenes_maptype"),
        ("nuscenes_aerial_scale", "nuscenes_aerial_scale"),
        ("nuscenes_aerial_zoom", "nuscenes_aerial_zoom"),
        ("nuscenes_aerial_size", "nuscenes_aerial_size"),
        ("nuscenes_db_crop", "nuscenes_db_crop_size"),
        ("nuscenes_val_pos_th", "nuscenes_val_positive_dist_threshold"),
    ]

    for src, dst in overrides:
        value = getattr(args, src, None)
        if value is not None:
            setattr(hparams, dst, value)

    # ============================================================
    # image size overrides
    # ============================================================
    if getattr(args, "query_img_size", None) is not None:
        hparams.query_img_size = tuple(args.query_img_size)

    if getattr(args, "db_img_size", None) is not None:
        hparams.db_img_size = tuple(args.db_img_size)

    if getattr(args, "nuscenes_q_size", None) is not None:
        hparams.nuscenes_query_img_size = tuple(args.nuscenes_q_size)

    if getattr(args, "nuscenes_db_size", None) is not None:
        hparams.nuscenes_db_img_size = tuple(args.nuscenes_db_size)

    # ============================================================
    # infer batch size 同步
    # ============================================================
    if getattr(args, "infer_batch_size", None) is not None:
        hparams.kitti360_infer_batch_size = args.infer_batch_size
        hparams.nuscenes_infer_batch_size = args.infer_batch_size

def _load_hparams(args: argparse.Namespace) -> tuple[HyperParams, Path | None]:
    hparams = HyperParams()

    checkpoint_path = Path(args.checkpoint)
    hparams_path = (
        Path(args.hparams)
        if args.hparams
        else _infer_hparams_path(checkpoint_path)
    )

    if hparams_path is not None and hparams_path.exists():
        _apply_mapping(hparams, _load_yaml(hparams_path))

    # 这里只调用一次
    _apply_cli_overrides(hparams, args)

    dataset_name = str(
        getattr(hparams, "dataset_name", getattr(args, "dataset_name", "kitti360"))
    ).lower()

    hparams.dataset_name = dataset_name
    hparams.silent = True
    hparams.compile = False
    hparams.use_pretrained_boq = True

    if dataset_name == "kitti360":
        hparams.use_kitti360_boq = True
        hparams.use_nuscenes_boq = False

        if not getattr(hparams, "kitti360_path", None):
            raise ValueError(
                "Please pass --kitti360_path when dataset_name='kitti360'."
            )

    elif dataset_name == "nuscenes":
        hparams.use_kitti360_boq = False
        hparams.use_nuscenes_boq = True

        if not getattr(hparams, "nuscenes_path", None):
            raise ValueError(
                "Please pass --nuscenes_path when dataset_name='nuscenes'."
            )

        if not hasattr(hparams, "nuscenes_infer_batch_size"):
            hparams.nuscenes_infer_batch_size = getattr(
                hparams,
                "kitti360_infer_batch_size",
                32,
            )

    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name}")

    return hparams, hparams_path


def _resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    if args.gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError("--gpu_id was set, but CUDA is not available.")
        gpu_id = int(args.gpu_id)
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"--gpu_id must be in [0, {torch.cuda.device_count() - 1}], got {gpu_id}.")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_model(hparams: HyperParams) -> tuple[Kitti360BoQTripletModel, tuple[int, int], tuple[int, int]]:
    if hparams.shared_query_boq:
        hparams.dual_branch = True

    use_view_specific_adapter = bool(hparams.use_view_specific_adapter)
    use_dual_branch = bool(hparams.dual_branch or hparams.shared_query_boq)
    if use_view_specific_adapter and use_dual_branch:
        raise ValueError("Choose only one of view_specific_adapter, dual_branch, or shared_query_boq.")
    if use_view_specific_adapter and hparams.use_dinov2_block_adapter:
        raise ValueError("view_specific_adapter cannot be combined with dinov2_block_adapter.")
    if use_dual_branch and hparams.use_dinov2_block_adapter:
        raise ValueError("dinov2_block_adapter is supported only for the standard single-backbone flow.")

    if use_view_specific_adapter:
        if "dinov2" not in hparams.backbone_name:
            raise ValueError("The view-specific adapter KITTI360 model supports DINOv2 backbones only.")
        backbone, train_img_size, _ = build_backbone(hparams)
        if hparams.adapter_layers is None:
            hparams.adapter_layers = list(range(backbone.num_blocks))
        aggregator = build_aggregator(hparams, backbone=backbone)
        image_encoder = DinoBoQViewSpecificAdapterEncoder(
            backbone=backbone,
            aggregator=aggregator,
            normalize_descriptor=hparams.normalize_descriptor,
            use_view_specific_adapter=hparams.use_view_specific_adapter,
            adapter_type=hparams.adapter_type,
            adapter_layers=hparams.adapter_layers,
            adapter_bottleneck_dim=hparams.adapter_bottleneck_dim,
            adapter_scale=hparams.adapter_scale,
            adapter_zero_init=hparams.adapter_zero_init,
            freeze_backbone=hparams.freeze_backbone,
            unfreeze_last_n_blocks=hparams.unfreeze_n_blocks,
            freeze_aggregation=hparams.freeze_aggregator,
            train_ground_adapter=hparams.train_ground_adapter,
            train_sat_adapter=hparams.train_sat_adapter,
        )
        query_img_size = hparams.kitti360_query_img_size or train_img_size
        db_img_size = hparams.kitti360_db_img_size or train_img_size
    elif use_dual_branch:
        query_backbone_name = hparams.query_backbone or hparams.backbone_name
        db_backbone_name = hparams.db_backbone or hparams.backbone_name
        query_backbone, query_train_img_size, _, resolved_query_backbone = build_backbone_from_name(
            query_backbone_name,
            hparams.unfreeze_n_blocks,
            pretrained=False,
        )
        db_backbone, db_train_img_size, _, resolved_db_backbone = build_backbone_from_name(
            db_backbone_name,
            hparams.unfreeze_n_blocks,
            pretrained=False,
        )
        hparams.query_backbone = resolved_query_backbone
        hparams.db_backbone = resolved_db_backbone

        if hparams.adapter_out_channels is None:
            hparams.adapter_out_channels = min(query_backbone.out_channels, db_backbone.out_channels)
        
        
    else:
        backbone, train_img_size, _ = build_backbone(hparams)
        aggregator = build_aggregator(hparams, backbone=backbone)
        image_encoder = None
        query_img_size = hparams.kitti360_query_img_size or train_img_size
        db_img_size = hparams.kitti360_db_img_size or train_img_size

    model_kwargs = dict(
        lr=hparams.lr,
        lr_mul=hparams.lr_mul,
        weight_decay=hparams.weight_decay,
        warmup_epochs=hparams.warmup_epochs,
        backbone_warmup_epochs=hparams.backbone_warmup_epochs,
        milestones=hparams.milestones,
        satellite_map_index=hparams.satellite_map_index,
        db_encode_chunk_size=hparams.db_encode_chunk_size,
        triplet_margin=hparams.triplet_margin,
        normalize_descriptor=hparams.normalize_descriptor,
        loss_type=hparams.loss_type,
        freeze_backbone=hparams.freeze_backbone,
        freeze_aggregator=hparams.freeze_aggregator,
        silent=True,
        debug_shapes=False,
        print_train_items=0,
        print_train_items_every_epoch=False,
    )
    
    if image_encoder is not None:
        model = Kitti360BoQTripletModel(image_encoder=image_encoder, **model_kwargs)
    else:
        model = Kitti360BoQTripletModel(backbone, aggregator, **model_kwargs)

    return model, query_img_size, db_img_size


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, allow_partial: bool = False):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=not allow_partial)
    return incompatible


# def _build_test_dataset(hparams: HyperParams, query_img_size: tuple[int, int], db_img_size: tuple[int, int]):
#     configure_kitti360_options(
#         dataroot=hparams.kitti360_path,
#         train_ratio=hparams.kitti360_train_ratio,
#         share_db=False,
#         q_resize=query_img_size,
#         q_jitter=0.0,
#         db_cropsize=hparams.kitti360_db_crop_size,
#         db_resize=db_img_size,
#         db_jitter=0.0,
#         camnames=hparams.kitti360_camnames,
#         maptype=hparams.kitti360_maptype,
#         traindownsample=1,
#         num_workers=hparams.num_workers,
#         val_positive_dist_threshold=hparams.kitti360_val_positive_dist_threshold,
#     )
#     dataset_args = SimpleNamespace(
#         resize=query_img_size,
#         test_method="single_query",
#         num_workers=hparams.num_workers,
#         infer_batch_size=hparams.kitti360_infer_batch_size,
#         device="cpu",
#     )
#     return KITTI360BaseDataset(args=dataset_args, dataset_name="kitti360", split="test"), dataset_args


def _build_test_dataset(
    hparams: HyperParams,
    query_img_size: tuple[int, int],
    db_img_size: tuple[int, int],
):
    dataset_name = str(
        getattr(hparams, "dataset_name", "kitti360")
    ).lower()

    if dataset_name == "kitti360":
        configure_kitti360_options(
            dataroot=hparams.kitti360_path,
            train_ratio=hparams.kitti360_train_ratio,
            share_db=False,
            q_resize=query_img_size,
            q_jitter=0.0,
            db_cropsize=hparams.kitti360_db_crop_size,
            db_resize=db_img_size,
            db_jitter=0.0,
            camnames=hparams.kitti360_camnames,
            maptype=hparams.kitti360_maptype,
            traindownsample=1,
            num_workers=hparams.num_workers,
            val_positive_dist_threshold=hparams.kitti360_val_positive_dist_threshold,
        )

        dataset_args = SimpleNamespace(
            dataset_name="kitti360",
            resize=query_img_size,
            test_method="single_query",
            num_workers=hparams.num_workers,
            infer_batch_size=hparams.kitti360_infer_batch_size,
            device="cpu",
        )

        dataset = KITTI360BaseDataset(
            args=dataset_args,
            dataset_name="kitti360",
            split="test",
        )

        return dataset, dataset_args

    if dataset_name == "nuscenes":
        q_size = getattr(
            hparams,
            "nuscenes_query_img_size",
            None,
        ) or query_img_size

        db_size = getattr(
            hparams,
            "nuscenes_db_img_size",
            None,
        ) or db_img_size

        configure_nuscenes_options(
            dataroot=hparams.nuscenes_path,
            train_version=getattr(
                hparams,
                "nuscenes_train_version",
                "v1.0-trainval",
            ),
            val_version=getattr(
                hparams,
                "nuscenes_val_version",
                "v1.0-test",
            ),
            train_ratio=getattr(
                hparams,
                "nuscenes_train_ratio",
                1.0,
            ),
            share_db=False,
            q_resize=q_size,
            q_jitter=0.0,
            db_cropsize=getattr(
                hparams,
                "nuscenes_db_crop_size",
                320,
            ),
            db_resize=db_size,
            db_jitter=0.0,
            camnames=getattr(
                hparams,
                "nuscenes_camnames",
                "CAM_FRONT",
            ),
            maptype=getattr(
                hparams,
                "nuscenes_maptype",
                "satellite",
            ),
            locations=getattr(
                hparams,
                "nuscenes_locations",
                (
                    "boston-seaport,"
                    "singapore-hollandvillage,"
                    "singapore-onenorth,"
                    "singapore-queenstown"
                ),
            ),
            aerial_scale=getattr(
                hparams,
                "nuscenes_aerial_scale",
                1,
            ),
            aerial_zoom=getattr(
                hparams,
                "nuscenes_aerial_zoom",
                20,
            ),
            aerial_size=getattr(
                hparams,
                "nuscenes_aerial_size",
                320,
            ),
            use_keyframes_only=getattr(
                hparams,
                "nuscenes_use_keyframes_only",
                True,
            ),
            traindownsample=1,
            num_workers=hparams.num_workers,
            val_positive_dist_threshold=getattr(
                hparams,
                "nuscenes_val_positive_dist_threshold",
                25.0,
            ),
        )

        dataset_args = SimpleNamespace(
            dataset_name="nuscenes",
            resize=q_size,
            test_method="single_query",
            num_workers=hparams.num_workers,
            infer_batch_size=getattr(
                hparams,
                "nuscenes_infer_batch_size",
                getattr(hparams, "kitti360_infer_batch_size", 32),
            ),
            device="cpu",
        )

        dataset = NuScenesBaseDataset(
            args=dataset_args,
            dataset_name="nuscenes",
            split="test",
        )

        return dataset, dataset_args

    raise ValueError(
        f"Unsupported dataset_name={dataset_name!r}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone KITTI360 evaluation for BoQ checkpoints. "
        "The test split is the last 15% of each video sequence by default."
    )
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to a .ckpt file.")
    parser.add_argument("--hparams", type=str, help="Path to hparams.yaml. Defaults to searching near the checkpoint.")
    parser.add_argument("--kitti360_path", type=str, help="KITTI360 root path.")
    parser.add_argument("--test_tail_ratio", type=float, default=0.15, help="Tail ratio of each sequence used for test.")
    parser.add_argument("--device", type=str, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--gpu_id", type=int, help="GPU id to use when --device is not set.")
    parser.add_argument("--num_workers", "--nw", type=int, help="Number of dataloader workers.")
    parser.add_argument("--infer_batch_size", "--bs", type=int, help="Evaluation batch size.")
    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys instead of failing.",
    )

    parser.add_argument("--backbone", type=str)
    parser.add_argument("--query_backbone", type=str)
    parser.add_argument("--db_backbone", type=str)
    parser.add_argument("--dual_branch", action="store_true")
    parser.add_argument("--shared_query_boq", action="store_true")
    parser.add_argument("--view_specific_adapter", action="store_true")
    parser.add_argument("--dinov2_block_adapter", action="store_true")
    parser.add_argument("--adapter_out_channels", type=int)
    parser.add_argument("--adapter_type", type=str)
    parser.add_argument("--adapter_layers", type=int, nargs="+")
    parser.add_argument("--adapter_bottleneck_dim", type=int)
    parser.add_argument("--adapter_scale", type=float)
    parser.add_argument("--unfreeze_n", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--num_queries", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--channel_proj", type=int)
    parser.add_argument("--db_encode_chunk_size", type=int)
    parser.add_argument("--satellite_map_index", type=int)
    parser.add_argument("--no_normalize_descriptor", action="store_true")

    parser.add_argument("--kitti360_camnames", type=str)
    parser.add_argument("--kitti360_maptype", type=str)
    parser.add_argument("--kitti360_q_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--kitti360_db_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--kitti360_db_crop", type=int)
    parser.add_argument("--kitti360_val_pos_th", type=float)
    parser.add_argument(
        "--use_patch_rerank",
        action="store_true",
        help="Use multi-scale patch matching to rerank global BoQ candidates.",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        choices=["boq", "vlaq"],
        help="Aggregator type used by the checkpoint.",
    )
    parser.add_argument(
        "--patch_sizes",
        type=int,
        nargs="+",
        default=[2,5, 8],
        help="Patch sizes on the spatial feature map.",
    )

    parser.add_argument(
        "--patch_strides",
        type=int,
        nargs="+",
        default=[1, 1, 1],
        help="Sliding strides corresponding to patch_sizes.",
    )

    parser.add_argument(
        "--patch_weights",
        type=float,
        nargs="+",
        default=[0.45, 0.15, 0.40],
        help="Fusion weights for patch scores at each scale.",
    )

    parser.add_argument(
        "--patch_rerank_topk",
        type=int,
        default=100,
        help="Number of global BoQ candidates reranked by patch matching.",
    )

    parser.add_argument(
        "--patch_global_weight",
        type=float,
        default=0.4,
        help="Weight of global BoQ similarity in the final score.",
    )

    parser.add_argument(
        "--patch_spatial_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of patch spatial consistency. "
            "For ground-to-satellite matching, start from 0.0."
        ),
    )

    parser.add_argument(
        "--patch_spatial_tau",
        type=float,
        default=0.15,
        help="Tolerance used by the optional rapid spatial score.",
    )

    parser.add_argument(
        "--patch_score_batch_size",
        type=int,
        default=4,
        help="Number of queries processed together during patch reranking.",)

  
    
    
    parser.add_argument(
    "--dataset_name",
    type=str,
    choices=["kitti360", "nuscenes"],
    default="kitti360",
    help="Dataset used for evaluation.",
)

    # ============================================================
    # nuScenes evaluation args
    # ============================================================
    parser.add_argument(
        "--nuscenes_path",
        type=str,
        help="nuScenes root path, e.g. /mnt/sda/ZhengyiXu/datasets/radar/nuscenes",
    )
    parser.add_argument(
        "--nuscenes_train_version",
        type=str,
        default="v1.0-trainval",
    )
    parser.add_argument(
        "--nuscenes_val_version",
        type=str,
        default="v1.0-test",
    )
    parser.add_argument(
        "--nuscenes_locations",
        type=str,
        default=(
            "boston-seaport,"
            "singapore-hollandvillage,"
            "singapore-onenorth,"
            "singapore-queenstown"
        ),
    )
    parser.add_argument(
        "--nuscenes_camnames",
        type=str,
        default="CAM_FRONT",
    )
    parser.add_argument(
        "--nuscenes_maptype",
        type=str,
        default="satellite",
    )
    parser.add_argument(
        "--nuscenes_aerial_scale",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nuscenes_aerial_zoom",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--nuscenes_aerial_size",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--nuscenes_q_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--nuscenes_db_size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--nuscenes_db_crop",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--nuscenes_val_pos_th",
        type=float,
        default=25.0,
    )




    return parser.parse_args()


# def main():
#     args = parse_args()
#     if not (
#         len(args.patch_sizes)
#         == len(args.patch_strides)
#         == len(args.patch_weights)
#     ):
#         raise ValueError(
#             "--patch_sizes、--patch_strides 和 --patch_weights "
#             "必须包含相同数量的元素。"
#         )

#     if any(size <= 0 for size in args.patch_sizes):
#         raise ValueError("--patch_sizes 必须全部大于 0。")

#     if any(stride <= 0 for stride in args.patch_strides):
#         raise ValueError("--patch_strides 必须全部大于 0。")

#     if any(weight < 0 for weight in args.patch_weights):
#         raise ValueError("--patch_weights 不能包含负数。")

#     weight_sum = sum(args.patch_weights)
#     if weight_sum <= 0:
#         raise ValueError("--patch_weights 的总和必须大于 0。")

#     args.patch_weights = [
#         weight / weight_sum
#         for weight in args.patch_weights
#     ]

    
#     checkpoint_path = Path(args.checkpoint)
#     hparams, hparams_path = _load_hparams(args)
#     device = _resolve_device(args)
   
#     test_dataset, eval_args = _build_test_dataset(hparams, query_img_size, db_img_size)
#     eval_args.device = str(device)
    
#     eval_args.use_patch_rerank = args.use_patch_rerank

#     eval_args.patch_sizes = tuple(args.patch_sizes)
#     eval_args.patch_strides = tuple(args.patch_strides)
#     eval_args.patch_weights = tuple(args.patch_weights)

#     eval_args.patch_rerank_topk = args.patch_rerank_topk
#     eval_args.patch_global_weight = args.patch_global_weight

#     eval_args.patch_spatial_weight = args.patch_spatial_weight
#     eval_args.patch_spatial_tau = args.patch_spatial_tau

#     eval_args.patch_score_batch_size = args.patch_score_batch_size


#     print(f"checkpoint: {checkpoint_path}")
#     if hparams_path is not None:
#         print(f"hparams: {hparams_path}")
#     print(f"device: {device}")
#     print(
#         "KITTI360 test split: "
#         f"last {args.test_tail_ratio * 100:.1f}% of each selected video sequence "
#         f"(train_ratio={hparams.kitti360_train_ratio:.4f}, share_db=False)"
#     )
#     print(f"queries: {test_dataset.queries_num}, database: {test_dataset.database_num}")
#     if incompatible.missing_keys:
#         print(f"missing checkpoint keys: {len(incompatible.missing_keys)}")
#     if incompatible.unexpected_keys:
#         print(f"unexpected checkpoint keys: {len(incompatible.unexpected_keys)}")

#     _, recall_str, metrics = evaluate_kitti360(
#         args=eval_args,
#         test_ds=test_dataset,
#         model=model,
#         test_method="single_query",
#     )
#     if args.use_patch_rerank:
#         print("Patch-BoQ reranking enabled")
#         print(f"patch sizes: {args.patch_sizes}")
#         print(f"patch strides: {args.patch_strides}")
#         print(f"patch weights: {args.patch_weights}")
#         print(f"rerank top-k: {args.patch_rerank_topk}")
#         print(f"global weight: {args.patch_global_weight}")
#         print(f"spatial weight: {args.patch_spatial_weight}")
#     else:
#         print("Patch-BoQ reranking disabled; using global BoQ retrieval only.")
#         print(recall_str)
#     for key in ("recall@1", "recall@5", "recall@10", "recall@20"):
#         print(f"{key}: {metrics[key]:.4f}")

def main():
    args = parse_args()

    # ------------------------------------------------------------
    # Patch参数检查
    # ------------------------------------------------------------
    if not (
        len(args.patch_sizes)
        == len(args.patch_strides)
        == len(args.patch_weights)
    ):
        raise ValueError(
            "--patch_sizes、--patch_strides 和 --patch_weights "
            "必须包含相同数量的元素。"
        )

    if any(size <= 0 for size in args.patch_sizes):
        raise ValueError("--patch_sizes 必须全部大于0。")

    if any(stride <= 0 for stride in args.patch_strides):
        raise ValueError("--patch_strides 必须全部大于0。")

    if any(weight < 0 for weight in args.patch_weights):
        raise ValueError("--patch_weights 不能包含负数。")

    weight_sum = sum(args.patch_weights)
    if weight_sum <= 0:
        raise ValueError("--patch_weights 的总和必须大于0。")

    args.patch_weights = [
        weight / weight_sum
        for weight in args.patch_weights
    ]

    checkpoint_path = Path(args.checkpoint)
    hparams, hparams_path = _load_hparams(args)
    device = _resolve_device(args)

    # ============================================================
    # 这一行不能删除：
    # _build_model返回 model、query尺寸、database尺寸
    # ============================================================
    model, query_img_size, db_img_size = _build_model(hparams)

    # 加载训练好的KITTI360 checkpoint
    incompatible = _load_checkpoint(
        model,
        checkpoint_path,
        allow_partial=args.allow_partial_checkpoint,
    )

    model.to(device)
    model.eval()

    # 必须在query_img_size和db_img_size定义后执行
    test_dataset, eval_args = _build_test_dataset(
        hparams,
        query_img_size,
        db_img_size,
    )

    eval_args.device = str(device)

    # ============================================================
    # 将根目录evaluate.py中的Patch参数传给src/evaluate.py
    # ============================================================
    eval_args.use_patch_rerank = args.use_patch_rerank

    eval_args.patch_sizes = tuple(args.patch_sizes)
    eval_args.patch_strides = tuple(args.patch_strides)
    eval_args.patch_weights = tuple(args.patch_weights)

    eval_args.patch_rerank_topk = args.patch_rerank_topk
    eval_args.patch_global_weight = args.patch_global_weight
    eval_args.patch_spatial_weight = args.patch_spatial_weight
    eval_args.patch_spatial_tau = args.patch_spatial_tau
    eval_args.patch_score_batch_size = args.patch_score_batch_size

    print(f"checkpoint: {checkpoint_path}")
    if hparams_path is not None:
        print(f"hparams: {hparams_path}")
    print(f"device: {device}")

    print(
        "KITTI360 test split: "
        f"last {args.test_tail_ratio * 100:.1f}% "
        f"of each selected video sequence "
        f"(train_ratio={hparams.kitti360_train_ratio:.4f}, "
        f"share_db=False)"
    )

    print(
        f"queries: {test_dataset.queries_num}, "
        f"database: {test_dataset.database_num}"
    )

    if incompatible.missing_keys:
        print(
            f"missing checkpoint keys: "
            f"{len(incompatible.missing_keys)}"
        )

    if incompatible.unexpected_keys:
        print(
            f"unexpected checkpoint keys: "
            f"{len(incompatible.unexpected_keys)}"
        )

    if args.use_patch_rerank:
        print("Patch-BoQ reranking enabled")
        print(f"patch sizes: {args.patch_sizes}")
        print(f"patch strides: {args.patch_strides}")
        print(f"patch weights: {args.patch_weights}")
        print(f"rerank top-k: {args.patch_rerank_topk}")
        print(f"global weight: {args.patch_global_weight}")
        print(f"spatial weight: {args.patch_spatial_weight}")
    else:
        print(
            "Patch-BoQ reranking disabled; "
            "using global BoQ retrieval only."
        )

    _, recall_str, metrics = evaluate_kitti360(
        args=eval_args,
        test_ds=test_dataset,
        model=model,
        test_method="single_query",
    )

    print(recall_str)

    for key in (
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
    ):
        print(f"{key}: {metrics[key]:.4f}")



#  python evaluate.py --checkpoint logs/view_adapter_boq_dinov2_vitb14/version_x/checkpoints/last.ckpt --dataset_name nuscenes --nuscenes_path /mnt/sda/ZhengyiXu/datasets/radar/nuscenes --nuscenes_train_version v1.0-trainval --nuscenes_val_version v1.0-test --nuscenes_camnames CAM_FRONT --nuscenes_maptype satellite --nuscenes_aerial_scale 1  --nuscenes_aerial_zoom 20 --nuscenes_aerial_size 320 --infer_batch_size 8
if __name__ == "__main__":
    main()

