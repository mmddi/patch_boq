import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
try:
    from lightning.pytorch import Trainer, callbacks, seed_everything
    from lightning.pytorch.loggers import TensorBoardLogger
except ModuleNotFoundError:
    from pytorch_lightning import Trainer, callbacks, seed_everything
    from pytorch_lightning.loggers import TensorBoardLogger

from src.backbones import DinoV2, ResNet
from src.boq import BoQ

from src.pretrained_boq import (
    get_official_pretrained_boq_spec,
    load_pretrained_boq_into_dual_branch_encoder,
    load_pretrained_boq_into_view_adapter_encoder,
    load_pretrained_boq_into_shared_query_encoder,
    load_pretrained_boq_weights,
)
from src.utils import display_datasets_stats


CSV_FIELDS = [
    "epoch",
    "train_loss",
    # "val_loss",
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@20",
    "lr",
    "train_time",
    "val_time",
    "epoch_time",
]


def _scalar_to_float(value, default=float("nan")):
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().float().mean().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_nan(value, digits=4):
    value = _scalar_to_float(value)
    if math.isfinite(value):
        return round(value, digits)
    return float("nan")


def _format_value(value, digits=4):
    value = _scalar_to_float(value)
    if math.isfinite(value):
        return f"{value:.{digits}f}"
    return "nan"


def append_metrics_to_csv(csv_path: Path, row_dict: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def ensure_metrics_csv(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()


def _find_metric(callback_metrics, *names):
    for name in names:
        if name in callback_metrics:
            return _scalar_to_float(callback_metrics[name])
    return float("nan")


def _find_recall_metric(callback_metrics, rank):
    direct_names = [f"val/recall@{rank}", f"val_recall{rank}", f"recall@{rank}"]
    for name in direct_names:
        if name in callback_metrics:
            return _scalar_to_float(callback_metrics[name])

    suffixes = (f"/R@{rank}", f"/recall@{rank}", f"R@{rank}", f"recall@{rank}")
    for name, value in callback_metrics.items():
        if any(str(name).endswith(suffix) for suffix in suffixes):
            return _scalar_to_float(value)
    return float("nan")


def _current_lr(trainer):
    if not trainer.optimizers:
        return float("nan")
    return _scalar_to_float(trainer.optimizers[0].param_groups[0].get("lr"))


def _has_validation_batches(trainer):
    num_val_batches = getattr(trainer, "num_val_batches", None)
    if num_val_batches is None:
        return False
    if isinstance(num_val_batches, (list, tuple)):
        return sum(int(batch_count) for batch_count in num_val_batches) > 0
    return int(num_val_batches) > 0


def _collect_metric_sources(trainer):
    merged = {}
    for attr_name in ("callback_metrics", "logged_metrics", "progress_bar_metrics"):
        metrics = getattr(trainer, attr_name, None)
        if not metrics:
            continue
        merged.update(metrics)
    return merged


def _has_configured_validation(trainer):
    val_dataloaders = getattr(trainer, "val_dataloaders", None)
    if val_dataloaders is None:
        return False
    if isinstance(val_dataloaders, (list, tuple)):
        return len(val_dataloaders) > 0
    return True


class EpochMetricsCallback(callbacks.Callback):
    def __init__(self, csv_filename="train_metrics.csv"):
        super().__init__()
        self.csv_filename = csv_filename
        self.csv_path = None
        self.best_recall1 = float("-inf")
        self.epoch_start_time = None
        self.train_start_time = None
        self.val_start_time = None
        self.train_time = float("nan")
        self.val_time = float("nan")
        self._epoch_logged = False
        self._train_epoch_finished = False
        self._validation_finished = False

    def on_fit_start(self, trainer, pl_module):
        log_dir = getattr(trainer.logger, "log_dir", None)
        self.csv_path = Path(log_dir) / self.csv_filename if log_dir else Path("logs") / self.csv_filename
        self.best_recall1 = _scalar_to_float(getattr(pl_module, "best_recall1", float("-inf")), default=float("-inf"))
        if trainer.is_global_zero and self.csv_path is not None and not self.csv_path.exists():
            ensure_metrics_csv(self.csv_path)

    def on_train_epoch_start(self, trainer, pl_module):
        del pl_module
        if trainer.sanity_checking:
            return
        now = time.perf_counter()
        self.epoch_start_time = now
        self.train_start_time = now
        self.val_start_time = None
        self.train_time = float("nan")
        self.val_time = float("nan")
        self._epoch_logged = False
        self._train_epoch_finished = False
        self._validation_finished = False

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        now = time.perf_counter()
        if self.train_start_time is None:
            return

        if self.val_start_time is not None:
            self.train_time = max(0.0, self.val_start_time - self.train_start_time)
        else:
            self.train_time = now - self.train_start_time

        self._train_epoch_finished = True
        if not _has_configured_validation(trainer) or self._validation_finished:
            self._finalize_epoch(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        del pl_module
        if trainer.sanity_checking:
            return
        self.val_start_time = time.perf_counter()

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.val_start_time is not None:
            self.val_time = time.perf_counter() - self.val_start_time
        self._validation_finished = True
        if self._train_epoch_finished:
            self._finalize_epoch(trainer, pl_module)

    def _finalize_epoch(self, trainer, pl_module=None):
        if self._epoch_logged or self.epoch_start_time is None:
            return

        callback_metrics = _collect_metric_sources(trainer)
        eval_metrics = getattr(pl_module, "last_eval_metrics", {}) if pl_module is not None else {}
        recall1 = _find_recall_metric(callback_metrics, 1)
        recall5 = _find_recall_metric(callback_metrics, 5)
        recall10 = _find_recall_metric(callback_metrics, 10)
        recall20 = _find_recall_metric(callback_metrics, 20)
        row = {
            "epoch": trainer.current_epoch + 1,
            "train_loss": _round_or_nan(
                _find_metric(callback_metrics, "train_loss", "train_loss_epoch", "loss_total_epoch", "loss_epoch", "loss"),
                digits=4,
            ),
            # "val_loss": _round_or_nan(_find_metric(callback_metrics, "val_loss", "val_loss_epoch"), digits=4),
            "recall@1": _round_or_nan(recall1 if math.isfinite(recall1) else eval_metrics.get("recall@1"), digits=4),
            "recall@5": _round_or_nan(recall5 if math.isfinite(recall5) else eval_metrics.get("recall@5"), digits=4),
            "recall@10": _round_or_nan(recall10 if math.isfinite(recall10) else eval_metrics.get("recall@10"), digits=4),
            "recall@20": _round_or_nan(recall20 if math.isfinite(recall20) else eval_metrics.get("recall@20"), digits=4),
            "lr": _round_or_nan(_current_lr(trainer), digits=6),
            "train_time": _round_or_nan(self.train_time, digits=3),
            "val_time": _round_or_nan(self.val_time, digits=3),
            "epoch_time": _round_or_nan(time.perf_counter() - self.epoch_start_time, digits=3),
        }

        if pl_module is not None:
            pl_module.last_epoch_metrics = dict(row)

        if trainer.logger is not None:
            logger_metrics = {
                "train/loss": row["train_loss"],
                # "val/loss": row["val_loss"],
                "val/recall@1": row["recall@1"],
                "val/recall@5": row["recall@5"],
                "val/recall@10": row["recall@10"],
                "val/recall@20": row["recall@20"],
                "train/lr": row["lr"],
                "time/train_time": row["train_time"],
                "time/val_time": row["val_time"],
                "time/epoch_time": row["epoch_time"],
            }
            logger_metrics = {key: value for key, value in logger_metrics.items() if math.isfinite(value)}
            if logger_metrics:
                trainer.logger.log_metrics(logger_metrics, step=row["epoch"])

        recall1 = row["recall@1"]
        best_updated = False
        if math.isfinite(recall1) and recall1 > self.best_recall1:
            self.best_recall1 = recall1
            best_updated = True
            if pl_module is not None:
                pl_module.best_recall1 = recall1

        if trainer.is_global_zero:
            if self.csv_path is not None:
                append_metrics_to_csv(self.csv_path, row)

            summary = (
                f"Epoch {row['epoch']}/{trainer.max_epochs} | "
                f"train_loss={_format_value(row['train_loss'])} | "
                # f"val_loss={_format_value(row['val_loss'])} | "
                f"R@1={_format_value(row['recall@1'], digits=2)} | "
                f"R@5={_format_value(row['recall@5'], digits=2)} | "
                f"R@10={_format_value(row['recall@10'], digits=2)} | "
                f"R@20={_format_value(row['recall@20'], digits=2)} | "
                f"lr={_format_value(row['lr'], digits=6)} | "
                f"train_time={_format_value(row['train_time'], digits=3)}s | "
                f"val_time={_format_value(row['val_time'], digits=3)}s | "
                f"epoch_time={_format_value(row['epoch_time'], digits=3)}s"
            )
            print(summary)
            if best_updated:
                print(f"Best Recall@1 updated: {recall1:.4f}")

        self._epoch_logged = True



class HyperParams:
    def __init__(self):
        project_root = Path(__file__).resolve().parent

        # ============================================================
        # 1. 当前固定分支
        # ============================================================
        self.dataset_name = "kitti360"
        self.use_kitti360_boq = True
        self.use_view_specific_adapter = True

        # train() 会直接读取这些属性，因此必须定义。
        # 当前主路径中必须保持为 False。
        self.dual_branch = False
        self.shared_query_boq = False
        self.use_dinov2_block_adapter = False

        # ============================================================
        # 2. DINOv2 backbone
        # ============================================================
        self.backbone_name = "dinov2_vitb14"

        # freeze_backbone=True 时，unfreeze_n_blocks 不产生实际解冻效果。
        self.freeze_backbone = True
        self.unfreeze_n_blocks = 0

        # 训练开始后的额外临时冻结轮数。
        # backbone 本身已永久冻结，因此当前保持 0。
        self.backbone_warmup_epochs = 0

        # ============================================================
        # 3. Ground / Satellite view-specific Adapter
        # ============================================================
        # None 表示在全部 DINOv2 block 中插入 Adapter。
        self.adapter_layers = None
        self.adapter_bottleneck_dim = 128
        self.adapter_scale = 0.1
        self.adapter_zero_init = True

        # 当前代码仍然会读取和传递该字段。
        # 名字虽然是 spatial，实际使用的是 MLP Adapter。
        self.adapter_type = "spatial"

        self.train_ground_adapter = True
        self.train_sat_adapter = True

        # ============================================================
        # 4. 预训练 DINOv2 + BoQ
        # ============================================================
        # 原来的 self.loss = True 是错误字段，应改为：
        self.use_pretrained_boq = True

        # 推荐将权重放在：
        # 项目根目录/pretrained/dinov2_12288.pth
        self.pretrained_boq_path = "src/dinov2_12288.pth"

        # 必须与 dinov2_12288.pth 的 BoQ 结构保持一致。
        self.channel_proj = 384
        self.num_queries = 64
        self.num_layers = 2
        self.output_dim = 12288
        self.normalize_descriptor = True

        # 训练共享 BoQ。
        self.freeze_aggregator = False

        # ============================================================
        # 5. KITTI360 数据
        # ============================================================
        self.kitti360_path = "/mnt/sda/ZhengyiXu/datasets/cmvpr/kitti360/KITTI-360"

        # ground query 使用 00 相机，database 使用 satellite。
        self.kitti360_camnames = "00"
        self.kitti360_maptype = "satellite"

        self.kitti360_train_ratio = 0.85
        self.kitti360_share_db = False
        self.kitti360_traindownsample = 1

        # None 表示继承 DINOv2 默认训练尺寸。
        self.kitti360_query_img_size = None
        self.kitti360_db_img_size = None

        self.kitti360_db_crop_size = 320
        self.kitti360_q_jitter = 0.0
        self.kitti360_db_jitter = 0.0

        # ============================================================
        # 6. Triplet mining
        # ============================================================
        self.kitti360_mining = "partial_sep"
        self.kitti360_neg_samples_num = 2000
        self.kitti360_pos_num_per_query = 4
        self.kitti360_negs_num_per_query = 6
        self.kitti360_cache_refresh_rate = 1280
        self.kitti360_infer_batch_size = 32

        self.kitti360_train_positives_dist_threshold = 10.0
        self.kitti360_val_positive_dist_threshold = 25.0


        self.nuscenes_path = "/mnt/sda/ZhengyiXu/datasets/radar/nuscenes"

        self.nuscenes_train_version = "v1.0-trainval"
        self.nuscenes_val_version = "v1.0-test"

        self.nuscenes_locations = (
            "boston-seaport,"
            "singapore-hollandvillage,"
            "singapore-onenorth,"
            "singapore-queenstown"
        )

        self.nuscenes_camnames = "CAM_FRONT"
        self.nuscenes_maptype = "satellite"

        self.nuscenes_aerial_scale = 1
        self.nuscenes_aerial_zoom = 20
        self.nuscenes_aerial_size = 320

        self.nuscenes_train_ratio = 1.0
        self.nuscenes_share_db = False
        self.nuscenes_traindownsample = 1

        self.nuscenes_query_img_size = None
        self.nuscenes_db_img_size = None
        self.nuscenes_db_crop_size = 320
        self.nuscenes_q_jitter = 0.0
        self.nuscenes_db_jitter = 0.0

        self.nuscenes_mining = "partial_sep"
        self.nuscenes_neg_samples_num = 1000
        self.nuscenes_pos_num_per_query = 4
        self.nuscenes_negs_num_per_query = 6
        self.nuscenes_cache_refresh_rate = 1024
        self.nuscenes_infer_batch_size = 32

        self.nuscenes_train_positives_dist_threshold = 10.0
        self.nuscenes_val_positive_dist_threshold = 25.0

        # ============================================================
        # 7. Triplet 模型
        # ============================================================
        self.satellite_map_index = 0
        self.db_encode_chunk_size = 16

        self.loss_type = "triplet"
        self.triplet_margin = 0.2

        # 仅用于调试打印训练样本。
        self.print_train_items = 0
        self.print_train_items_every_epoch = False

        # ============================================================
        # 8. 优化器和训练器
        # ============================================================
        self.batch_size = 16
        self.max_epochs = 40
        self.num_workers = 8

        self.lr = 5e-5
        self.weight_decay = 1e-4

        self.warmup_epochs = 5
        self.lr_mul = 0.2
        self.milestones = [5, 20]

        self.gpu_id = 1
        self.seed = 2026
        self.log_every_n_steps = 1

        # ============================================================
        # 9. 日志和调试
        # ============================================================
        self.silent = True
        self.debug_shapes = False
        self.print_model_modules = False

        # 当前 KITTI360 分支不会执行 torch.compile，
        # 但 train.py 仍会读取该属性，因此保留。
        self.compile = False


   

def build_backbone_from_name(
    backbone_name,
    unfreeze_n_blocks,
    pretrained=True,
    use_dinov2_block_adapter=False,
    adapter_bottleneck_dim=128,
    adapter_scale=0.1,
    adapter_zero_init=True,
):
    if "dinov2" in backbone_name:
        backbone = DinoV2(
            backbone_name=backbone_name,
            pretrained=pretrained,
            unfreeze_n_blocks=unfreeze_n_blocks,
            use_block_adapter=use_dinov2_block_adapter,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_scale=adapter_scale,
            adapter_zero_init=adapter_zero_init,
        )
        train_img_size = (224, 224)
        val_img_size = (322, 322)
        resolved_backbone_name = backbone.backbone_name
    elif "resnet" in backbone_name:
        backbone = ResNet(
            backbone_name=backbone_name,
            pretrained=pretrained,
            unfreeze_n_blocks=unfreeze_n_blocks,
            crop_last_block=True,
            freeze_layers=True,
        )
        train_img_size = (320, 320)
        val_img_size = (384, 384)
        resolved_backbone_name = backbone_name
    else:
        raise ValueError(f"backbone {backbone_name} not recognized or not implemented!")

    return backbone, train_img_size, val_img_size, resolved_backbone_name

#根据 backbone 名称构建骨干网络
def build_backbone(hparams):
    backbone, train_img_size, val_img_size, resolved_backbone_name = build_backbone_from_name(
        hparams.backbone_name,
        hparams.unfreeze_n_blocks,
        pretrained=not hparams.use_pretrained_boq,
        use_dinov2_block_adapter=hparams.use_dinov2_block_adapter,
        adapter_bottleneck_dim=hparams.adapter_bottleneck_dim,
        adapter_scale=hparams.adapter_scale,
        adapter_zero_init=hparams.adapter_zero_init,
    )
    hparams.backbone_name = resolved_backbone_name

    hparams.train_img_size = train_img_size
    hparams.val_img_size = val_img_size
    return backbone, train_img_size, val_img_size

#构建 BoQ（Bag-of-Queries）聚合器
def build_aggregator(hparams, backbone=None, in_channels=None):
    if in_channels is None:
        if backbone is None:
            raise ValueError("Either backbone or in_channels must be provided to build the aggregator.")
        in_channels = backbone.out_channels
    return BoQ(
        in_channels=in_channels,
        proj_channels=hparams.channel_proj,
        num_queries=hparams.num_queries,
        num_layers=hparams.num_layers,
        row_dim=hparams.output_dim // hparams.channel_proj,
    )

#同步超参数以匹配预训练 BoQ 的规格
def _sync_hparams_with_official_pretrained_boq(hparams):
    if not hparams.use_pretrained_boq:
        return None, []

    if hparams.use_kitti360_boq and (hparams.dual_branch or hparams.shared_query_boq):
        query_backbone_name = hparams.query_backbone or hparams.backbone_name
        db_backbone_name = hparams.db_backbone or hparams.backbone_name

        query_spec = get_official_pretrained_boq_spec(query_backbone_name)
        db_spec = get_official_pretrained_boq_spec(db_backbone_name)
        if query_spec["backbone_name"] != db_spec["backbone_name"]:
            raise ValueError(
                "Official pre-trained BoQ initialization for dual-branch/shared-query training currently "
                "requires query_backbone and db_backbone to use the same official backbone family."
            )
        spec = query_spec
    else:
        spec = get_official_pretrained_boq_spec(hparams.backbone_name)

    changed_values = []
    expected_values = {
        "channel_proj": spec["proj_channels"],
        "num_queries": spec["num_queries"],
        "num_layers": spec["num_layers"],
        "output_dim": spec["output_dim"],
    }
    for attr_name, expected_value in expected_values.items():
        current_value = getattr(hparams, attr_name)
        if current_value != expected_value:
            setattr(hparams, attr_name, expected_value)
            changed_values.append((attr_name, current_value, expected_value))

    return spec, changed_values

#格式化预训练权重加载报告
def _format_pretrained_boq_report(report):
    def _format_items(items, max_items=8):
        if not items:
            return "[]"
        items = list(items)
        shown_items = ", ".join(str(item) for item in items[:max_items])
        if len(items) > max_items:
            shown_items += f", ... (+{len(items) - max_items} more)"
        return f"[{shown_items}]"

    module_summaries = []
    for module_name, module_report in report.items():
        if module_name == "spec":
            continue
        module_summary = (
            f"{module_name}:loaded={module_report['loaded_keys']},"
            f"missing={len(module_report['missing_keys'])},"
            f"unexpected={len(module_report['unexpected_keys'])},"
            f"skipped={len(module_report['skipped_keys'])}"
        )
        details = []
        if module_report["missing_keys"]:
            details.append(f"missing_keys={_format_items(module_report['missing_keys'])}")
        if module_report["unexpected_keys"]:
            details.append(f"unexpected_keys={_format_items(module_report['unexpected_keys'])}")
        if module_report["skipped_keys"]:
            details.append(f"skipped_keys={_format_items(module_report['skipped_keys'].keys())}")
        initialized_keys = module_report.get("initialized_keys", [])
        if initialized_keys:
            details.append(f"initialized_keys={_format_items(initialized_keys)}")
        if details:
            module_summary += " (" + "; ".join(details) + ")"
        module_summaries.append(module_summary)
    return " | ".join(module_summaries)

#将预训练 BoQ 权重加载到模型中
def _initialize_model_from_pretrained_boq(hparams, model):
    if not hparams.use_pretrained_boq:
        return None
    use_cross_view_boq = bool(
        getattr(hparams, "use_kitti360_boq", False)
        or getattr(hparams, "use_nuscenes_boq", False))
    checkpoint_path = getattr(hparams, "pretrained_boq_path", None)

    if use_cross_view_boq  and hparams.shared_query_boq:
        return load_pretrained_boq_into_shared_query_encoder(
            image_encoder=model.image_encoder,
            backbone_name=hparams.query_backbone,
            output_dim=hparams.output_dim,
            strict=True,
            checkpoint_path=checkpoint_path,
        )

    if use_cross_view_boq  and getattr(model, "is_view_specific_adapter_encoder", False):
        return load_pretrained_boq_into_view_adapter_encoder(
            image_encoder=model.image_encoder,
            backbone_name=hparams.backbone_name,
            output_dim=hparams.output_dim,
            strict=False,
            checkpoint_path=checkpoint_path,
        )

    if use_cross_view_boq  and getattr(model, "is_dual_branch", False):
        return load_pretrained_boq_into_dual_branch_encoder(
            image_encoder=model.image_encoder,
            backbone_name=hparams.query_backbone,
            output_dim=hparams.output_dim,
            strict=True,
            checkpoint_path=checkpoint_path,
        )

    if hparams.use_kitti360_boq:
        return load_pretrained_boq_weights(
            backbone=model.image_encoder.backbone,
            aggregator=model.image_encoder.aggregator,
            backbone_name=hparams.backbone_name,
            output_dim=hparams.output_dim,
            strict=True,
            checkpoint_path=checkpoint_path,
        )

    return load_pretrained_boq_weights(
        backbone=model.backbone,
        aggregator=model.aggregator,
        backbone_name=hparams.backbone_name,
        output_dim=hparams.output_dim,
        strict=True,
        checkpoint_path=checkpoint_path,
    )


def _is_multi_device(devices):
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return False


def _distributed_world_size():
    for env_name in ("WORLD_SIZE", "SLURM_NTASKS", "LOCAL_WORLD_SIZE"):
        env_value = os.environ.get(env_name)
        if env_value is None:
            continue
        try:
            world_size = int(env_value)
        except ValueError:
            continue
        if world_size > 0:
            return world_size
    return 1


def resolve_trainer_device_config(hparams):
    if hparams.gpu_id is None:
        accelerator = "auto"
        devices = 1
    else:
        if not torch.cuda.is_available():
            raise ValueError(f"--gpu_id={hparams.gpu_id} was requested, but CUDA is not available.")
        available_gpu_count = torch.cuda.device_count()

        if isinstance(hparams.gpu_id, (list, tuple)):
            devices = [int(device_id) for device_id in hparams.gpu_id]
            if not devices:
                raise ValueError("--gpu_id must contain at least one GPU id when a sequence is provided.")
            invalid_devices = [device_id for device_id in devices if device_id < 0 or device_id >= available_gpu_count]
            if invalid_devices:
                raise ValueError(
                    f"--gpu_id entries must be in [0, {available_gpu_count - 1}], got {invalid_devices}."
                )
            accelerator = "gpu"
        else:
            if hparams.gpu_id < 0 or hparams.gpu_id >= available_gpu_count:
                raise ValueError(
                    f"--gpu_id must be in [0, {available_gpu_count - 1}], got {hparams.gpu_id}."
                )
            accelerator = "gpu"
            devices = [hparams.gpu_id]

    precision = "16-mixed" if torch.cuda.is_available() else 32
    return accelerator, devices, precision


def resolve_trainer_strategy(hparams, accelerator, devices):
    del accelerator

    is_distributed = _is_multi_device(devices) or _distributed_world_size() > 1
    if not is_distributed:
        return "auto"

    # During backbone warmup some branches temporarily exclude backbone
    # parameters from gradient updates, so DDP must tolerate unused parameters
    # on each rank.
    if getattr(hparams, "backbone_warmup_epochs", 0) > 0:
        return "ddp_find_unused_parameters_true"

    return "auto"

def train(hparams, dev_mode=False):
    # ============================================================
    # 0. Dataset switch
    # ============================================================
    dataset_name = str(getattr(hparams, "dataset_name", "kitti360")).lower()

    # 兼容用户拼写
    if dataset_name in {"nuscenes", "nuscene", "nescenes", "nuscences"}:
        dataset_name = "nuscenes"

    # 如果命令行显式打开 --use_nuscenes_boq，优先走 nuScenes
    if getattr(hparams, "use_nuscenes_boq", False) or dataset_name == "nuscenes":
        hparams.dataset_name = "nuscenes"
        hparams.use_nuscenes_boq = True
        hparams.use_kitti360_boq = False
    else:
        hparams.dataset_name = "kitti360"
        hparams.use_kitti360_boq = True
        hparams.use_nuscenes_boq = False

    use_cross_view_boq = bool(
        getattr(hparams, "use_kitti360_boq", False)
        or getattr(hparams, "use_nuscenes_boq", False)
    )

    if hparams.shared_query_boq:
        hparams.dual_branch = True

    pretrained_boq_spec, changed_pretrained_hparams = _sync_hparams_with_official_pretrained_boq(hparams)

    seed_everything(hparams.seed, workers=True)
    accelerator, devices, precision = resolve_trainer_device_config(hparams)
    strategy = resolve_trainer_strategy(hparams, accelerator, devices)

    backbone = None
    aggregator = None
    pretrained_boq_report = None

    use_dual_branch = False
    use_view_specific_adapter = False

    # ============================================================
    # 1. KITTI360 / nuScenes cross-view BoQ branch
    # ============================================================
    if use_cross_view_boq:
        from src.models.dino_boq_view_adapter import DinoBoQViewSpecificAdapterEncoder
        from src.models.kitti360_boq_triplet import Kitti360BoQTripletModel

        if hparams.use_nuscenes_boq:
            from src.dataloaders.nuscenes_datamodule import NuScenesTripletDataModule as CrossViewTripletDataModule

            dataset_tag = "NUSCENES"

            if not getattr(hparams, "nuscenes_path", None):
                raise ValueError(
                    "Please set --nuscenes_path when using --use_nuscenes_boq "
                    "or --dataset_name nuscenes."
                )

        else:
            from src.dataloaders.kitti360_datamodule import Kitti360TripletDataModule as CrossViewTripletDataModule

            dataset_tag = "KITTI360"

            if not getattr(hparams, "kitti360_path", None):
                raise ValueError(
                    "Please set --kitti360_path when using --use_kitti360_boq "
                    "or --dataset_name kitti360."
                )

        # ------------------------------------------------------------
        # 1.1 Model checks
        # ------------------------------------------------------------
        use_view_specific_adapter = bool(getattr(hparams, "use_view_specific_adapter", False))
        use_dual_branch = bool(getattr(hparams, "dual_branch", False) or getattr(hparams, "shared_query_boq", False))

        if use_view_specific_adapter and use_dual_branch:
            raise ValueError(
                "use_view_specific_adapter cannot be combined with dual_branch or shared_query_boq. "
                "The view-specific adapter model already uses a single shared backbone."
            )

        if use_view_specific_adapter and getattr(hparams, "use_dinov2_block_adapter", False):
            raise ValueError(
                "use_view_specific_adapter cannot be combined with dinov2_block_adapter. "
                "Both options insert adapters into the same DINOv2 blocks."
            )

        if use_dual_branch and getattr(hparams, "use_dinov2_block_adapter", False):
            raise ValueError(
                "dinov2_block_adapter is currently supported only for the standard single-backbone flow. "
                "Please disable dual_branch/shared_query_boq, or keep using the existing branch adapters."
            )

        if not use_view_specific_adapter:
            raise NotImplementedError(
                "当前这版 cross-view train() 只接通了 view_specific_adapter 分支。"
                "请训练时保留 --view_specific_adapter。"
            )

        if "dinov2" not in hparams.backbone_name:
            raise ValueError(
                "The shared-backbone view-specific adapter model currently supports DINOv2 backbones only, "
                f"got {hparams.backbone_name!r}."
            )

        # ------------------------------------------------------------
        # 1.2 Build image encoder
        # ------------------------------------------------------------
        backbone, train_img_size, _ = build_backbone(hparams)

        if getattr(hparams, "adapter_layers", None) is None:
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

        # ------------------------------------------------------------
        # 1.3 Image size
        # ------------------------------------------------------------
        if hparams.use_nuscenes_boq:
            query_img_size = getattr(hparams, "nuscenes_query_img_size", None) or train_img_size
            db_img_size = getattr(hparams, "nuscenes_db_img_size", None) or train_img_size
        else:
            query_img_size = getattr(hparams, "kitti360_query_img_size", None) or train_img_size
            db_img_size = getattr(hparams, "kitti360_db_img_size", None) or train_img_size

        # ------------------------------------------------------------
        # 1.4 Build Lightning model
        # ------------------------------------------------------------
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
            silent=hparams.silent,
            debug_shapes=hparams.debug_shapes,
            print_train_items=hparams.print_train_items,
            print_train_items_every_epoch=hparams.print_train_items_every_epoch,
        )

        model = Kitti360BoQTripletModel(
            image_encoder=image_encoder,
            **model_kwargs,
        )

        # ------------------------------------------------------------
        # 1.5 Load pretrained BoQ correctly for KITTI360 / nuScenes
        #     不再直接调用 _initialize_model_from_pretrained_boq，
        #     因为原函数只判断 use_kitti360_boq，nuScenes 会走错分支。
        # ------------------------------------------------------------
        if getattr(hparams, "use_pretrained_boq", False):
            checkpoint_path = getattr(hparams, "pretrained_boq_path", None)

            if getattr(model, "is_view_specific_adapter_encoder", False):
                pretrained_boq_report = load_pretrained_boq_into_view_adapter_encoder(
                    image_encoder=model.image_encoder,
                    backbone_name=hparams.backbone_name,
                    output_dim=hparams.output_dim,
                    strict=False,
                    checkpoint_path=checkpoint_path,
                )
            elif getattr(model, "is_dual_branch", False):
                pretrained_boq_report = load_pretrained_boq_into_dual_branch_encoder(
                    image_encoder=model.image_encoder,
                    backbone_name=getattr(hparams, "query_backbone", hparams.backbone_name),
                    output_dim=hparams.output_dim,
                    strict=True,
                    checkpoint_path=checkpoint_path,
                )
            elif getattr(hparams, "shared_query_boq", False):
                pretrained_boq_report = load_pretrained_boq_into_shared_query_encoder(
                    image_encoder=model.image_encoder,
                    backbone_name=getattr(hparams, "query_backbone", hparams.backbone_name),
                    output_dim=hparams.output_dim,
                    strict=True,
                    checkpoint_path=checkpoint_path,
                )
            else:
                pretrained_boq_report = load_pretrained_boq_weights(
                    backbone=model.image_encoder.backbone,
                    aggregator=model.image_encoder.aggregator,
                    backbone_name=hparams.backbone_name,
                    output_dim=hparams.output_dim,
                    strict=True,
                    checkpoint_path=checkpoint_path,
                )

        # ------------------------------------------------------------
        # 1.6 Build DataModule
        # ------------------------------------------------------------
        if hparams.use_nuscenes_boq:
            datamodule = CrossViewTripletDataModule(
                dataroot=hparams.nuscenes_path,
                batch_size=hparams.batch_size,
                num_workers=hparams.num_workers,
                features_dim=model.descriptor_dim,
                query_img_size=query_img_size,
                db_img_size=db_img_size,
                db_crop_size=getattr(hparams, "nuscenes_db_crop_size", 320),
                q_jitter=getattr(hparams, "nuscenes_q_jitter", 0.0),
                db_jitter=getattr(hparams, "nuscenes_db_jitter", 0.0),
                train_version=getattr(hparams, "nuscenes_train_version", "v1.0-trainval"),
                val_version=getattr(hparams, "nuscenes_val_version", "v1.0-test"),
                train_ratio=getattr(hparams, "nuscenes_train_ratio", 1.0),
                share_db=getattr(hparams, "nuscenes_share_db", False),
                traindownsample=getattr(hparams, "nuscenes_traindownsample", 1),
                camnames=getattr(hparams, "nuscenes_camnames", "CAM_FRONT"),
                maptype=getattr(hparams, "nuscenes_maptype", "satellite"),
                locations=getattr(
                    hparams,
                    "nuscenes_locations",
                    "boston-seaport,singapore-hollandvillage,singapore-onenorth,singapore-queenstown",
                ),
                aerial_scale=getattr(hparams, "nuscenes_aerial_scale", 1),
                aerial_zoom=getattr(hparams, "nuscenes_aerial_zoom", 20),
                aerial_size=getattr(hparams, "nuscenes_aerial_size", 320),
                use_keyframes_only=getattr(hparams, "nuscenes_use_keyframes_only", True),
                mining=getattr(hparams, "nuscenes_mining", "partial_sep"),
                neg_samples_num=getattr(hparams, "nuscenes_neg_samples_num", 1000),
                pos_num_per_query=getattr(hparams, "nuscenes_pos_num_per_query", 4),
                negs_num_per_query=getattr(hparams, "nuscenes_negs_num_per_query", 6),
                cache_refresh_rate=getattr(hparams, "nuscenes_cache_refresh_rate", 1024),
                infer_batch_size=getattr(hparams, "nuscenes_infer_batch_size", 32),
                train_positives_dist_threshold=getattr(
                    hparams,
                    "nuscenes_train_positives_dist_threshold",
                    10.0,
                ),
                val_positive_dist_threshold=getattr(
                    hparams,
                    "nuscenes_val_positive_dist_threshold",
                    25.0,
                ),
                shuffle=True,
            )

        else:
            datamodule = CrossViewTripletDataModule(
                dataroot=hparams.kitti360_path,
                batch_size=hparams.batch_size,
                num_workers=hparams.num_workers,
                features_dim=model.descriptor_dim,
                query_img_size=query_img_size,
                db_img_size=db_img_size,
                db_crop_size=hparams.kitti360_db_crop_size,
                q_jitter=hparams.kitti360_q_jitter,
                db_jitter=hparams.kitti360_db_jitter,
                train_ratio=hparams.kitti360_train_ratio,
                share_db=hparams.kitti360_share_db,
                traindownsample=hparams.kitti360_traindownsample,
                camnames=hparams.kitti360_camnames,
                maptype=hparams.kitti360_maptype,
                mining=hparams.kitti360_mining,
                neg_samples_num=hparams.kitti360_neg_samples_num,
                pos_num_per_query=hparams.kitti360_pos_num_per_query,
                negs_num_per_query=hparams.kitti360_negs_num_per_query,
                cache_refresh_rate=hparams.kitti360_cache_refresh_rate,
                infer_batch_size=hparams.kitti360_infer_batch_size,
                train_positives_dist_threshold=hparams.kitti360_train_positives_dist_threshold,
                val_positive_dist_threshold=hparams.kitti360_val_positive_dist_threshold,
                shuffle=True,
            )

        checkpointing = callbacks.ModelCheckpoint(
            monitor="val_recall1",
            filename="epoch[{epoch:02d}]_R1[{val_recall1:.2f}]_R5[{val_recall5:.2f}]",
            auto_insert_metric_name=False,
            save_weights_only=False,
            save_top_k=3,
            save_last=True,
            mode="max",
        )

        limit_val_batches = 1.0

    # ============================================================
    # 2. Original non-cross-view branch
    #    一般你当前项目不会走这里，保留原逻辑。
    # ============================================================
    else:
        backbone, train_img_size, val_img_size = build_backbone(hparams)
        aggregator = build_aggregator(hparams, backbone=backbone)

        model = BoQModel(
            backbone,
            aggregator,
            lr=hparams.lr,
            lr_mul=hparams.lr_mul,
            weight_decay=hparams.weight_decay,
            warmup_epochs=hparams.warmup_epochs,
            milestones=hparams.milestones,
            silent=hparams.silent,
            debug_shapes=hparams.debug_shapes,
        )

        pretrained_boq_report = _initialize_model_from_pretrained_boq(hparams, model)

        datamodule = VPRDataModule(
            gsv_cities_path=hparams.gsv_cities_path,
            cities=hparams.cities,
            img_per_place=hparams.img_per_place,
            val_sets=hparams.val_sets,
            train_img_size=train_img_size,
            val_img_size=val_img_size,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            shuffle=False,
        )

        checkpointing = callbacks.ModelCheckpoint(
            monitor="msls-val/R@1",
            filename="epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]",
            auto_insert_metric_name=False,
            save_weights_only=False,
            save_top_k=3,
            mode="max",
        )

        limit_val_batches = 1.0

    # ============================================================
    # 3. Compile
    # ============================================================
    if getattr(hparams, "compile", False) and not use_cross_view_boq:
        model = torch.compile(model)

    # ============================================================
    # 4. Logging / debug print
    # ============================================================
    if not hparams.silent:
        if hparams.use_pretrained_boq and pretrained_boq_spec is not None:
            print(
                "[PretrainedBoQ]",
                f"backbone={pretrained_boq_spec['backbone_name']}",
                f"output_dim={pretrained_boq_spec['output_dim']}",
                f"proj_channels={pretrained_boq_spec['proj_channels']}",
                f"num_queries={pretrained_boq_spec['num_queries']}",
                f"num_layers={pretrained_boq_spec['num_layers']}",
            )

            if changed_pretrained_hparams:
                changed_summary = ", ".join(
                    f"{name}:{old}->{new}"
                    for name, old, new in changed_pretrained_hparams
                )
                print("[PretrainedBoQ]", f"aligned_hparams={changed_summary}")

            if pretrained_boq_report is not None:
                print("[PretrainedBoQ]", _format_pretrained_boq_report(pretrained_boq_report))

        datamodule.setup()

        if use_cross_view_boq:
            if use_view_specific_adapter:
                print(
                    f"[{dataset_tag}][ViewSpecificAdapter]",
                    f"backbone={hparams.backbone_name}",
                    "adapter_impl=dual_dinov2_block_adapter",
                    f"adapter_type={hparams.adapter_type}",
                    f"adapter_layers={list(model.image_encoder.adapter_layers)}",
                    f"adapter_bottleneck_dim={hparams.adapter_bottleneck_dim}",
                    f"adapter_scale={hparams.adapter_scale}",
                    f"freeze_backbone={hparams.freeze_backbone}",
                    f"unfreeze_last_n_blocks={hparams.unfreeze_n_blocks}",
                    f"freeze_aggregation={hparams.freeze_aggregator}",
                    f"train_ground_adapter={hparams.train_ground_adapter}",
                    f"train_sat_adapter={hparams.train_sat_adapter}",
                    f"backbone_warmup_epochs={hparams.backbone_warmup_epochs}",
                )

                print(f"[{dataset_tag}][ViewSpecificAdapter] adapter_status:")
                for line in model.image_encoder.format_adapter_status_lines():
                    print(f"  {line}")

                print(f"[{dataset_tag}][ViewSpecificAdapter] trainable_params:")
                for line in model.image_encoder.format_trainable_summary_lines():
                    print(f"  {line}")

                print(f"[{dataset_tag}][ViewSpecificAdapter] module_summary:")
                for line in model.image_encoder.format_module_summary_lines():
                    print(f"  {line}")

            print(
                f"[{dataset_tag}]",
                f"query_size={datamodule.query_img_size}",
                f"db_size={datamodule.db_img_size}",
                f"mining={datamodule.mining}",
                f"npos={datamodule.pos_num_per_query}",
                f"nneg={datamodule.negs_num_per_query}",
            )

            if hparams.use_nuscenes_boq:
                print(
                    "[NUSCENES]",
                    f"path={hparams.nuscenes_path}",
                    f"train_version={getattr(hparams, 'nuscenes_train_version', 'v1.0-trainval')}",
                    f"val_version={getattr(hparams, 'nuscenes_val_version', 'v1.0-test')}",
                    f"camnames={getattr(hparams, 'nuscenes_camnames', 'CAM_FRONT')}",
                    f"maptype={getattr(hparams, 'nuscenes_maptype', 'satellite')}",
                    f"aerial_size={getattr(hparams, 'nuscenes_aerial_size', 320)}",
                    f"aerial_zoom={getattr(hparams, 'nuscenes_aerial_zoom', 20)}",
                    f"aerial_scale={getattr(hparams, 'nuscenes_aerial_scale', 1)}",
                )

        else:
            display_datasets_stats(datamodule)

        standard_backbone = None
        if use_cross_view_boq and not (
            getattr(hparams, "shared_query_boq", False)
            or use_view_specific_adapter
            or use_dual_branch
        ):
            standard_backbone = model.image_encoder.backbone
        elif not use_cross_view_boq:
            standard_backbone = model.backbone

        if isinstance(standard_backbone, DinoV2) and standard_backbone.use_block_adapter:
            print(
                "[DINOv2][BlockAdapter]",
                f"backbone={standard_backbone.backbone_name}",
                f"freeze_backbone={standard_backbone.freeze_backbone}",
                f"unfreeze_n_blocks={standard_backbone.unfreeze_n_blocks}",
            )

            print("[DINOv2][BlockAdapter] adapter_status:")
            for line in standard_backbone.format_adapter_status_lines():
                print(f"  {line}")

            print("[DINOv2][BlockAdapter] trainable_params:")
            for line in standard_backbone.format_trainable_summary_lines():
                print(f"  {line}")

        if hparams.print_model_modules:
            module_to_print = getattr(model, "image_encoder", model)
            module_name = "image_encoder" if hasattr(model, "image_encoder") else "model"
            print(f"[Model] {module_name}:")
            print(module_to_print)

        if hparams.gpu_id is not None:
            print(
                "[Trainer]",
                f"accelerator={accelerator}",
                f"devices={devices}",
                f"gpu_id={hparams.gpu_id}",
            )

        if strategy != "auto":
            print("[Trainer]", f"strategy={strategy}")

    # ============================================================
    # 5. TensorBoard logger
    # ============================================================
    logger_name = hparams.backbone_name

    if use_cross_view_boq:
        prefix = "nuscenes" if hparams.use_nuscenes_boq else "kitti360"

        if getattr(hparams, "shared_query_boq", False):
            logger_name = (
                f"{prefix}_shared_query_"
                f"{getattr(hparams, 'query_backbone', hparams.backbone_name)}_"
                f"{getattr(hparams, 'db_backbone', hparams.backbone_name)}"
            )
        elif use_view_specific_adapter:
            logger_name = f"{prefix}_view_adapter_{hparams.backbone_name}"
        elif use_dual_branch:
            logger_name = (
                f"{prefix}_dual_"
                f"{getattr(hparams, 'query_backbone', hparams.backbone_name)}_"
                f"{getattr(hparams, 'db_backbone', hparams.backbone_name)}"
            )

    elif getattr(hparams, "use_dinov2_block_adapter", False) and "dinov2" in logger_name:
        logger_name = f"{logger_name}_block_adapter"

    tensorboard_logger = TensorBoardLogger(
        save_dir="./logs",
        name=logger_name,
        default_hp_metric=False,
    )

    tensorboard_logger.log_hyperparams(hparams.__dict__)

    # ============================================================
    # 6. Trainer
    # ============================================================
    callback_list = [checkpointing, EpochMetricsCallback()]

    if not hparams.silent and hasattr(callbacks, "RichProgressBar"):
        callback_list.append(callbacks.RichProgressBar())

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=tensorboard_logger,
        precision=precision,
        callbacks=callback_list,
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=max(1, hparams.log_every_n_steps),
        fast_dev_run=dev_mode,
        enable_model_summary=not hparams.silent,
        enable_progress_bar=not hparams.silent,
        limit_val_batches=limit_val_batches,
    )

    trainer.fit(model=model, datamodule=datamodule)

def parse_args():
    parser = argparse.ArgumentParser(description="Train BOQ or KITTI360 cross-view BOQ")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--debug_shapes", action="store_true")
    parser.add_argument("--print_model_modules", action="store_true")
    parser.add_argument("--scratch_boq", action="store_true")
    parser.add_argument("--pretrained_boq_path", type=str)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--log_every_n_steps", type=int)
    parser.add_argument("--nw", type=int)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--backbone_warmup", type=int)
    parser.add_argument("--dual_branch", action="store_true")
    parser.add_argument("--shared_query_boq", action="store_true")
    parser.add_argument("--view_specific_adapter", action="store_true")
    parser.add_argument("--dinov2_block_adapter", action="store_true")
    parser.add_argument("--query_backbone", type=str)
    parser.add_argument("--db_backbone", type=str)
    parser.add_argument("--adapter_out_channels", type=int)
    parser.add_argument("--adapter_type", type=str)
    parser.add_argument("--adapter_layers", type=int, nargs="+")
    parser.add_argument("--adapter_bottleneck_dim", type=int)
    parser.add_argument("--adapter_scale", type=float)
    parser.add_argument("--disable_adapter_zero_init", action="store_true")
    parser.add_argument("--unfreeze_n", type=int)
    parser.add_argument("--dim", type=int)

    parser.add_argument("--use_kitti360_boq", action="store_true")
    parser.add_argument("--kitti360_path", type=str)
    parser.add_argument("--kitti360_camnames", type=str)
    parser.add_argument("--kitti360_maptype", type=str)
    parser.add_argument("--kitti360_train_ratio", type=float)
    parser.add_argument("--kitti360_share_db", action="store_true")
    parser.add_argument("--kitti360_traindownsample", type=int)
    parser.add_argument("--kitti360_q_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--kitti360_db_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--kitti360_db_crop", type=int)
    parser.add_argument("--kitti360_q_jitter", type=float)
    parser.add_argument("--kitti360_db_jitter", type=float)
    parser.add_argument("--kitti360_mining", type=str)
    parser.add_argument("--kitti360_neg_samples", type=int)
    parser.add_argument("--kitti360_npos", type=int)
    parser.add_argument("--kitti360_nneg", type=int)
    parser.add_argument("--kitti360_cache_refresh", type=int)
    parser.add_argument("--kitti360_infer_bs", type=int)
    parser.add_argument("--kitti360_train_pos_th", type=float)
    parser.add_argument("--kitti360_val_pos_th", type=float)

    parser.add_argument("--satellite_map_index", type=int)
    parser.add_argument("--db_encode_chunk_size", type=int)
    parser.add_argument("--triplet_margin", type=float)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--ranking_loss", type=str)
    parser.add_argument("--ranking_tau", type=float)
    parser.add_argument("--slot_loss_weight", type=float)
    parser.add_argument("--no_normalize_descriptor", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_aggregator", action="store_true")
    parser.add_argument("--freeze_query_backbone", action="store_true")
    parser.add_argument("--freeze_db_backbone", action="store_true")
    parser.add_argument("--freeze_shared_aggregator", action="store_true")
    parser.add_argument("--freeze_ground_adapter", action="store_true")
    parser.add_argument("--freeze_sat_adapter", action="store_true")
    parser.add_argument("--print_train_items", type=int)
    parser.add_argument("--print_train_items_every_epoch", action="store_true")
    parser.add_argument("--dataset_name", type=str, choices=["kitti360", "nuscenes"])
    parser.add_argument("--use_nuscenes_boq", action="store_true")

    parser.add_argument("--nuscenes_path", type=str)
    parser.add_argument("--nuscenes_train_version", type=str)
    parser.add_argument("--nuscenes_val_version", type=str)
    parser.add_argument("--nuscenes_locations", type=str)
    parser.add_argument("--nuscenes_camnames", type=str)
    parser.add_argument("--nuscenes_maptype", type=str)
    parser.add_argument("--nuscenes_aerial_scale", type=int)
    parser.add_argument("--nuscenes_aerial_zoom", type=int)
    parser.add_argument("--nuscenes_aerial_size", type=int)

    parser.add_argument("--nuscenes_train_ratio", type=float)
    parser.add_argument("--nuscenes_share_db", action="store_true")
    parser.add_argument("--nuscenes_traindownsample", type=int)
    parser.add_argument("--nuscenes_q_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--nuscenes_db_size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--nuscenes_db_crop", type=int)
    parser.add_argument("--nuscenes_q_jitter", type=float)
    parser.add_argument("--nuscenes_db_jitter", type=float)

    parser.add_argument("--nuscenes_mining", type=str)
    parser.add_argument("--nuscenes_neg_samples", type=int)
    parser.add_argument("--nuscenes_npos", type=int)
    parser.add_argument("--nuscenes_nneg", type=int)
    parser.add_argument("--nuscenes_cache_refresh", type=int)
    parser.add_argument("--nuscenes_infer_bs", type=int)
    parser.add_argument("--nuscenes_train_pos_th", type=float)
    parser.add_argument("--nuscenes_val_pos_th", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hparams = HyperParams()

    for src, dst in [
        ("seed", "seed"),
        ("bs", "batch_size"),
        ("lr", "lr"),
        ("wd", "weight_decay"),
        ("epochs", "max_epochs"),
        ("warmup", "warmup_epochs"),
        ("log_every_n_steps", "log_every_n_steps"),
        ("nw", "num_workers"),
        ("gpu_id", "gpu_id"),
        ("pretrained_boq_path", "pretrained_boq_path"),
        ("backbone", "backbone_name"),
        ("backbone_warmup", "backbone_warmup_epochs"),
        ("query_backbone", "query_backbone"),
        ("db_backbone", "db_backbone"),
        ("adapter_out_channels", "adapter_out_channels"),
        ("adapter_type", "adapter_type"),
        ("adapter_bottleneck_dim", "adapter_bottleneck_dim"),
        ("adapter_scale", "adapter_scale"),
        ("unfreeze_n", "unfreeze_n_blocks"),
        ("dim", "output_dim"),
        ("kitti360_path", "kitti360_path"),
        ("kitti360_camnames", "kitti360_camnames"),
        ("kitti360_maptype", "kitti360_maptype"),
        ("kitti360_train_ratio", "kitti360_train_ratio"),
        ("kitti360_traindownsample", "kitti360_traindownsample"),
        ("kitti360_db_crop", "kitti360_db_crop_size"),
        ("kitti360_q_jitter", "kitti360_q_jitter"),
        ("kitti360_db_jitter", "kitti360_db_jitter"),
        ("kitti360_mining", "kitti360_mining"),
        ("kitti360_neg_samples", "kitti360_neg_samples_num"),
        ("kitti360_npos", "kitti360_pos_num_per_query"),
        ("kitti360_nneg", "kitti360_negs_num_per_query"),
        ("kitti360_cache_refresh", "kitti360_cache_refresh_rate"),
        ("kitti360_infer_bs", "kitti360_infer_batch_size"),
        ("kitti360_train_pos_th", "kitti360_train_positives_dist_threshold"),
        ("kitti360_val_pos_th", "kitti360_val_positive_dist_threshold"),
        ("satellite_map_index", "satellite_map_index"),
        ("db_encode_chunk_size", "db_encode_chunk_size"),
        ("triplet_margin", "triplet_margin"),
        ("loss_type", "loss_type"),
        ("ranking_loss", "ranking_loss_type"),
        ("ranking_tau", "ranking_tau"),
        ("slot_loss_weight", "slot_loss_weight"),
        ("print_train_items", "print_train_items"),
        ("dataset_name", "dataset_name"),
        ("nuscenes_path", "nuscenes_path"),
        ("nuscenes_train_version", "nuscenes_train_version"),
        ("nuscenes_val_version", "nuscenes_val_version"),
        ("nuscenes_locations", "nuscenes_locations"),
        ("nuscenes_camnames", "nuscenes_camnames"),
        ("nuscenes_maptype", "nuscenes_maptype"),
        ("nuscenes_aerial_scale", "nuscenes_aerial_scale"),
        ("nuscenes_aerial_zoom", "nuscenes_aerial_zoom"),
        ("nuscenes_aerial_size", "nuscenes_aerial_size"),
        ("nuscenes_train_ratio", "nuscenes_train_ratio"),
        ("nuscenes_traindownsample", "nuscenes_traindownsample"),
        ("nuscenes_db_crop", "nuscenes_db_crop_size"),
        ("nuscenes_q_jitter", "nuscenes_q_jitter"),
        ("nuscenes_db_jitter", "nuscenes_db_jitter"),
        ("nuscenes_mining", "nuscenes_mining"),
        ("nuscenes_neg_samples", "nuscenes_neg_samples_num"),
        ("nuscenes_npos", "nuscenes_pos_num_per_query"),
        ("nuscenes_nneg", "nuscenes_negs_num_per_query"),
        ("nuscenes_cache_refresh", "nuscenes_cache_refresh_rate"),
        ("nuscenes_infer_bs", "nuscenes_infer_batch_size"),
        ("nuscenes_train_pos_th", "nuscenes_train_positives_dist_threshold"),
        ("nuscenes_val_pos_th", "nuscenes_val_positive_dist_threshold"),
    ]:
        value = getattr(args, src)
        if value is not None:
            setattr(hparams, dst, value)

    if args.compile:
        hparams.compile = True
    if args.silent:
        hparams.silent = True
    if args.debug_shapes:
        hparams.debug_shapes = True
    if args.print_model_modules:
        hparams.print_model_modules = True
    if args.scratch_boq:
        hparams.use_pretrained_boq = False
    if args.dual_branch:
        hparams.dual_branch = True
    if args.shared_query_boq:
        hparams.shared_query_boq = True
    if args.view_specific_adapter:
        hparams.use_view_specific_adapter = True
    if args.dinov2_block_adapter:
        hparams.use_dinov2_block_adapter = True
    if args.use_kitti360_boq:
        hparams.use_kitti360_boq = True
    if args.kitti360_share_db:
        hparams.kitti360_share_db = True
    if args.adapter_layers is not None:
        hparams.adapter_layers = list(args.adapter_layers)
    if args.kitti360_q_size is not None:
        hparams.kitti360_query_img_size = tuple(args.kitti360_q_size)
    if args.kitti360_db_size is not None:
        hparams.kitti360_db_img_size = tuple(args.kitti360_db_size)
    if args.disable_adapter_zero_init:
        hparams.adapter_zero_init = False
    if args.no_normalize_descriptor:
        hparams.normalize_descriptor = False
    if args.freeze_backbone:
        hparams.freeze_backbone = True
    if args.freeze_aggregator:
        hparams.freeze_aggregator = True
    if args.freeze_query_backbone:
        hparams.freeze_query_backbone = True
    if args.freeze_db_backbone:
        hparams.freeze_db_backbone = True
    if args.freeze_shared_aggregator:
        hparams.freeze_shared_aggregator = True
    if args.freeze_ground_adapter:
        hparams.train_ground_adapter = False
    if args.freeze_sat_adapter:
        hparams.train_sat_adapter = False
    if args.print_train_items_every_epoch:
        hparams.print_train_items_every_epoch = True
    
    if args.dataset_name == "nuscenes" or args.use_nuscenes_boq:
        hparams.dataset_name = "nuscenes"
        hparams.use_kitti360_boq = False
        hparams.use_nuscenes_boq = True

    elif args.dataset_name == "kitti360" or args.use_kitti360_boq:
        hparams.dataset_name = "kitti360"
        hparams.use_kitti360_boq = True
        hparams.use_nuscenes_boq = False

    if args.nuscenes_share_db:
        hparams.nuscenes_share_db = True

    if args.nuscenes_q_size is not None:
        hparams.nuscenes_query_img_size = tuple(args.nuscenes_q_size)

    if args.nuscenes_db_size is not None:
        hparams.nuscenes_db_img_size = tuple(args.nuscenes_db_size)

    train(hparams, dev_mode=args.dev)


# python train.py --dataset_name nuscenes 