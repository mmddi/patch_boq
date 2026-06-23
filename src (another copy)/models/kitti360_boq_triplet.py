import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
try:
    import lightning as L
except ModuleNotFoundError:
    import pytorch_lightning as L


from .dino_boq_view_adapter import DinoBoQViewSpecificAdapterEncoder
from src.evaluate import evaluate_descriptors


class Kitti360BoQTripletModel(L.LightningModule):
    def __init__(
        self,
        backbone=None,
        aggregator=None,
        image_encoder=None,
        lr=1e-4,
        lr_mul=0.1,
        weight_decay=1e-3,
        warmup_epochs=10,
        backbone_warmup_epochs=0,
        milestones=[10, 20],
        satellite_map_index=0,
        triplet_margin=0.2,
        normalize_descriptor=True,
        loss_type="triplet",
        freeze_backbone=False,
        freeze_aggregator=False,
        silent=False,
        debug_shapes=False,
        print_train_items=0,
        print_train_items_every_epoch=False,
        db_encode_chunk_size=8,
    ):
        super().__init__()
        if image_encoder is None:
            raise ValueError("image_encoder must be provided.")

        self.image_encoder = image_encoder

        self.lr = lr
        self.lr_mul = lr_mul
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.backbone_warmup_epochs = int(backbone_warmup_epochs)
        if self.backbone_warmup_epochs < 0:
            raise ValueError("backbone_warmup_epochs must be >= 0")
        self.milestones = milestones
        self.satellite_map_index = satellite_map_index
        self.triplet_margin = triplet_margin
        self.loss_type = loss_type
        self.silent = silent
        self.debug_shapes = debug_shapes
        self.print_train_items = int(print_train_items)
        self.print_train_items_every_epoch = print_train_items_every_epoch
        self.hparams.db_encode_chunk_size = (
            None if db_encode_chunk_size is None else int(db_encode_chunk_size)
        )
        self._train_shapes_printed = False
        self._train_items_printed = False
        self.best_recall1 = float("-inf")
        self.last_eval_metrics = {}

        if self.loss_type != "triplet":
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

    @property
    def descriptor_dim(self):
        return self.image_encoder.output_dim

    
    @property
    def is_view_specific_adapter_encoder(self):
        return isinstance(self.image_encoder, DinoBoQViewSpecificAdapterEncoder)

    @staticmethod
    def _append_param_group(optimizer_params, module, lr, weight_decay, seen_param_ids):
        if module is None:
            return

        params = []
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            params.append(param)

        if params:
            optimizer_params.append({"params": params, "lr": lr, "weight_decay": weight_decay})

    def configure_optimizers(self):
        optimizer_params = []
        seen_param_ids = set()
        if self.is_view_specific_adapter_encoder:
            self._append_param_group(
                optimizer_params,
                self.image_encoder.backbone,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            for adapter in self.image_encoder.ground_adapters:
                self._append_param_group(
                    optimizer_params,
                    adapter,
                    self.lr,
                    self.weight_decay,
                    seen_param_ids,
                )
            for adapter in self.image_encoder.satellite_adapters:
                self._append_param_group(
                    optimizer_params,
                    adapter,
                    self.lr,
                    self.weight_decay,
                    seen_param_ids,
                )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.aggregator,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
        elif self.is_dual_branch:
            self._append_param_group(
                optimizer_params,
                self.image_encoder.q_backbone,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.db_backbone,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.q_adapter,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.db_adapter,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.shared_aggregator,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
        else:
            self._append_param_group(
                optimizer_params,
                self.image_encoder.backbone,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )
            self._append_param_group(
                optimizer_params,
                self.image_encoder.aggregator,
                self.lr,
                self.weight_decay,
                seen_param_ids,
            )

        if not optimizer_params:
            raise ValueError("No trainable parameters found for optimizer setup.")

        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.milestones,
            gamma=self.lr_mul,
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = min(1.0, (self.trainer.global_step + 1) / total_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg.get("initial_lr", self.lr)
        optimizer.step(closure=optimizer_closure)
        self.log("_LR", optimizer.param_groups[-1]["lr"], on_step=True, on_epoch=False, prog_bar=False, logger=True)

    def _select_satellite_maps(self, db_map: torch.Tensor):
        if db_map.ndim == 6:
            positive = db_map[:, 0, self.satellite_map_index]  # [B, 3, H, W]
            negatives = db_map[:, 1:, self.satellite_map_index]  # [B, N, 3, H, W]
            return positive, negatives
        if db_map.ndim == 5:
            return db_map[:, self.satellite_map_index], None  # [B, 3, H, W]
        raise ValueError(f"Unexpected db_map shape: {tuple(db_map.shape)}")

    def _select_training_satellite_maps(self, output_dict):
        if "positive_db_map" in output_dict and "negative_db_map" in output_dict:
            positive = output_dict["positive_db_map"][:, :, self.satellite_map_index]  # [B, P, 3, H, W]
            negatives = output_dict["negative_db_map"][:, :, self.satellite_map_index]  # [B, N, 3, H, W]
            return positive, negatives

        db = output_dict["db_map"]  # [B, 1+nneg, nmap, 3, H, W]
        positive, negatives = self._select_satellite_maps(db)
        if positive.ndim == 4:
            positive = positive.unsqueeze(1)  # [B, 1, 3, H, W]
        return positive, negatives

    def _should_freeze_backbone_during_warmup(self):
        return self.backbone_warmup_epochs > 0 and self.current_epoch < self.backbone_warmup_epochs

    @staticmethod
    def _forward_backbone(backbone, images, freeze_backbone=False):
        if getattr(backbone, "supports_freeze_backbone_forward", False):
            return backbone(images, freeze_backbone=freeze_backbone)

        if not freeze_backbone:
            return backbone(images)

        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            features = backbone(images)
        if was_training:
            backbone.train()
        return features


    def _encode_view_specific_adapter_branch(self, images, view_type, return_debug=False):
        freeze_backbone = self._should_freeze_backbone_during_warmup()
        return self.image_encoder.encode_view(
            images,
            view_type=view_type,
            freeze_backbone=freeze_backbone,
            return_debug=return_debug,
        )

    def _encode_query(self, images, return_debug=False):
        return self._encode_view_specific_adapter_branch(
            images,
            view_type="ground",
            return_debug=return_debug,
        )
        

    def _encode_db(self, images, return_debug=False):
        return self._encode_view_specific_adapter_branch(
            images,
            view_type="satellite",
            return_debug=return_debug,
        )
    
    def _encode_db_in_chunks(self, images, chunk_size):
        num_images = images.size(0)
        if chunk_size is None or chunk_size <= 0 or chunk_size >= num_images:
            return self._encode_db(images)

        descriptors = []
        for start_idx in range(0, num_images, chunk_size):
            end_idx = min(start_idx + chunk_size, num_images)
            descriptors.append(self._encode_db(images[start_idx:end_idx]))
        return torch.cat(descriptors, dim=0)

   
    def forward(self, batch_or_images, mode=None):
        # 保留原来的 Tensor 输入行为。
        if isinstance(batch_or_images, torch.Tensor):
            return self._encode_query(batch_or_images)

        # ------------------------------------------------------------
        # 原有全局描述子路径
        # ------------------------------------------------------------
        if mode == "q":
            return {
                "embedding": self._encode_query(
                    batch_or_images["query_image"]
                )
            }

        if mode == "db":
            satellite, _ = self._select_satellite_maps(
                batch_or_images["db_map"]
            )
            return {
                "embedding": self._encode_db(satellite)
            }

        # ------------------------------------------------------------
        # 新增：测试阶段 global + patch map
        # ------------------------------------------------------------
        if mode == "q_patch":
            if not self.is_view_specific_adapter_encoder:
                raise RuntimeError(
                    "q_patch currently requires "
                    "DinoBoQViewSpecificAdapterEncoder."
                )

            return self.image_encoder.encode_view_with_patch_map(
                batch_or_images["query_image"],
                view_type="ground",
                freeze_backbone=False,
            )

        if mode == "db_patch":
            if not self.is_view_specific_adapter_encoder:
                raise RuntimeError(
                    "db_patch currently requires "
                    "DinoBoQViewSpecificAdapterEncoder."
                )

            satellite, _ = self._select_satellite_maps(
                batch_or_images["db_map"]
            )

            return self.image_encoder.encode_view_with_patch_map(
                satellite,
                view_type="satellite",
                freeze_backbone=False,
            )

        raise ValueError(f"Unsupported forward mode: {mode!r}")



    def forward_triplet_batch(self, output_dict, return_debug=False):
        q = output_dict["query_image"]  # [B, 3, H, W]
        positive, negatives = self._select_training_satellite_maps(output_dict)
        if negatives is None or negatives.size(1) == 0:
            raise ValueError("Expected negative satellite maps in the training batch.")

        batch_size, num_positives = positive.shape[:2]
        _, num_negatives = negatives.shape[:2]
        pos_flat = positive.flatten(0, 1)
        neg_flat = negatives.flatten(0, 1)
        if return_debug:
            q_desc, q_debug = self._encode_query(q, return_debug=True)  # [B, D]
            p_desc = self._encode_db_in_chunks(pos_flat, self.hparams.db_encode_chunk_size)
            debug_chunk = self.hparams.db_encode_chunk_size
            if debug_chunk is None or debug_chunk <= 0 or debug_chunk >= pos_flat.size(0):
                _, db_debug = self._encode_db(pos_flat, return_debug=True)
            else:
                _, db_debug = self._encode_db(pos_flat[:debug_chunk], return_debug=True)
            p_desc = p_desc.view(batch_size, num_positives, -1)  # [B, P, D]
        else:
            q_desc = self._encode_query(q)  # [B, D]
            p_desc = self._encode_db_in_chunks(
                pos_flat,
                self.hparams.db_encode_chunk_size,
            ).view(batch_size, num_positives, -1)  # [B, P, D]

        n_desc = self._encode_db_in_chunks(
            neg_flat,
            self.hparams.db_encode_chunk_size,
        ).view(batch_size, num_negatives, -1)  # [B, N, D]
        if return_debug:
            return q_desc, p_desc, n_desc, {"query_branch": q_debug, "db_branch": db_debug}
        return q_desc, p_desc, n_desc

    def compute_triplet_loss(self, q_desc, p_desc, n_desc):
        if p_desc.ndim == 3:
            sim_pos = torch.einsum("bd,bpd->bp", q_desc, p_desc)  # [B, P]
            pos_score = sim_pos.mean(dim=1)  # [B]
        else:
            sim_pos = (q_desc * p_desc).sum(dim=-1)  # [B]
            pos_score = sim_pos
        sim_neg = torch.einsum("bd,bnd->bn", q_desc, n_desc)  # [B, N]
        hardest_neg = sim_neg.max(dim=1).values
        loss = F.relu(self.triplet_margin + hardest_neg - pos_score).mean()
        return loss, pos_score, hardest_neg

    def training_step(self, batch, batch_idx):
        output_dict, _, triplets_global_indexes = batch

        if self._should_print_train_items(batch_idx, triplets_global_indexes):
            self._print_train_items(output_dict, triplets_global_indexes)

        if self.debug_shapes and not self._train_shapes_printed:
            q_desc, p_desc, n_desc, debug_shapes = self.forward_triplet_batch(output_dict, return_debug=True)
        else:
            q_desc, p_desc, n_desc = self.forward_triplet_batch(output_dict)
        loss, sim_pos, hardest_neg = self.compute_triplet_loss(q_desc, p_desc, n_desc)

        if self.debug_shapes and not self._train_shapes_printed:
            print(
                "[ShapeDebug][KITTI360]",
                f"query_image={tuple(output_dict['query_image'].shape)}",
                f"positive_db_map={tuple(output_dict['positive_db_map'].shape)}",
                f"negative_db_map={tuple(output_dict['negative_db_map'].shape)}",
                f"q_backbone_out={debug_shapes['query_branch']['backbone_out']}",
                f"q_after_adapter={debug_shapes['query_branch']['after_adapter']}",
                f"q_desc={debug_shapes['query_branch']['descriptor']}",
                f"db_backbone_out={debug_shapes['db_branch']['backbone_out']}",
                f"db_after_adapter={debug_shapes['db_branch']['after_adapter']}",
                f"db_desc={debug_shapes['db_branch']['descriptor']}",
                f"p_desc={tuple(p_desc.shape)}",
                f"n_desc={tuple(n_desc.shape)}",
            )
            self._train_shapes_printed = True

        batch_size = q_desc.size(0)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "loss_total",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "pos_sim_mean",
            sim_pos.mean(),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "hardest_neg_sim_mean",
            hardest_neg.mean(),
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_start(self):
        self._train_items_printed = False

    def on_validation_epoch_start(self):
        self._val_db_outputs = []
        self._val_q_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict, indexes = batch
        if dataloader_idx == 0:
            outputs = self.forward(data_dict, mode="db")
            self._val_db_outputs.append((indexes.detach().cpu(), outputs["embedding"].detach().cpu()))
        else:
            outputs = self.forward(data_dict, mode="q")
            self._val_q_outputs.append((indexes.detach().cpu(), outputs["embedding"].detach().cpu()))

    def on_validation_epoch_end(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        val_dataset = getattr(datamodule, "val_dataset", None)
        if val_dataset is None:
            self._clear_validation_outputs()
            return

        database_indices, database_descriptors = self._gather_validation_outputs(self._val_db_outputs)
        query_indices, query_descriptors = self._gather_validation_outputs(self._val_q_outputs)
        if len(database_descriptors) == 0 or len(query_descriptors) == 0:
            metrics = {
                "recall@1": float("nan"),
                "recall@5": float("nan"),
                "recall@10": float("nan"),
                "recall@20": float("nan"),
                "recall_str": "R@1: nan, R@5: nan, R@10: nan, R@20: nan",
            }
        else:
            if len(database_indices) != val_dataset.database_num:
                raise RuntimeError(
                    f"Validation database descriptor count mismatch: {len(database_indices)} vs {val_dataset.database_num}"
                )
            if len(query_indices) != val_dataset.queries_num:
                raise RuntimeError(
                    f"Validation query descriptor count mismatch: {len(query_indices)} vs {val_dataset.queries_num}"
                )

            metrics = evaluate_descriptors(
                args=getattr(datamodule, "dataset_args", None),
                queries_features=query_descriptors,
                database_features=database_descriptors,
                test_ds=val_dataset,
                test_method=getattr(val_dataset, "test_method", "single_query"),
            )

        self.last_eval_metrics = metrics
        log_metrics = {
            "val/recall@1": metrics["recall@1"],
            "val/recall@5": metrics["recall@5"],
            "val/recall@10": metrics["recall@10"],
            "val/recall@20": metrics["recall@20"],
            "val_recall1": metrics["recall@1"],
            "val_recall5": metrics["recall@5"],
        }
        for metric_name, metric_value in log_metrics.items():
            if math.isfinite(metric_value):
                self.log(
                    metric_name,
                    metric_value,
                    prog_bar=metric_name == "val/recall@1",
                    logger=True,
                    sync_dist=True,
                    batch_size=1,
                )

        if not self.silent and self.trainer.is_global_zero:
            print(f"[KITTI360][Eval] {metrics['recall_str']}")

        self._clear_validation_outputs()

    def on_train_epoch_end(self):
        if self.current_epoch + 1 < self.trainer.max_epochs:
            self._refresh_triplets()

    def _refresh_triplets(self):
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None or not hasattr(datamodule, "refresh_train_triplets"):
            return
        was_training = self.training
        datamodule.refresh_train_triplets(self, self)
        if was_training:
            self.train()
        if not self.silent and self.trainer.is_global_zero:
            print(f"[KITTI360] Refreshed triplets with descriptor_dim={self.descriptor_dim}")

    def _gather_validation_outputs(self, outputs):
        if not outputs:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty((0, self.descriptor_dim), dtype=torch.float32),
            )

        indices = torch.cat([item[0] for item in outputs], dim=0).to(torch.long)
        descriptors = torch.cat([item[1] for item in outputs], dim=0).to(torch.float32)

        if dist.is_available() and dist.is_initialized():
            gathered_outputs = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                gathered_outputs,
                {"indices": indices, "descriptors": descriptors},
            )
            indices = torch.cat([item["indices"] for item in gathered_outputs], dim=0)
            descriptors = torch.cat([item["descriptors"] for item in gathered_outputs], dim=0)

        order = torch.argsort(indices)
        indices = indices[order]
        descriptors = descriptors[order]

        if len(indices) > 1:
            unique_mask = torch.ones_like(indices, dtype=torch.bool)
            unique_mask[1:] = indices[1:] != indices[:-1]
            indices = indices[unique_mask]
            descriptors = descriptors[unique_mask]
        return indices, descriptors

    def _clear_validation_outputs(self):
        self._val_db_outputs = []
        self._val_q_outputs = []

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_recall1"] = float(self.best_recall1)
        checkpoint["last_eval_metrics"] = dict(self.last_eval_metrics)

    def _should_print_train_items(self, batch_idx, triplets_global_indexes):
        if self.print_train_items <= 0:
            return False
        if batch_idx != 0 or self._train_items_printed:
            return False
        if self.trainer is not None and hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return False
        if self.current_epoch > 0 and not self.print_train_items_every_epoch:
            return False
        if not torch.is_tensor(triplets_global_indexes) or triplets_global_indexes.numel() == 0:
            return False
        return True

    def _print_train_items(self, output_dict, triplets_global_indexes):
        datamodule = getattr(self.trainer, "datamodule", None)
        train_dataset = getattr(datamodule, "train_dataset", None)
        samples_to_print = min(self.print_train_items, triplets_global_indexes.shape[0])

        print(
            f"[KITTI360][TrainItems] epoch={self.current_epoch + 1} "
            f"showing {samples_to_print} sample(s) from the first training batch"
        )
        print(
            f"  batch query_image shape={tuple(output_dict['query_image'].shape)} "
            f"positive_db_map shape={tuple(output_dict['positive_db_map'].shape)} "
            f"negative_db_map shape={tuple(output_dict['negative_db_map'].shape)}"
        )

        for sample_idx in range(samples_to_print):
            triplet = triplets_global_indexes[sample_idx]
            query_index = int(triplet[0].item())
            datamodule_pos_count = getattr(train_dataset, "pos_num_per_query", getattr(datamodule, "pos_num_per_query", 1))
            positive_indexes = [int(idx) for idx in triplet[1 : 1 + datamodule_pos_count].tolist()]
            negative_indexes = [int(idx) for idx in triplet[1 + datamodule_pos_count :].tolist()]

            print(f"  sample[{sample_idx}] query_index={query_index} positive_indexes={positive_indexes}")
            print(f"    negative_indexes={negative_indexes}")

            if train_dataset is not None and hasattr(train_dataset, "queries_infos") and hasattr(train_dataset, "database_infos"):
                query_info = train_dataset.queries_infos[query_index]
                positive_infos = [train_dataset.database_infos[idx] for idx in positive_indexes]
                negative_paths = [train_dataset.database_infos[idx]["db_satellite_path"] for idx in negative_indexes]

                print(f"    query_path={query_info['qimage00path']}")
                print(f"    query_eastnorth=({query_info['east']:.2f}, {query_info['north']:.2f})")
                print(f"    positive_satellite_paths={[info['db_satellite_path'] for info in positive_infos]}")
                print(
                    "    positive_eastnorths="
                    f"{[(round(info['east'], 2), round(info['north'], 2)) for info in positive_infos]}"
                )
                print(f"    negative_satellite_paths={negative_paths}")
            else:
                print("    train_dataset metadata unavailable; only indexes are shown")

        self._train_items_printed = True
