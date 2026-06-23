from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F

DEFAULT_RECALL_VALUES = (1, 5, 10, 20)


def _resolve_recall_values(args=None, recall_values: Sequence[int] | None = None) -> tuple[int, ...]:
    values = set(DEFAULT_RECALL_VALUES)
    if args is not None and getattr(args, "recall_values", None) is not None:
        values.update(int(k) for k in getattr(args, "recall_values"))
    if recall_values is not None:
        values.update(int(k) for k in recall_values)
    return tuple(sorted(values))


def _to_numpy(features) -> np.ndarray:
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected [N, D] features, got shape {features.shape}")
    return features


def _move_to_device(data_dict, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in data_dict.items()
    }


def compute_recall(args, queries_features, database_features, test_ds, test_method="single_query", recall_values=None):
    del test_method  # The current KITTI360 path evaluates a single descriptor per query/database image.

    recall_values = _resolve_recall_values(args, recall_values)
    queries_features = _to_numpy(queries_features)
    database_features = _to_numpy(database_features)

    recalls = {k: float("nan") for k in recall_values}
    if len(queries_features) == 0 or len(database_features) == 0:
        return recalls

    max_rank = min(max(recall_values), len(database_features))
    if max_rank <= 0:
        return recalls

    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
    faiss_index.add(database_features)
    _, predictions = faiss_index.search(queries_features, max_rank)

    positives_per_query = test_ds.get_positives()
    correct_at_k = np.zeros(len(recall_values), dtype=np.float32)

    for query_idx, pred in enumerate(predictions):
        positives = np.asarray(positives_per_query[query_idx], dtype=np.int64)
        if positives.size == 0:
            continue

        for recall_idx, k in enumerate(recall_values):
            effective_k = min(k, len(database_features))
            if effective_k <= 0:
                continue
            if np.any(np.isin(pred[:effective_k], positives)):
                correct_at_k[recall_idx:] += 1
                break

    correct_at_k = (correct_at_k / len(predictions)) * 100.0
    return {k: float(v) for k, v in zip(recall_values, correct_at_k)}


def format_recall_string(metrics: dict, gallery_size: int | None = None) -> str:
    parts = []
    for k in DEFAULT_RECALL_VALUES:
        value = metrics.get(f"recall@{k}", float("nan"))
        if np.isfinite(value):
            parts.append(f"R@{k}: {value:.2f}")
        else:
            parts.append(f"R@{k}: nan")
    if gallery_size is not None and gallery_size < max(DEFAULT_RECALL_VALUES):
        parts.append(f"(gallery={gallery_size}, clipped)")
    return ", ".join(parts)


def build_retrieval_metrics(recalls: dict[int, float], gallery_size: int | None = None) -> dict:
    metrics = {}
    for k in DEFAULT_RECALL_VALUES:
        metrics[f"recall@{k}"] = float(recalls.get(k, float("nan")))
    metrics["recall_str"] = format_recall_string(metrics, gallery_size=gallery_size)
    return metrics


def evaluate_descriptors(args, queries_features, database_features, test_ds, test_method="single_query", recall_values=None):
    recalls = compute_recall(
        args=args,
        queries_features=queries_features,
        database_features=database_features,
        test_ds=test_ds,
        test_method=test_method,
        recall_values=recall_values,
    )
    return build_retrieval_metrics(recalls, gallery_size=len(database_features))


def _extract_descriptors(model, dataloader: DataLoader, mode: str, device: str | torch.device):
    was_training = model.training
    model.eval()

    all_indices = []
    all_descriptors = []
    with torch.no_grad():
        for data_dict, indices in tqdm(dataloader, leave=False):
            data_dict = _move_to_device(data_dict, device)
            outputs = model(data_dict, mode=mode)
            descriptors = outputs["embedding"] if isinstance(outputs, dict) else outputs
            all_indices.append(indices.detach().cpu())
            all_descriptors.append(descriptors.detach().cpu())

    if was_training:
        model.train()

    if not all_descriptors:
        return torch.empty((0, 0), dtype=torch.float32)

    all_indices = torch.cat(all_indices, dim=0)
    all_descriptors = torch.cat(all_descriptors, dim=0)
    order = torch.argsort(all_indices)
    return all_descriptors[order]



def _extract_global_and_patch_features(
    model,
    dataloader: DataLoader,
    mode: str,
    device: str | torch.device,
    patch_sizes: Sequence[int],
    patch_strides: Sequence[int],
):
    """
    mode:
        'q_patch' or 'db_patch'
    """
    if mode not in {"q_patch", "db_patch"}:
        raise ValueError(f"Unsupported patch mode: {mode}")

    was_training = model.training
    model.eval()

    all_indices = []
    all_global = []
    all_patches = {int(size): [] for size in patch_sizes}
    scale_coordinates = {}

    with torch.no_grad():
        for data_dict, indices in tqdm(dataloader, leave=False):
            data_dict = _move_to_device(data_dict, device)

            outputs = model(data_dict, mode=mode)
            global_desc = outputs["embedding"]
            patch_map = outputs["patch_map"]

            patch_descs, patch_coords = _extract_patch_descriptors(
                patch_map=patch_map,
                patch_sizes=patch_sizes,
                patch_strides=patch_strides,
            )

            all_indices.append(indices.detach().cpu())
            all_global.append(global_desc.detach().float().cpu())

            for scale in patch_sizes:
                scale = int(scale)

                # float16 storage greatly reduces CPU memory.
                all_patches[scale].append(
                    patch_descs[scale].detach().half().cpu()
                )

                if scale not in scale_coordinates:
                    scale_coordinates[scale] = (
                        patch_coords[scale].detach().float().cpu()
                    )

    if was_training:
        model.train()

    if not all_global:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            {},
            {},
        )

    all_indices = torch.cat(all_indices, dim=0)
    all_global = torch.cat(all_global, dim=0)

    order = torch.argsort(all_indices)
    all_global = all_global[order]

    for scale in all_patches:
        all_patches[scale] = torch.cat(
            all_patches[scale],
            dim=0,
        )[order]

    return all_global, all_patches, scale_coordinates

def _extract_global_and_patch_features(
    model,
    dataloader: DataLoader,
    mode: str,
    device: str | torch.device,
    patch_sizes: Sequence[int],
    patch_strides: Sequence[int],
):
    """
    mode:
        'q_patch' or 'db_patch'
    """
    if mode not in {"q_patch", "db_patch"}:
        raise ValueError(f"Unsupported patch mode: {mode}")

    was_training = model.training
    model.eval()

    all_indices = []
    all_global = []
    all_patches = {int(size): [] for size in patch_sizes}
    scale_coordinates = {}

    with torch.no_grad():
        for data_dict, indices in tqdm(dataloader, leave=False):
            data_dict = _move_to_device(data_dict, device)

            outputs = model(data_dict, mode=mode)
            global_desc = outputs["embedding"]
            patch_map = outputs["patch_map"]

            patch_descs, patch_coords = _extract_patch_descriptors(
                patch_map=patch_map,
                patch_sizes=patch_sizes,
                patch_strides=patch_strides,
            )

            all_indices.append(indices.detach().cpu())
            all_global.append(global_desc.detach().float().cpu())

            for scale in patch_sizes:
                scale = int(scale)

                # float16 storage greatly reduces CPU memory.
                all_patches[scale].append(
                    patch_descs[scale].detach().half().cpu()
                )

                if scale not in scale_coordinates:
                    scale_coordinates[scale] = (
                        patch_coords[scale].detach().float().cpu()
                    )

    if was_training:
        model.train()

    if not all_global:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            {},
            {},
        )

    all_indices = torch.cat(all_indices, dim=0)
    all_global = torch.cat(all_global, dim=0)

    order = torch.argsort(all_indices)
    all_global = all_global[order]

    for scale in all_patches:
        all_patches[scale] = torch.cat(
            all_patches[scale],
            dim=0,
        )[order]

    return all_global, all_patches, scale_coordinates


def _extract_patch_descriptors(
    patch_map: torch.Tensor,
    patch_sizes: Sequence[int],
    patch_strides: Sequence[int],
):
    """
    Args:
        patch_map:
            [B, C, H, W]
        patch_sizes:
            e.g. (4, 8, 12)
        patch_strides:
            e.g. (4, 4, 4)

    Returns:
        descriptors:
            dict[size] -> [B, P, C]
        coordinates:
            dict[size] -> [P, 2], normalized x/y coordinates
    """
    if patch_map.ndim != 4:
        raise ValueError(
            f"Expected patch_map [B,C,H,W], got {tuple(patch_map.shape)}"
        )

    if len(patch_sizes) != len(patch_strides):
        raise ValueError(
            "patch_sizes and patch_strides must have equal length."
        )

    _, _, height, width = patch_map.shape

    descriptors = {}
    coordinates = {}

    for patch_size, stride in zip(patch_sizes, patch_strides):
        patch_size = int(patch_size)
        stride = int(stride)

        if patch_size <= 0 or stride <= 0:
            raise ValueError(
                f"Invalid patch_size={patch_size}, stride={stride}"
            )

        if patch_size > height or patch_size > width:
            raise ValueError(
                f"patch_size={patch_size} is larger than "
                f"feature map {(height, width)}"
            )

        # [B, C, Hout, Wout]
        pooled = F.avg_pool2d(
            patch_map.float(),
            kernel_size=patch_size,
            stride=stride,
        )

        out_h, out_w = pooled.shape[-2:]

        # [B, P, C]
        patch_desc = pooled.flatten(2).transpose(1, 2)
        patch_desc = F.normalize(patch_desc, p=2, dim=-1)

        # Patch centre coordinates, normalized to [0, 1].
        ys = (
            torch.arange(out_h, device=patch_map.device) * stride
            + (patch_size - 1) / 2.0
        ) / max(height - 1, 1)

        xs = (
            torch.arange(out_w, device=patch_map.device) * stride
            + (patch_size - 1) / 2.0
        ) / max(width - 1, 1)

        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)

        descriptors[patch_size] = patch_desc
        coordinates[patch_size] = coords

    return descriptors, coordinates

# def _build_eval_loaders(args, test_ds):
#     from src.dataloaders.datasets_ws_kitti360 import (
#         kitti360_collate_fn_cache_db,
#         kitti360_collate_fn_cache_q,
#     )

#     database_subset = Subset(test_ds, list(range(test_ds.database_num)))
#     query_subset = Subset(test_ds, list(range(test_ds.database_num, len(test_ds))))

#     common_loader_kwargs = dict(
#         num_workers=getattr(args, "num_workers", 0),
#         batch_size=getattr(args, "infer_batch_size", 32),
#         shuffle=False,
#         pin_memory=str(getattr(args, "device", "cpu")).startswith("cuda"),
#     )

#     database_loader = DataLoader(
#         database_subset,
#         collate_fn=kitti360_collate_fn_cache_db,
#         **common_loader_kwargs,
#     )
#     query_loader = DataLoader(
#         query_subset,
#         collate_fn=kitti360_collate_fn_cache_q,
#         **common_loader_kwargs,
#     )
#     return database_loader, query_loader

def _build_eval_loaders(args, test_ds):
    dataset_name = str(
        getattr(
            args,
            "dataset_name",
            getattr(test_ds, "dataset_name", "kitti360"),
        )
    ).lower()

    if dataset_name == "nuscenes":
        from src.dataloaders.datasets_ws_nuscenes import (
            nuscenes_collate_fn_cache_db as collate_fn_cache_db,
            nuscenes_collate_fn_cache_q as collate_fn_cache_q,
        )
    else:
        from src.dataloaders.datasets_ws_kitti360 import (
            kitti360_collate_fn_cache_db as collate_fn_cache_db,
            kitti360_collate_fn_cache_q as collate_fn_cache_q,
        )

    database_subset = Subset(
        test_ds,
        list(range(test_ds.database_num)),
    )
    query_subset = Subset(
        test_ds,
        list(range(test_ds.database_num, len(test_ds))),
    )

    common_loader_kwargs = dict(
        num_workers=getattr(args, "num_workers", 0),
        batch_size=getattr(args, "infer_batch_size", 32),
        shuffle=False,
        pin_memory=str(
            getattr(args, "device", "cpu")
        ).startswith("cuda"),
    )

    database_loader = DataLoader(
        database_subset,
        collate_fn=collate_fn_cache_db,
        **common_loader_kwargs,
    )

    query_loader = DataLoader(
        query_subset,
        collate_fn=collate_fn_cache_q,
        **common_loader_kwargs,
    )

    return database_loader, query_loader


def _rerank_with_multiscale_patches(
    predictions: np.ndarray,
    global_scores: np.ndarray,
    query_patches: dict[int, torch.Tensor],
    database_patches: dict[int, torch.Tensor],
    query_coordinates: dict[int, torch.Tensor],
    database_coordinates: dict[int, torch.Tensor],
    patch_weights: dict[int, float],
    device: str | torch.device,
    global_weight: float = 0.4,
    spatial_weight: float = 0.0,
    spatial_tau: float = 0.15,
    query_batch_size: int = 8,
):
    """
    predictions:
        Global retrieval candidates [Nq, K].

    global_scores:
        Global cosine scores [Nq, K].
    """
    num_queries, top_k = predictions.shape
    reranked_predictions = np.empty_like(predictions)

    valid_scales = [
        scale
        for scale in patch_weights
        if scale in query_patches and scale in database_patches
    ]

    if not valid_scales:
        raise ValueError("No valid patch scales available for reranking.")

    weight_sum = sum(float(patch_weights[s]) for s in valid_scales)
    normalized_weights = {
        s: float(patch_weights[s]) / weight_sum
        for s in valid_scales
    }

    for start in tqdm(
        range(0, num_queries, query_batch_size),
        desc="Patch reranking",
    ):
        end = min(start + query_batch_size, num_queries)

        candidate_indices_cpu = torch.from_numpy(
            predictions[start:end]
        ).long()

        batch_patch_score = torch.zeros(
            (end - start, top_k),
            dtype=torch.float32,
            device=device,
        )

        for scale in valid_scales:
            q_patch = query_patches[scale][start:end]
            q_patch = q_patch.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            # [B,K,P,C]
            db_patch = database_patches[scale][
                candidate_indices_cpu
            ]
            db_patch = db_patch.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            q_coords = query_coordinates[scale].to(
                device=device,
                dtype=torch.float32,
            )
            db_coords = database_coordinates[scale].to(
                device=device,
                dtype=torch.float32,
            )

            scale_score = _mutual_nn_patch_score(
                query_patches=q_patch,
                database_patches=db_patch,
                query_coords=q_coords,
                database_coords=db_coords,
                spatial_weight=spatial_weight,
                spatial_tau=spatial_tau,
            )

            batch_patch_score += (
                normalized_weights[scale] * scale_score
            )

        batch_global_score = torch.from_numpy(
            global_scores[start:end]
        ).to(device=device, dtype=torch.float32)

        final_score = (
            global_weight * batch_global_score
            + (1.0 - global_weight) * batch_patch_score
        )

        rerank_order = torch.argsort(
            final_score,
            dim=1,
            descending=True,
        ).cpu()

        reranked_batch = torch.gather(
            candidate_indices_cpu,
            dim=1,
            index=rerank_order,
        )

        reranked_predictions[start:end] = reranked_batch.numpy()

    return reranked_predictions


def _mutual_nn_patch_score(
    query_patches: torch.Tensor,
    database_patches: torch.Tensor,
    query_coords: torch.Tensor,
    database_coords: torch.Tensor,
    spatial_weight: float = 0.0,
    spatial_tau: float = 0.15,
):
    """
    Args:
        query_patches:
            [B, Pq, C]
        database_patches:
            [B, K, Pd, C]

    Returns:
        score:
            [B, K], higher is better.
    """
    # Cosine similarity because patch descriptors are L2-normalized.
    similarity = torch.einsum(
        "bpc,bkqc->bkpq",
        query_patches,
        database_patches,
    )

    # For each query patch, its best database patch.
    q_to_db = similarity.argmax(dim=-1)       # [B,K,Pq]

    # For each database patch, its best query patch.
    db_to_q = similarity.argmax(dim=-2)       # [B,K,Pd]

    # Check whether the reverse nearest neighbour points back
    # to the original query patch.
    reverse_query_index = torch.gather(
        db_to_q,
        dim=2,
        index=q_to_db,
    )                                         # [B,K,Pq]

    num_query_patches = query_patches.shape[1]

    query_patch_index = torch.arange(
        num_query_patches,
        device=query_patches.device,
    ).view(1, 1, -1)

    mutual_mask = reverse_query_index.eq(query_patch_index)

    matched_similarity = torch.gather(
        similarity,
        dim=-1,
        index=q_to_db.unsqueeze(-1),
    ).squeeze(-1)                             # [B,K,Pq]

    mutual_count = mutual_mask.sum(dim=-1)    # [B,K]
    denominator = mutual_count.clamp_min(1).float()

    appearance_score = (
        matched_similarity * mutual_mask.float()
    ).sum(dim=-1) / denominator

    # Avoid a candidate receiving a high score from only one patch.
    coverage = mutual_count.float() / max(num_query_patches, 1)
    appearance_score = appearance_score * torch.sqrt(
        coverage.clamp_min(1e-6)
    )

    appearance_score = torch.where(
        mutual_count > 0,
        appearance_score,
        torch.full_like(appearance_score, -1.0),
    )

    if spatial_weight <= 0:
        return appearance_score

    # Optional rapid spatial consistency.
    # Ground-satellite matching should normally keep this weight low.
    matched_db_coords = database_coords[q_to_db]
    query_coords_expanded = query_coords.view(
        1, 1, num_query_patches, 2
    )

    displacement = matched_db_coords - query_coords_expanded
    mask_float = mutual_mask.float().unsqueeze(-1)

    mean_displacement = (
        displacement * mask_float
    ).sum(dim=2) / denominator.unsqueeze(-1)

    residual = torch.linalg.vector_norm(
        displacement - mean_displacement.unsqueeze(2),
        dim=-1,
    )

    spatial_similarity = torch.exp(
        -residual / max(float(spatial_tau), 1e-6)
    )

    spatial_score = (
        spatial_similarity * mutual_mask.float()
    ).sum(dim=-1) / denominator

    # Convert [0,1] to approximately [-1,1].
    spatial_score = 2.0 * spatial_score - 1.0

    return (
        (1.0 - spatial_weight) * appearance_score
        + spatial_weight * spatial_score
    )


def _compute_recalls_from_predictions(
    predictions: np.ndarray,
    test_ds,
    recall_values: Sequence[int],
):
    positives_per_query = test_ds.get_positives()
    correct_at_k = np.zeros(
        len(recall_values),
        dtype=np.float32,
    )

    for query_idx, prediction in enumerate(predictions):
        positives = np.asarray(
            positives_per_query[query_idx],
            dtype=np.int64,
        )

        if positives.size == 0:
            continue

        for recall_idx, k in enumerate(recall_values):
            effective_k = min(k, prediction.shape[0])

            if effective_k <= 0:
                continue

            if np.any(
                np.isin(
                    prediction[:effective_k],
                    positives,
                )
            ):
                correct_at_k[recall_idx:] += 1
                break

    correct_at_k = (
        correct_at_k / len(predictions)
    ) * 100.0

    return {
        k: float(value)
        for k, value in zip(recall_values, correct_at_k)
    }



def test(
    args,
    test_ds,
    model,
    test_method="single_query",
    pca=None,
    modelq=None,
    database_loader=None,
    query_loader=None,
):
    del pca

    if modelq is None:
        modelq = model

    if database_loader is None or query_loader is None:
        database_loader, query_loader = _build_eval_loaders(
            args,
            test_ds,
        )

    device = getattr(
        args,
        "device",
        getattr(model, "device", "cpu"),
    )

    use_patch_rerank = bool(
        getattr(args, "use_patch_rerank", False)
    )

    # ------------------------------------------------------------
    # Original global BoQ evaluation
    # ------------------------------------------------------------
    if not use_patch_rerank:
        database_features = _extract_descriptors(
            model,
            database_loader,
            mode="db",
            device=device,
        )

        query_features = _extract_descriptors(
            modelq,
            query_loader,
            mode="q",
            device=device,
        )

        metrics = evaluate_descriptors(
            args=args,
            queries_features=query_features,
            database_features=database_features,
            test_ds=test_ds,
            test_method=test_method,
        )

        recalls = {
            int(key.split("@", maxsplit=1)[1]): value
            for key, value in metrics.items()
            if key.startswith("recall@")
        }

        return recalls, metrics["recall_str"], metrics

    # ------------------------------------------------------------
    # Multi-scale Patch-BoQ reranking
    # ------------------------------------------------------------
    patch_sizes = tuple(
        getattr(args, "patch_sizes", (4, 8, 12))
    )
    patch_strides = tuple(
        getattr(args, "patch_strides", (4, 4, 4))
    )
    patch_weight_values = tuple(
        getattr(args, "patch_weights", (0.40, 0.35, 0.25))
    )

    if not (
        len(patch_sizes)
        == len(patch_strides)
        == len(patch_weight_values)
    ):
        raise ValueError(
            "patch_sizes, patch_strides and patch_weights "
            "must have equal length."
        )

    patch_weights = {
        int(size): float(weight)
        for size, weight in zip(
            patch_sizes,
            patch_weight_values,
        )
    }

    database_global, database_patches, database_coords = (
        _extract_global_and_patch_features(
            model=model,
            dataloader=database_loader,
            mode="db_patch",
            device=device,
            patch_sizes=patch_sizes,
            patch_strides=patch_strides,
        )
    )

    query_global, query_patches, query_coords = (
        _extract_global_and_patch_features(
            model=modelq,
            dataloader=query_loader,
            mode="q_patch",
            device=device,
            patch_sizes=patch_sizes,
            patch_strides=patch_strides,
        )
    )

    recall_values = _resolve_recall_values(args)

    rerank_top_k = int(
        getattr(args, "patch_rerank_topk", 50)
    )
    rerank_top_k = max(
        rerank_top_k,
        max(recall_values),
    )
    rerank_top_k = min(
        rerank_top_k,
        len(database_global),
    )

    query_numpy = _to_numpy(query_global)
    database_numpy = _to_numpy(database_global)

    # Global descriptors are L2-normalized, so use inner product
    # as cosine similarity.
    index = faiss.IndexFlatIP(database_numpy.shape[1])
    index.add(database_numpy)

    global_scores, predictions = index.search(
        query_numpy,
        rerank_top_k,
    )

    predictions = _rerank_with_multiscale_patches(
        predictions=predictions,
        global_scores=global_scores,
        query_patches=query_patches,
        database_patches=database_patches,
        query_coordinates=query_coords,
        database_coordinates=database_coords,
        patch_weights=patch_weights,
        device=device,
        global_weight=float(
            getattr(args, "patch_global_weight", 0.4)
        ),
        spatial_weight=float(
            getattr(args, "patch_spatial_weight", 0.0)
        ),
        spatial_tau=float(
            getattr(args, "patch_spatial_tau", 0.15)
        ),
        query_batch_size=int(
            getattr(args, "patch_score_batch_size", 8)
        ),
    )

    recalls = _compute_recalls_from_predictions(
        predictions=predictions,
        test_ds=test_ds,
        recall_values=recall_values,
    )

    metrics = build_retrieval_metrics(
        recalls,
        gallery_size=len(database_global),
    )

    return recalls, metrics["recall_str"], metrics
