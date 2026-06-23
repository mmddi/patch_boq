import logging
import os
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as TVT
import utm
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


trainselectlocationlist = [
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]

testselectlocationlist = [
    "2013_05_28_drive_0000_sync",
    # "2013_05_28_drive_0002_sync",
    "2013_05_28_drive_0003_sync",
    "2013_05_28_drive_0004_sync",
    "2013_05_28_drive_0005_sync",
    "2013_05_28_drive_0006_sync",
    "2013_05_28_drive_0007_sync",
    # "2013_05_28_drive_0009_sync",
    "2013_05_28_drive_0010_sync",
]


def _default_kitti360_options():
    return SimpleNamespace(
        train_ratio=0.5,
        share_db=False,
        dataroot="",
        q_resize=(224, 224),
        q_jitter=0.0,
        db_cropsize=320,
        db_resize=(224, 224),
        db_jitter=0.0,
        camnames="00",
        maptype="satellite",
        traindownsample=1,
        num_workers=0,
        train_positives_dist_threshold=10.0,
        val_positive_dist_threshold=25.0,
    )


opt = _default_kitti360_options()
train_ratio = opt.train_ratio
share_db = opt.share_db


def configure_kitti360_options(**kwargs):
    global opt, train_ratio, share_db
    for key, value in kwargs.items():
        if value is not None:
            setattr(opt, key, value)
    train_ratio = opt.train_ratio
    share_db = opt.share_db
    return opt


def kitti360_collate_fn(batch):
    query_image = torch.stack([e[0]["query_image"] for e in batch])  # [B, 3, H, W]
    query_eastnorth = torch.stack([e[0]["query_eastnorth"] for e in batch]).float()  # [B, 2]
    positive_db_map = torch.stack([e[0]["positive_db_map"] for e in batch])  # [B, npos, nmap, 3, H, W]
    negative_db_map = torch.stack([e[0]["negative_db_map"] for e in batch])  # [B, nneg, nmap, 3, H, W]
    positive_db_eastnorth = torch.stack([e[0]["positive_db_eastnorth"] for e in batch]).float()  # [B, npos, 2]
    negative_db_eastnorth = torch.stack([e[0]["negative_db_eastnorth"] for e in batch]).float()  # [B, nneg, 2]

    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for batch_idx, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * batch_idx

    output_dict = {
        "query_image": query_image,
        "query_eastnorth": query_eastnorth,
        "positive_db_map": positive_db_map,
        "negative_db_map": negative_db_map,
        "positive_db_eastnorth": positive_db_eastnorth,
        "negative_db_eastnorth": negative_db_eastnorth,
    }
    return output_dict, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


def kitti360_collate_fn_cache_db(batch):
    db_map = torch.stack([e[0]["db_map"] for e in batch])  # [B, nmap, 3, H, W]
    indices = torch.tensor([e[1] for e in batch], dtype=torch.long)
    return {"db_map": db_map}, indices


def kitti360_collate_fn_cache_q(batch):
    query_image = torch.stack([e[0]["query_image"] for e in batch])  # [B, 3, H, W]
    query_eastnorth = torch.stack([e[0]["query_eastnorth"] for e in batch]).float()
    indices = torch.tensor([e[1] for e in batch], dtype=torch.long)
    return {"query_image": query_image, "query_eastnorth": query_eastnorth}, indices


def _query_transform(split):
    ops = [TVT.Resize(opt.q_resize)]
    if split == "train" and opt.q_jitter > 0:
        ops.append(
            TVT.ColorJitter(
                brightness=opt.q_jitter,
                contrast=opt.q_jitter,
                saturation=opt.q_jitter,
                hue=min(0.5, opt.q_jitter),
            )
        )
    ops.extend([TVT.ToTensor(), TVT.Normalize(mean=0.5, std=0.22)])
    return TVT.Compose(ops)


def _db_transform(split):
    ops = [TVT.CenterCrop(opt.db_cropsize), TVT.Resize(opt.db_resize)]
    if split == "train" and opt.db_jitter > 0:
        ops.append(
            TVT.ColorJitter(
                brightness=opt.db_jitter,
                contrast=opt.db_jitter,
                saturation=opt.db_jitter,
                hue=min(0.5, opt.db_jitter),
            )
        )
    ops.extend([TVT.ToTensor(), TVT.Normalize(mean=0.5, std=0.22)])
    return TVT.Compose(ops)


def load_qimage(datapath, split):
    image = Image.open(datapath).convert("RGB")
    return _query_transform(split)(image)


def load_dbimage(datapath, split):
    image = Image.open(datapath).convert("RGB")
    return _db_transform(split)(image)


def _move_to_device(data_dict, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in data_dict.items()}


class KITTI360BaseDataset(data.Dataset):
    def __init__(self, args, dataset_name="kitti360", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        self.test_method = getattr(args, "test_method", "single_query")
        self.resize = getattr(args, "resize", opt.q_resize)

        dataroot = opt.dataroot
        if not dataroot:
            raise ValueError("KITTI360 dataroot is empty. Please set --kitti360_path.")

        selectlocationlist = trainselectlocationlist if split == "train" else testselectlocationlist

        self.queries_infos = []
        self.queries_utms = []
        resize = 320
        for selectlocation in selectlocationlist:
            qposedir = os.path.join(dataroot, "data_poses", selectlocation, "oxts/data")
            qimage00dir = os.path.join(dataroot, f"data_2d_raw_resize{resize}", selectlocation, "image_00/data_rect")
            qimage00names = sorted(os.listdir(qimage00dir))

            if split == "train":
                qimage00names = qimage00names[: int(len(qimage00names) * train_ratio)]
            else:
                qimage00names = qimage00names[int(len(qimage00names) * train_ratio) :]

            for i_sample, qimage00name in enumerate(qimage00names):
                if split == "train" and i_sample % opt.traindownsample != 0:
                    continue

                qimage00path = os.path.join(qimage00dir, qimage00name)
                qposepath = os.path.join(qposedir, qimage00name.replace(".png", ".txt"))
                with open(qposepath, "r", encoding="utf-8") as pose_file:
                    lat, lon = map(float, pose_file.readline().split(" ")[:2])
                east, north, _, _ = utm.from_latlon(lat, lon)
                self.queries_infos.append(
                    {
                        "east": east,
                        "north": north,
                        "qimage00path": qimage00path,
                        "qposepath": qposepath,
                        "location": selectlocation,
                    }
                )
                self.queries_utms.append([east, north])

        self.queries_utms = np.array(self.queries_utms, dtype=np.float32)

        self.database_infos = []
        self.database_utms = []
        scale = 1
        zoom = 20
        size = 320
        for selectlocation in selectlocationlist:
            db_satellite_dir = os.path.join(dataroot, f"data_aerial_{scale}_{zoom}_{size}_satellite", selectlocation)
            db_roadmap_dir = os.path.join(dataroot, f"data_aerial_{scale}_{zoom}_{size}_roadmap", selectlocation)
            dbnames = sorted(os.listdir(db_satellite_dir))
            if not share_db:
                if split == "train":
                    dbnames = dbnames[: int(len(dbnames) * train_ratio)]
                else:
                    dbnames = dbnames[int(len(dbnames) * train_ratio) :]

            for i_dbname, dbname in enumerate(dbnames):
                if split == "train" and i_dbname % opt.traindownsample != 0:
                    continue

                dbname_pure = dbname.replace(".png", "")
                east, north = map(float, dbname_pure.split("@")[1:3])
                db_satellite_path = os.path.join(db_satellite_dir, dbname)
                db_roadmap_path = os.path.join(db_roadmap_dir, dbname)
                self.database_infos.append(
                    {
                        "east": east,
                        "north": north,
                        "db_satellite_path": db_satellite_path,
                        "db_roadmap_path": db_roadmap_path,
                        "location": selectlocation,
                    }
                )
                self.database_utms.append([east, north])

        self.database_utms = np.array(self.database_utms, dtype=np.float32)
        knn = NearestNeighbors(n_jobs=max(1, opt.num_workers))
        knn.fit(self.database_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,
            radius=opt.val_positive_dist_threshold,
            return_distance=False,
        )

        self.database_queries_infos = self.database_infos + self.queries_infos
        self.database_num = len(self.database_infos)
        self.queries_num = len(self.queries_infos)

    def _load_query_image(self, query_info):
        if opt.camnames != "00":
            raise NotImplementedError(
                f"Pure-visual KITTI360 path is fixed to image_00/data_rect, got camnames={opt.camnames}"
            )
        return load_qimage(query_info["qimage00path"], self.split)

    def _load_db_maps(self, db_info):
        maps = []
        for maptype in opt.maptype.split("_"):
            if maptype == "satellite":
                maps.append(load_dbimage(db_info["db_satellite_path"], self.split))
            elif maptype == "roadmap":
                maps.append(load_dbimage(db_info["db_roadmap_path"], self.split))
            else:
                raise NotImplementedError(f"Unsupported maptype: {maptype}")
        return torch.stack(maps, dim=0)  # [nmap, 3, H, W]

    def __getitem__(self, index):
        if index >= self.database_num:
            query_info = self.database_queries_infos[index]
            return {
                "query_image": self._load_query_image(query_info),
                "query_eastnorth": torch.tensor([query_info["east"], query_info["north"]], dtype=torch.float32),
                "db_map": torch.empty(0),
                "db_eastnorth": torch.empty(0),
            }, index

        db_info = self.database_queries_infos[index]
        return {
            "query_image": torch.empty(0),
            "query_eastnorth": torch.empty(0),
            "db_map": self._load_db_maps(db_info),  # [nmap, 3, H, W]
            "db_eastnorth": torch.tensor([db_info["east"], db_info["north"]], dtype=torch.float32),
        }, index

    def __len__(self):
        return len(self.database_queries_infos)

    def get_positives(self):
        return self.soft_positives_per_query


class KITTI360TripletsDataset(KITTI360BaseDataset):
    def __init__(self, args, dataset_name="kitti360", split="train", pos_num_per_query=5, negs_num_per_query=5):
        super().__init__(args, dataset_name=dataset_name, split=split)
        self.args = args
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num
        self.pos_num_per_query = pos_num_per_query
        self.negs_num_per_query = negs_num_per_query
        self.is_inference = False
        self.default_triplets_count = min(getattr(args, "cache_refresh_rate", self.queries_num), self.queries_num)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(
            knn.radius_neighbors(
                self.queries_utms,
                radius=opt.train_positives_dist_threshold,
                return_distance=False,
            )
        )

        queries_without_any_hard_positive = np.where(
            np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0
        )[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(
                "There are %d queries without training positives. They will be dropped.",
                len(queries_without_any_hard_positive),
            )
        self.hard_positives_per_query = [
            p for i, p in enumerate(self.hard_positives_per_query) if i not in queries_without_any_hard_positive
        ]
        self.soft_positives_per_query = [
            p for i, p in enumerate(self.soft_positives_per_query) if i not in queries_without_any_hard_positive
        ]
        self.queries_infos = [q for i, q in enumerate(self.queries_infos) if i not in queries_without_any_hard_positive]
        self.queries_utms = np.array(
            [q for i, q in enumerate(self.queries_utms) if i not in queries_without_any_hard_positive],
            dtype=np.float32,
        )
        self.database_queries_infos = self.database_infos + self.queries_infos
        self.queries_num = len(self.queries_infos)

    @staticmethod
    def _ensure_fixed_count(indexes, target_count):
        indexes = np.asarray(indexes, dtype=np.int32)
        if indexes.size == 0:
            raise RuntimeError("Cannot sample from an empty index list.")
        if indexes.size >= target_count:
            return indexes[:target_count]
        extra = np.random.choice(indexes, target_count - indexes.size, replace=True)
        return np.concatenate([indexes, extra], axis=0)

    def bootstrap_triplets_from_geometry(self, triplets_count=None):
        if self.queries_num == 0 or self.database_num == 0:
            self.triplets_global_indexes = torch.empty((0, 1 + self.pos_num_per_query + self.negs_num_per_query), dtype=torch.long)
            return

        if triplets_count is None:
            triplets_count = self.default_triplets_count
        triplets_count = min(int(triplets_count), self.queries_num)
        if triplets_count <= 0:
            self.triplets_global_indexes = torch.empty((0, 1 + self.pos_num_per_query + self.negs_num_per_query), dtype=torch.long)
            return

        sampled_queries_indexes = np.random.choice(self.queries_num, triplets_count, replace=False)
        all_database_indexes = np.arange(self.database_num, dtype=np.int32)
        triplets = []

        for query_index in sampled_queries_indexes:
            positive_indexes = np.asarray(self.hard_positives_per_query[query_index], dtype=np.int32)
            if positive_indexes.size == 0:
                continue

            query_utm = self.queries_utms[query_index]
            positive_utms = self.database_utms[positive_indexes]
            positive_distances = np.linalg.norm(positive_utms - query_utm[None, :], axis=1)
            ordered_positive_indexes = positive_indexes[np.argsort(positive_distances)]
            selected_positive_indexes = self._ensure_fixed_count(ordered_positive_indexes, self.pos_num_per_query)

            soft_positives = np.asarray(self.soft_positives_per_query[query_index], dtype=np.int32)
            neg_candidates = np.setdiff1d(all_database_indexes, soft_positives, assume_unique=False)
            if neg_candidates.size == 0:
                neg_candidates = np.setdiff1d(
                    all_database_indexes,
                    selected_positive_indexes,
                    assume_unique=False,
                )
            if neg_candidates.size == 0:
                continue

            replace = neg_candidates.size < self.negs_num_per_query
            neg_indexes = np.random.choice(neg_candidates, self.negs_num_per_query, replace=replace)
            triplets.append((query_index, *selected_positive_indexes.tolist(), *neg_indexes.tolist()))

        if not triplets:
            raise RuntimeError("Failed to bootstrap KITTI360 triplets from geometry.")
        self.triplets_global_indexes = torch.tensor(triplets, dtype=torch.long)

    def __getitem__(self, index):
        if self.is_inference:
            return super().__getitem__(index)

        query_index, positive_indexes, neg_indexes = torch.split(
            self.triplets_global_indexes[index],
            (1, self.pos_num_per_query, self.negs_num_per_query),
        )
        query_index = int(query_index.item())
        positive_indexes = positive_indexes.tolist()
        neg_indexes = neg_indexes.tolist()

        query_info = self.queries_infos[query_index]
        query_image = self._load_query_image(query_info)  # [3, H, W]
        query_eastnorth = torch.tensor([query_info["east"], query_info["north"]], dtype=torch.float32)

        positive_db_map = torch.stack(
            [self._load_db_maps(self.database_infos[pos_index]) for pos_index in positive_indexes],
            dim=0,
        )  # [npos, nmap, 3, H, W]
        negative_db_map = torch.stack(
            [self._load_db_maps(self.database_infos[neg_index]) for neg_index in neg_indexes],
            dim=0,
        )  # [nneg, nmap, 3, H, W]

        positive_db_eastnorth = torch.stack(
            [
                torch.tensor([self.database_infos[pos_index]["east"], self.database_infos[pos_index]["north"]], dtype=torch.float32)
                for pos_index in positive_indexes
            ],
            dim=0,
        )  # [npos, 2]
        negative_db_eastnorth = torch.stack(
            [
                torch.tensor([self.database_infos[neg_index]["east"], self.database_infos[neg_index]["north"]], dtype=torch.float32)
                for neg_index in neg_indexes
            ],
            dim=0,
        )  # [nneg, 2]

        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int64)
        for pos_num in range(len(positive_indexes)):
            for neg_num in range(len(neg_indexes)):
                triplets_local_indexes = torch.cat(
                    (
                        triplets_local_indexes,
                        torch.tensor([[0, 1 + pos_num, 1 + self.pos_num_per_query + neg_num]], dtype=torch.int64),
                    ),
                    dim=0,
                )

        output_dict = {
            "query_image": query_image,
            "query_eastnorth": query_eastnorth,
            "positive_db_map": positive_db_map,
            "negative_db_map": negative_db_map,
            "positive_db_eastnorth": positive_db_eastnorth,
            "negative_db_eastnorth": negative_db_eastnorth,
        }
        return output_dict, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self):
        if self.is_inference:
            return super().__len__()
        if not hasattr(self, "triplets_global_indexes"):
            return self.default_triplets_count
        return len(self.triplets_global_indexes)

    def compute_triplets(self, args, model, modelq=None):
        self.is_inference = True
        if self.mining not in {"partial_sep", "partial"}:
            raise NotImplementedError("Pure-visual KITTI360 path currently supports partial_sep/partial mining only.")
        if modelq is None:
            modelq = model
        self.compute_triplets_partial_sep(args, model, modelq)
        self.is_inference = False

    @staticmethod
    def compute_cache_sep(args, model, subset_ds, cache_shape, modelq):
        model = model.eval()
        modelq = modelq.eval()

        subset_indices = subset_ds.indices if isinstance(subset_ds, Subset) else list(range(len(subset_ds)))
        dataset = subset_ds.dataset if isinstance(subset_ds, Subset) else subset_ds

        db_indices = [idx for idx in subset_indices if idx < dataset.database_num]
        q_indices = [idx for idx in subset_indices if idx >= dataset.database_num]

        subset_ds_db = Subset(dataset, db_indices)
        subset_ds_q = Subset(dataset, q_indices)

        subset_dl_db = DataLoader(
            subset_ds_db,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=kitti360_collate_fn_cache_db,
        )
        subset_dl_q = DataLoader(
            subset_ds_q,
            num_workers=args.num_workers,
            batch_size=args.infer_batch_size,
            shuffle=False,
            pin_memory=args.device.startswith("cuda"),
            collate_fn=kitti360_collate_fn_cache_q,
        )

        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32)
        with torch.no_grad():
            for data_dict, indexes in tqdm(subset_dl_db):
                data_dict = _move_to_device(data_dict, args.device)
                features = model(data_dict, mode="db")
                cache[indexes.numpy()] = features["embedding"].cpu().numpy()
            for data_dict, indexes in tqdm(subset_dl_q):
                data_dict = _move_to_device(data_dict, args.device)
                features = modelq(data_dict, mode="q")
                cache[indexes.numpy()] = features["embedding"].cpu().numpy()
        return cache

    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"Query feature for index {query_index} was not cached.")
        return query_features

    def get_positive_indexes(self, args, query_index, cache, query_features):
        positive_indexes = np.asarray(self.hard_positives_per_query[query_index], dtype=np.int32)
        positives_features = cache[positive_indexes]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        top_k = min(self.pos_num_per_query, len(positive_indexes))
        _, positive_nums = faiss_index.search(query_features.reshape(1, -1), top_k)
        selected_positive_indexes = positive_indexes[positive_nums.reshape(-1)]
        return self._ensure_fixed_count(selected_positive_indexes, self.pos_num_per_query)

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_samples = np.asarray(neg_samples, dtype=np.int32)
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        top_k = min(self.negs_num_per_query, len(neg_samples))
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), top_k)
        selected_neg_indexes = neg_samples[neg_nums.reshape(-1)]
        return self._ensure_fixed_count(selected_neg_indexes, self.negs_num_per_query)

    def compute_triplets_partial_sep(self, args, model, modelq):
        self.triplets_global_indexes = []
        sampled_queries_num = min(self.queries_num, args.cache_refresh_rate)
        sampled_queries_indexes = np.random.choice(self.queries_num, sampled_queries_num, replace=False)

        sampled_database_num = min(self.database_num, self.neg_samples_num)
        sampled_database_indexes = np.random.choice(self.database_num, sampled_database_num, replace=False)

        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]
        database_indexes = list(np.unique(list(sampled_database_indexes) + positives_indexes))

        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache_sep(
            args,
            model,
            subset_ds,
            cache_shape=(len(self), args.features_dim),
            modelq=modelq,
        )

        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache)
            positive_indexes = self.get_positive_indexes(args, query_index, cache, query_features)

            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, *positive_indexes.tolist(), *neg_indexes.tolist()))

        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes, dtype=torch.long)


class RAMEfficient2DMatrix:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]

    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)

    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        return self.matrix[index]
