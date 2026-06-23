import json
import logging
import os
import re
from collections import defaultdict
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as TVT
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


DEFAULT_NUSCENES_LOCATIONS = [
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
]


def _default_nuscenes_options():
    return SimpleNamespace(
        dataroot="",
        train_version="v1.0-trainval",
        val_version="v1.0-test",
        train_ratio=1.0,
        share_db=False,
        q_resize=(224, 224),
        q_jitter=0.0,
        db_cropsize=320,
        db_resize=(224, 224),
        db_jitter=0.0,
        camnames="CAM_FRONT",
        maptype="satellite",
        locations=",".join(DEFAULT_NUSCENES_LOCATIONS),
        aerial_scale=1,
        aerial_zoom=20,
        aerial_size=320,
        use_keyframes_only=True,
        traindownsample=1,
        num_workers=0,
        train_positives_dist_threshold=10.0,
        val_positive_dist_threshold=25.0,
    )


opt = _default_nuscenes_options()
train_ratio = opt.train_ratio
share_db = opt.share_db


def configure_nuscenes_options(**kwargs):
    global opt, train_ratio, share_db
    for key, value in kwargs.items():
        if value is not None:
            setattr(opt, key, value)
    train_ratio = float(opt.train_ratio)
    share_db = bool(opt.share_db)
    return opt


def _as_tuple_hw(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Expected a 2-tuple/list size, got {value!r}.")


def _parse_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def nuscenes_collate_fn(batch):
    query_image = torch.stack([e[0]["query_image"] for e in batch])
    query_eastnorth = torch.stack([e[0]["query_eastnorth"] for e in batch]).float()
    positive_db_map = torch.stack([e[0]["positive_db_map"] for e in batch])
    negative_db_map = torch.stack([e[0]["negative_db_map"] for e in batch])
    positive_db_eastnorth = torch.stack([e[0]["positive_db_eastnorth"] for e in batch]).float()
    negative_db_eastnorth = torch.stack([e[0]["negative_db_eastnorth"] for e in batch]).float()

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


def nuscenes_collate_fn_cache_db(batch):
    db_map = torch.stack([e[0]["db_map"] for e in batch])
    indices = torch.tensor([e[1] for e in batch], dtype=torch.long)
    return {"db_map": db_map}, indices


def nuscenes_collate_fn_cache_q(batch):
    query_image = torch.stack([e[0]["query_image"] for e in batch])
    query_eastnorth = torch.stack([e[0]["query_eastnorth"] for e in batch]).float()
    indices = torch.tensor([e[1] for e in batch], dtype=torch.long)
    return {"query_image": query_image, "query_eastnorth": query_eastnorth}, indices


def _query_transform(split):
    q_resize = _as_tuple_hw(opt.q_resize)
    ops = []
    if q_resize is not None:
        ops.append(TVT.Resize(q_resize))
    if split == "train" and opt.q_jitter > 0:
        ops.append(TVT.ColorJitter(brightness=opt.q_jitter, contrast=opt.q_jitter, saturation=opt.q_jitter, hue=min(0.5, opt.q_jitter)))
    ops.extend([TVT.ToTensor(), TVT.Normalize(mean=0.5, std=0.22)])
    return TVT.Compose(ops)


def _db_transform(split):
    db_resize = _as_tuple_hw(opt.db_resize)
    ops = []
    if opt.db_cropsize is not None and int(opt.db_cropsize) > 0:
        ops.append(TVT.CenterCrop(int(opt.db_cropsize)))
    if db_resize is not None:
        ops.append(TVT.Resize(db_resize))
    if split == "train" and opt.db_jitter > 0:
        ops.append(TVT.ColorJitter(brightness=opt.db_jitter, contrast=opt.db_jitter, saturation=opt.db_jitter, hue=min(0.5, opt.db_jitter)))
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


def _load_json_table(dataroot, version, table_name):
    table_path = os.path.join(dataroot, version, f"{table_name}.json")
    if not os.path.isfile(table_path):
        raise FileNotFoundError(f"nuScenes metadata table not found: {table_path}")
    with open(table_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _resolve_image_path(dataroot, filename):
    path = os.path.join(dataroot, filename)
    if os.path.isfile(path):
        return path
    parts = filename.split("/")
    if len(parts) >= 3 and parts[0] == "samples":
        alt_path = os.path.join(dataroot, "samples2", *parts[1:])
        if os.path.isfile(alt_path):
            return alt_path
    return path


def _split_records_by_location(records, split, train_ratio_value, should_split):
    if not should_split:
        return records
    by_location = defaultdict(list)
    for record in records:
        by_location[record["location"]].append(record)
    selected = []
    for _, location_records in by_location.items():
        location_records = sorted(location_records, key=lambda item: item.get("timestamp", 0))
        split_idx = int(len(location_records) * train_ratio_value)
        selected.extend(location_records[:split_idx] if split == "train" else location_records[split_idx:])
    return selected


def _parse_xy_from_tile_name(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("@")
    if len(parts) >= 3:
        try:
            return float(parts[1]), float(parts[2])
        except ValueError:
            pass
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", stem)
    if len(numbers) >= 2:
        return float(numbers[-2]), float(numbers[-1])
    raise ValueError(f"Cannot parse x/y coordinates from aerial tile name: {filename}")


def _candidate_aerial_dirs(dataroot, version, location, maptype):
    scale = int(opt.aerial_scale)
    zoom = int(opt.aerial_zoom)
    size = int(opt.aerial_size)
    base = f"aerial_{version}_{location}_{scale}_{zoom}_{size}_{maptype}"
    return [
        os.path.join(dataroot, base),
        os.path.join(dataroot, f"{size}_{base}"),
        os.path.join(dataroot, f"{zoom}_{base}"),
        os.path.join(dataroot, f"{scale}_{zoom}_{size}_{base}"),
    ]


def _find_aerial_dir(dataroot, version, location, maptype):
    for directory in _candidate_aerial_dirs(dataroot, version, location, maptype):
        if os.path.isdir(directory):
            return directory
    return None


class NuScenesBaseDataset(data.Dataset):
    def __init__(self, args, dataset_name="nuscenes", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.split = split
        self.test_method = getattr(args, "test_method", "single_query")
        self.resize = getattr(args, "resize", opt.q_resize)
        dataroot = opt.dataroot
        if not dataroot:
            raise ValueError("NuScenes dataroot is empty. Please set --nuscenes_path.")
        self.dataroot = dataroot
        self.version = opt.train_version if split == "train" else opt.val_version
        self.train_version = opt.train_version
        self.val_version = opt.val_version
        self.locations = set(_parse_csv(opt.locations)) or set(DEFAULT_NUSCENES_LOCATIONS)
        self.camnames = set(_parse_csv(opt.camnames)) or {"CAM_FRONT"}

        self.queries_infos, self.queries_utms = self._build_query_records()
        self.database_infos, self.database_utms = self._build_database_records()

        if self.database_infos:
            valid_locations = {item["location"] for item in self.database_infos}
            kept_queries = [(info, xy) for info, xy in zip(self.queries_infos, self.queries_utms) if info["location"] in valid_locations]
            if kept_queries:
                self.queries_infos, query_xy = zip(*kept_queries)
                self.queries_infos = list(self.queries_infos)
                self.queries_utms = np.array(query_xy, dtype=np.float32)
            else:
                self.queries_infos = []
                self.queries_utms = np.zeros((0, 2), dtype=np.float32)

        if len(self.database_infos) == 0:
            raise RuntimeError(f"No nuScenes aerial database images found for split={split}, version={self.version}.")
        if len(self.queries_infos) == 0:
            raise RuntimeError(f"No nuScenes camera query images found for split={split}, version={self.version}.")

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

    def _build_query_records(self):
        version = self.version
        sample_data = _load_json_table(self.dataroot, version, "sample_data")
        samples = {item["token"]: item for item in _load_json_table(self.dataroot, version, "sample")}
        scenes = {item["token"]: item for item in _load_json_table(self.dataroot, version, "scene")}
        logs = {item["token"]: item for item in _load_json_table(self.dataroot, version, "log")}
        ego_poses = {item["token"]: item for item in _load_json_table(self.dataroot, version, "ego_pose")}
        calibrated_sensors = {item["token"]: item for item in _load_json_table(self.dataroot, version, "calibrated_sensor")}
        sensors = {item["token"]: item for item in _load_json_table(self.dataroot, version, "sensor")}

        records = []
        for item in sample_data:
            if opt.use_keyframes_only and not bool(item.get("is_key_frame", False)):
                continue
            calib = calibrated_sensors.get(item.get("calibrated_sensor_token"))
            if calib is None:
                continue
            sensor = sensors.get(calib.get("sensor_token"))
            if sensor is None or sensor.get("modality") != "camera":
                continue
            channel = sensor.get("channel", "")
            if channel not in self.camnames:
                continue
            sample = samples.get(item.get("sample_token"))
            if sample is None:
                continue
            scene = scenes.get(sample.get("scene_token"))
            if scene is None:
                continue
            log = logs.get(scene.get("log_token"))
            if log is None:
                continue
            location = log.get("location", "")
            if location not in self.locations:
                continue
            ego_pose = ego_poses.get(item.get("ego_pose_token"))
            if ego_pose is None:
                continue
            translation = ego_pose.get("translation", [0.0, 0.0, 0.0])
            x, y = float(translation[0]), float(translation[1])
            image_path = _resolve_image_path(self.dataroot, item.get("filename", ""))
            if not os.path.isfile(image_path):
                logging.warning("Missing nuScenes camera image: %s", image_path)
                continue
            records.append({
                "east": x,
                "north": y,
                "qimage_path": image_path,
                "location": location,
                "scene_token": scene["token"],
                "scene_name": scene.get("name", ""),
                "sample_token": item.get("sample_token"),
                "sample_data_token": item.get("token"),
                "channel": channel,
                "timestamp": int(item.get("timestamp", 0)),
            })
        should_split = self.train_version == self.val_version
        records = _split_records_by_location(records, self.split, train_ratio, should_split)
        if self.split == "train" and int(opt.traindownsample) > 1:
            records = records[:: int(opt.traindownsample)]
        records = sorted(records, key=lambda item: (item["location"], item["timestamp"], item["channel"]))
        utms = np.array([[item["east"], item["north"]] for item in records], dtype=np.float32)
        if utms.size == 0:
            utms = np.zeros((0, 2), dtype=np.float32)
        return records, utms

    def _build_database_records(self):
        records = []
        maptypes = _parse_csv(str(opt.maptype).replace("_", ",")) or ["satellite"]
        primary_maptype = maptypes[0]
        for location in sorted(self.locations):
            db_dir = _find_aerial_dir(self.dataroot, self.version, location, primary_maptype)
            if db_dir is None:
                logging.warning("No aerial directory found for version=%s location=%s maptype=%s.", self.version, location, primary_maptype)
                continue
            dbnames = sorted(name for name in os.listdir(db_dir) if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
            if len(dbnames) == 0:
                logging.warning("Aerial directory is empty: %s", db_dir)
                continue
            if not share_db and self.train_version == self.val_version:
                split_idx = int(len(dbnames) * train_ratio)
                dbnames = dbnames[:split_idx] if self.split == "train" else dbnames[split_idx:]
            if self.split == "train" and int(opt.traindownsample) > 1:
                dbnames = dbnames[:: int(opt.traindownsample)]
            for dbname in dbnames:
                try:
                    x, y = _parse_xy_from_tile_name(dbname)
                except ValueError as exc:
                    logging.warning(str(exc))
                    continue
                record = {"east": x, "north": y, "location": location, "db_paths": {}, "basename": dbname, "timestamp": 0}
                valid_for_all_maptypes = True
                for maptype in maptypes:
                    mt_dir = _find_aerial_dir(self.dataroot, self.version, location, maptype)
                    if mt_dir is None:
                        valid_for_all_maptypes = False
                        break
                    mt_path = os.path.join(mt_dir, dbname)
                    if not os.path.isfile(mt_path):
                        valid_for_all_maptypes = False
                        break
                    record["db_paths"][maptype] = mt_path
                if valid_for_all_maptypes:
                    records.append(record)
        records = sorted(records, key=lambda item: (item["location"], item["east"], item["north"], item["basename"]))
        utms = np.array([[item["east"], item["north"]] for item in records], dtype=np.float32)
        if utms.size == 0:
            utms = np.zeros((0, 2), dtype=np.float32)
        return records, utms

    def _load_query_image(self, query_info):
        return load_qimage(query_info["qimage_path"], self.split)

    def _load_db_maps(self, db_info):
        maps = []
        for maptype in _parse_csv(str(opt.maptype).replace("_", ",")):
            if maptype not in db_info["db_paths"]:
                raise NotImplementedError(f"Unsupported or missing maptype={maptype} for {db_info['basename']}.")
            maps.append(load_dbimage(db_info["db_paths"][maptype], self.split))
        return torch.stack(maps, dim=0)

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
            "db_map": self._load_db_maps(db_info),
            "db_eastnorth": torch.tensor([db_info["east"], db_info["north"]], dtype=torch.float32),
        }, index

    def __len__(self):
        return len(self.database_queries_infos)

    def get_positives(self):
        return self.soft_positives_per_query


class NuScenesTripletsDataset(NuScenesBaseDataset):
    def __init__(self, args, dataset_name="nuscenes", split="train", pos_num_per_query=5, negs_num_per_query=5):
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
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms, radius=opt.train_positives_dist_threshold, return_distance=False))
        drop = set(np.where(np.array([len(p) for p in self.hard_positives_per_query]) == 0)[0].astype(int).tolist())
        if drop:
            logging.info("There are %d nuScenes queries without training positives. They will be dropped.", len(drop))
        self.hard_positives_per_query = [p for i, p in enumerate(self.hard_positives_per_query) if i not in drop]
        self.soft_positives_per_query = [p for i, p in enumerate(self.soft_positives_per_query) if i not in drop]
        self.queries_infos = [q for i, q in enumerate(self.queries_infos) if i not in drop]
        self.queries_utms = np.array([q for i, q in enumerate(self.queries_utms) if i not in drop], dtype=np.float32)
        if self.queries_utms.size == 0:
            self.queries_utms = np.zeros((0, 2), dtype=np.float32)
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
                neg_candidates = np.setdiff1d(all_database_indexes, selected_positive_indexes, assume_unique=False)
            if neg_candidates.size == 0:
                continue
            replace = neg_candidates.size < self.negs_num_per_query
            neg_indexes = np.random.choice(neg_candidates, self.negs_num_per_query, replace=replace)
            triplets.append((query_index, *selected_positive_indexes.tolist(), *neg_indexes.tolist()))
        if not triplets:
            raise RuntimeError("Failed to bootstrap nuScenes triplets from geometry.")
        self.triplets_global_indexes = torch.tensor(triplets, dtype=torch.long)

    def __getitem__(self, index):
        if self.is_inference:
            return super().__getitem__(index)
        query_index, positive_indexes, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, self.pos_num_per_query, self.negs_num_per_query))
        query_index = int(query_index.item())
        positive_indexes = positive_indexes.tolist()
        neg_indexes = neg_indexes.tolist()
        query_info = self.queries_infos[query_index]
        query_image = self._load_query_image(query_info)
        query_eastnorth = torch.tensor([query_info["east"], query_info["north"]], dtype=torch.float32)
        positive_db_map = torch.stack([self._load_db_maps(self.database_infos[pos_index]) for pos_index in positive_indexes], dim=0)
        negative_db_map = torch.stack([self._load_db_maps(self.database_infos[neg_index]) for neg_index in neg_indexes], dim=0)
        positive_db_eastnorth = torch.stack([torch.tensor([self.database_infos[pos_index]["east"], self.database_infos[pos_index]["north"]], dtype=torch.float32) for pos_index in positive_indexes], dim=0)
        negative_db_eastnorth = torch.stack([torch.tensor([self.database_infos[neg_index]["east"], self.database_infos[neg_index]["north"]], dtype=torch.float32) for neg_index in neg_indexes], dim=0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int64)
        for pos_num in range(len(positive_indexes)):
            for neg_num in range(len(neg_indexes)):
                triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([[0, 1 + pos_num, 1 + self.pos_num_per_query + neg_num]], dtype=torch.int64)), dim=0)
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
            raise NotImplementedError("Pure-visual nuScenes path currently supports partial_sep/partial mining only.")
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
        subset_dl_db = DataLoader(subset_ds_db, num_workers=args.num_workers, batch_size=args.infer_batch_size, shuffle=False, pin_memory=args.device.startswith("cuda"), collate_fn=nuscenes_collate_fn_cache_db)
        subset_dl_q = DataLoader(subset_ds_q, num_workers=args.num_workers, batch_size=args.infer_batch_size, shuffle=False, pin_memory=args.device.startswith("cuda"), collate_fn=nuscenes_collate_fn_cache_q)
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
        cache = self.compute_cache_sep(args, model, subset_ds, cache_shape=(len(self), args.features_dim), modelq=modelq)
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
