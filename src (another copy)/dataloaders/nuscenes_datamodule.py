from types import SimpleNamespace

try:
    import lightning as L
except ModuleNotFoundError:
    import pytorch_lightning as L
from torch.utils.data import DataLoader, Subset

from .datasets_ws_nuscenes import (
    NuScenesBaseDataset,
    NuScenesTripletsDataset,
    configure_nuscenes_options,
    nuscenes_collate_fn,
    nuscenes_collate_fn_cache_db,
    nuscenes_collate_fn_cache_q,
)


class NuScenesTripletDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataroot,
        batch_size,
        num_workers,
        features_dim,
        query_img_size=(224, 224),
        db_img_size=(224, 224),
        db_crop_size=320,
        q_jitter=0.0,
        db_jitter=0.0,
        train_version="v1.0-trainval",
        val_version="v1.0-test",
        train_ratio=1.0,
        share_db=False,
        traindownsample=1,
        camnames="CAM_FRONT",
        maptype="satellite",
        locations="boston-seaport,singapore-hollandvillage,singapore-onenorth,singapore-queenstown",
        aerial_scale=1,
        aerial_zoom=20,
        aerial_size=320,
        use_keyframes_only=True,
        mining="partial_sep",
        neg_samples_num=1000,
        pos_num_per_query=5,
        negs_num_per_query=5,
        cache_refresh_rate=1000,
        infer_batch_size=32,
        train_positives_dist_threshold=10.0,
        val_positive_dist_threshold=25.0,
        shuffle=True,
        pin_memory=True,
    ):
        super().__init__()
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features_dim = features_dim
        self.query_img_size = query_img_size
        self.db_img_size = db_img_size
        self.db_crop_size = db_crop_size
        self.q_jitter = q_jitter
        self.db_jitter = db_jitter
        self.train_version = train_version
        self.val_version = val_version
        self.train_ratio = train_ratio
        self.share_db = share_db
        self.traindownsample = traindownsample
        self.camnames = camnames
        self.maptype = maptype
        self.locations = locations
        self.aerial_scale = aerial_scale
        self.aerial_zoom = aerial_zoom
        self.aerial_size = aerial_size
        self.use_keyframes_only = use_keyframes_only
        self.mining = mining
        self.neg_samples_num = neg_samples_num
        self.pos_num_per_query = pos_num_per_query
        self.negs_num_per_query = negs_num_per_query
        self.cache_refresh_rate = cache_refresh_rate
        self.infer_batch_size = infer_batch_size
        self.train_positives_dist_threshold = train_positives_dist_threshold
        self.val_positive_dist_threshold = val_positive_dist_threshold
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None

        self.dataset_args = SimpleNamespace(
            resize=query_img_size,
            test_method="single_query",
            mining=mining,
            neg_samples_num=neg_samples_num,
            pos_num_per_query=pos_num_per_query,
            negs_num_per_query=negs_num_per_query,
            num_workers=num_workers,
            infer_batch_size=infer_batch_size,
            cache_refresh_rate=cache_refresh_rate,
            device="cpu",
            features_dim=features_dim,
        )

    def setup(self, stage=None):
        if stage not in ("fit", None, "reload"):
            return
        if self.train_dataset is not None:
            return

        configure_nuscenes_options(
            dataroot=self.dataroot,
            train_version=self.train_version,
            val_version=self.val_version,
            train_ratio=self.train_ratio,
            share_db=self.share_db,
            q_resize=self.query_img_size,
            q_jitter=self.q_jitter,
            db_cropsize=self.db_crop_size,
            db_resize=self.db_img_size,
            db_jitter=self.db_jitter,
            camnames=self.camnames,
            maptype=self.maptype,
            locations=self.locations,
            aerial_scale=self.aerial_scale,
            aerial_zoom=self.aerial_zoom,
            aerial_size=self.aerial_size,
            use_keyframes_only=self.use_keyframes_only,
            traindownsample=self.traindownsample,
            num_workers=self.num_workers,
            train_positives_dist_threshold=self.train_positives_dist_threshold,
            val_positive_dist_threshold=self.val_positive_dist_threshold,
        )

        self.train_dataset = NuScenesTripletsDataset(
            args=self.dataset_args,
            dataset_name="nuscenes",
            split="train",
            pos_num_per_query=self.pos_num_per_query,
            negs_num_per_query=self.negs_num_per_query,
        )
        self.dataset_args.cache_refresh_rate = min(self.cache_refresh_rate, self.train_dataset.queries_num)
        self.train_dataset.default_triplets_count = self.dataset_args.cache_refresh_rate
        self.train_dataset.bootstrap_triplets_from_geometry(triplets_count=self.dataset_args.cache_refresh_rate)

        self.val_dataset = NuScenesBaseDataset(
            args=self.dataset_args,
            dataset_name="nuscenes",
            split="test",
        )

    def refresh_train_triplets(self, db_model, q_model=None):
        self.setup(stage="fit")
        device = getattr(db_model, "device", "cpu")
        self.dataset_args.device = "cuda" if hasattr(device, "type") and device.type == "cuda" else str(device)
        self.dataset_args.features_dim = getattr(db_model, "descriptor_dim", self.features_dim)
        self.train_dataset.compute_triplets(
            self.dataset_args,
            db_model,
            modelq=q_model if q_model is not None else db_model,
        )

    def train_dataloader(self):
        self.setup(stage="fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=nuscenes_collate_fn,
        )

    def val_dataloader(self):
        self.setup(stage="fit")
        if self.val_dataset is None or self.val_dataset.database_num == 0 or self.val_dataset.queries_num == 0:
            return []

        database_subset = Subset(self.val_dataset, list(range(self.val_dataset.database_num)))
        query_subset = Subset(self.val_dataset, list(range(self.val_dataset.database_num, len(self.val_dataset))))

        common_kwargs = dict(
            num_workers=self.num_workers,
            batch_size=self.infer_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
        return [
            DataLoader(
                database_subset,
                collate_fn=nuscenes_collate_fn_cache_db,
                **common_kwargs,
            ),
            DataLoader(
                query_subset,
                collate_fn=nuscenes_collate_fn_cache_q,
                **common_kwargs,
            ),
        ]
