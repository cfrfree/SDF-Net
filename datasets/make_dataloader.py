import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import random
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .hoss import HOSS
from .pretrain import Pretrain


__factory = {
    "HOSS": HOSS,
    "Pretrain": Pretrain,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_collate_fn(batch):
    imgs, pids, camids, viewids, _, img_size = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_size


def train_pair_collate_fn(batch):
    rgb_batch = [i[0] for i in batch]
    sar_batch = [i[1] for i in batch]
    batch = rgb_batch + sar_batch
    imgs, pids, camids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, img_size = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    img_size = torch.tensor(img_size, dtype=torch.float32)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        camids_batch,
        viewids,
        img_paths,
        img_size,
    )


def make_dataloader(cfg, is_train=True):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            # T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
            ),
        ]
    )

    val_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](
        root=cfg.DATASETS.ROOT_DIR, is_train=is_train
    )
    if is_train:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
        num_classes = dataset.num_train_pids
        cam_num = dataset.num_train_cams
        train_loader_normal = DataLoader(
            train_set_normal,
            batch_size=cfg.TEST.IMS_PER_BATCH,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=val_collate_fn,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
        )
        if "triplet" in cfg.DATALOADER.SAMPLER:
            if cfg.MODEL.DIST_TRAIN:
                print("DIST_TRAIN START")
                mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE
                )
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True
                )
                train_loader = DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
                )
            else:
                train_loader = DataLoader(
                    train_set,
                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(
                        dataset.train,
                        cfg.SOLVER.IMS_PER_BATCH,
                        cfg.DATALOADER.NUM_INSTANCE,
                    ),
                    num_workers=num_workers,
                    collate_fn=train_collate_fn,
                    worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
                )
        elif cfg.DATALOADER.SAMPLER == "softmax":
            print("using softmax sampler")
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=train_collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
            )
        else:
            print(
                "unsupported sampler! expected softmax or triplet but got {}".format(
                    cfg.SAMPLER
                )
            )
    else:
        train_loader = None
        train_loader_normal = None
        num_classes = 0
        cam_num = 2

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.TEST.IMS_PER_BATCH,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collate_fn,
    )

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    return (
        train_loader,
        train_loader_normal,
        val_loader,
        len(dataset.query),
        num_classes,
        cam_num,
    )


def make_dataloader_pair(cfg):
    train_transforms = T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB, mode="pixel", max_count=1, device="cpu"
            ),
        ]
    )

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set_pair = ImageDataset(dataset.train_pair, train_transforms, pair=True)
    num_classes = dataset.num_train_pair_pids
    cam_num = dataset.num_train_pair_cams

    if cfg.SOLVER.IMS_PER_BATCH % 2 != 0:
        raise ValueError("cfg.SOLVER.IMS_PER_BATCH should be even number")
    if cfg.MODEL.DIST_TRAIN:
        sampler = torch.utils.data.distributed.DistributedSampler(train_set_pair)
        train_loader_pair = DataLoader(
            train_set_pair,
            batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=train_pair_collate_fn,
        )
    else:
        train_loader_pair = DataLoader(
            train_set_pair,
            batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_pair_collate_fn,
        )
    return train_loader_pair, num_classes, cam_num
