from PIL import Image, ImageFile
from torch.utils.data import Dataset
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm  # 引入进度条，让你知道加载进度

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def sar32bit2RGB(img):
    nimg = np.array(img, dtype=np.float32)
    nimg = nimg / nimg.max() * 255
    nimg_8 = nimg.astype(np.uint8)
    cv_img = cv2.cvtColor(nimg_8, cv2.COLOR_GRAY2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        if train is not None:
            (
                num_train_pids,
                num_train_imgs,
                num_train_cams,
                num_train_views,
            ) = self.get_imagedata_info(train)
        (
            num_query_pids,
            num_query_imgs,
            num_query_cams,
            num_train_views,
        ) = self.get_imagedata_info(query)
        (
            num_gallery_pids,
            num_gallery_imgs,
            num_gallery_cams,
            num_train_views,
        ) = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        if train is not None:
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, pair=False):
        self.dataset = dataset
        self.transform = transform
        self.pair = pair

        # [修改点] 开启内存缓存
        # 对于 HOSS 这种小数据集，这会极大地加速训练
        print(f"Initializing Dataset (Pair={pair}). Loading {len(self.dataset)} images to RAM Cache...")
        self.cached_samples = []

        # 预加载所有数据
        for i in tqdm(range(len(self.dataset)), desc="Loading Images"):
            if self.pair:
                # 处理 Pair 模式: dataset[i] 是一个列表 [(path, pid, cam), ...]
                pair_items = self.dataset[i]
                cached_pair_group = []
                for item in pair_items:
                    img_path, pid, camid = item
                    # 调用内部函数读取并处理图片 (不包含transform)
                    img_pil, img_size_meta = self._load_raw_image(img_path)
                    img_name = img_path.split("/")[-1]
                    # 存入缓存
                    cached_pair_group.append((img_pil, pid, camid, img_name, img_size_meta))
                self.cached_samples.append(cached_pair_group)
            else:
                # 处理 普通 模式: dataset[i] 是 tuple (path, pid, cam, track)
                img_path, pid, camid, trackid = self.dataset[i]
                img_pil, img_size_meta = self._load_raw_image(img_path)
                img_name = img_path.split("/")[-1]
                # 存入缓存
                self.cached_samples.append((img_pil, pid, camid, trackid, img_name, img_size_meta))

        print("RAM Cache Loaded Successfully!")

    def __len__(self):
        return len(self.dataset)

    def _load_raw_image(self, img_path):
        """
        读取图片并进行基础的格式转换 (SAR->RGB)，但不进行 Data Augmentation。
        这些操作只在初始化时做一次。
        """
        if img_path.endswith("SAR.tif"):
            img = read_image(img_path)
            img = sar32bit2RGB(img)
            img_size = img.size
        else:
            img = read_image(img_path).convert("RGB")
            img_size = img.size
            # 原始逻辑：如果是光学图，尺寸参数按 0.75 计算
            img_size = [img_size[0] * 0.75, img_size[1] * 0.75]

        # 预先计算好 img_size 的元组信息
        img_size_meta = (
            (img_size[0] / 93 - 0.434) / 0.031,
            (img_size[1] / 427 - 0.461) / 0.031,
            img_size[1] / img_size[0],
        )
        return img, img_size_meta

    def __getitem__(self, index):
        # [修改点] 直接从内存列表 self.cached_samples 取数据，不再读盘

        if self.pair:
            cached_pair_group = self.cached_samples[index]
            imgs = []
            for item in cached_pair_group:
                img_pil, pid, camid, img_name, img_size_meta = item

                # 实时进行数据增强 (Random Crop, Flip, Erasing 等)
                if self.transform is not None:
                    img_tensor = self.transform(img_pil)
                else:
                    img_tensor = img_pil

                imgs.append((img_tensor, pid, camid, img_name, img_size_meta))
            return imgs
        else:
            # 普通模式
            img_pil, pid, camid, trackid, img_name, img_size_meta = self.cached_samples[index]

            if self.transform is not None:
                img_tensor = self.transform(img_pil)
            else:
                img_tensor = img_pil

            return img_tensor, pid, camid, trackid, img_name, img_size_meta
