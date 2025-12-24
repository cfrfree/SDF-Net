import os
import os.path as osp
import glob
import re
from .bases import BaseImageDataset


class SYSU(BaseImageDataset):
    """
    SYSU-MM01 Dataset
    Directory structure:
    root/
      cam1/ ... cam6/
      exp/
        train_id.txt
        test_id.txt
    """

    dataset_dir = ""

    def __init__(self, root="", verbose=True, **kwargs):
        super(SYSU, self).__init__()
        self.dataset_dir = root

        self.train_id_file = osp.join(self.dataset_dir, "exp", "train_id.txt")
        self.test_id_file = osp.join(self.dataset_dir, "exp", "test_id.txt")
        self.val_id_file = osp.join(self.dataset_dir, "exp", "val_id.txt")

        self._check_before_run()

        # 1. 加载原始 ID
        train_ids_raw = self._load_ids(self.train_id_file)

        # [关键步骤] 建立 ID 映射表：Raw PID -> Continuous Label (0, 1, 2...)
        # 这一步是为了防止 CrossEntropyLoss 越界
        self.pid2label = {pid: i for i, pid in enumerate(sorted(train_ids_raw))}

        if osp.exists(self.test_id_file):
            test_ids_raw = self._load_ids(self.test_id_file)
        else:
            print(f"Warning: {self.test_id_file} not found. Trying val_id.txt")
            if osp.exists(self.val_id_file):
                test_ids_raw = self._load_ids(self.val_id_file)
            else:
                raise RuntimeError("Could not find test_id.txt or val_id.txt in exp/")

        # 2. 处理数据 (传入映射逻辑)
        train = self._process_dir(train_ids_raw, is_train=True)
        query = self._process_dir(test_ids_raw, mode="query")
        gallery = self._process_dir(test_ids_raw, mode="gallery")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = len(train_ids_raw)
        self.num_query_pids = len(test_ids_raw)
        self.num_gallery_pids = len(test_ids_raw)

        self.num_train_imgs = len(train)
        self.num_query_imgs = len(query)
        self.num_gallery_imgs = len(gallery)

        self.num_train_cams = 6
        self.num_query_cams = 2
        self.num_gallery_cams = 4

        if verbose:
            print("=> SYSU-MM01 Loaded")
            self.print_dataset_statistics(train, query, gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_id_file):
            raise RuntimeError("'{}' is not available".format(self.train_id_file))

    def _load_ids(self, file_path):
        with open(file_path, "r") as f:
            ids = f.read().splitlines()
            if len(ids) == 1 and "," in ids[0]:
                ids = ids[0].split(",")
        return [int(i) for i in ids if i.strip().isdigit()]

    def _process_dir(self, pids, is_train=False, mode="all"):
        dataset = []
        pids_container = set(pids)

        for cam_idx in range(1, 7):
            cam_dir = osp.join(self.dataset_dir, f"cam{cam_idx}")
            if not osp.exists(cam_dir):
                continue

            # 模态映射: 3,6 是 IR(1), 其他是 RGB(0)
            if cam_idx in [3, 6]:
                modality_label = 1
                is_ir_cam = True
            else:
                modality_label = 0
                is_ir_cam = False

            if not is_train:
                if mode == "query" and not is_ir_cam:
                    continue
                if mode == "gallery" and is_ir_cam:
                    continue

            pid_dirs = os.listdir(cam_dir)
            for pid_dir in pid_dirs:
                if not pid_dir.isdigit():
                    continue

                pid = int(pid_dir)
                if pid not in pids_container:
                    continue

                # [关键修改] 如果是训练集，必须进行 ID 重映射 (Relabel)
                if is_train:
                    pid_label = self.pid2label[pid]
                else:
                    # 测试集不需要重映射，或者保留原始 PID 用于匹配
                    pid_label = pid

                img_paths = glob.glob(osp.join(cam_dir, pid_dir, "*.jpg"))

                for img_path in img_paths:
                    dataset.append((img_path, pid_label, modality_label, cam_idx))

        return dataset
