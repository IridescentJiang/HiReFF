import itertools
import os
import os.path as osp
import json
import cv2
import numpy as np
import random
import re
import glob
import logging
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from vggt.utils.load_fn import load_and_preprocess_images, adjust_intrinsic, normalize_extrinsic, \
    convert_extrinsics_to_relative


class StreamingDnaRenderingDataset(IterableDataset, Dataset):
    def __init__(
            self,
            data_root: str,
            min_frames: int = 1,
            target_size: int = 518,
            load_depth: bool = True,
            load_mask: bool = True,
            frame_numbers: [] = None,
            split: str = "train"
    ):
        super().__init__()
        self.data_root = data_root
        self.target_size = target_size
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.image_load_mode = "crop"
        self.origin_width = 2048
        self.origin_height = 2448
        self.split = split
        self.worker_info = None  # 用于存储 worker 信息
        self.frame_numbers = frame_numbers
        self.frame_records = []

        # 初始化视角组 (1-48步长6取8个，分为4组)
        base_views = list(range(1, 48, 3))
        self.view_groups = [
            [f"{v:02d}" for v in base_views[::4][:4]],  # 1, 13, 25, 37
            [f"{v:02d}" for v in base_views[1::4][:4]],
            [f"{v:02d}" for v in base_views[2::4][:4]],
            [f"{v:02d}" for v in base_views[3::4][:4]],
        ]

        #
        # self.supervise_view_groups = [
        #     # [f"{v:02d}" for v in base_views[::4][:4]],
        #     # [f"{v:02d}" for v in base_views[1::4][:4]],
        #     [f"{v:02d}" for v in base_views[2::4][:4]],  # 7, 19, 31, 43
        #     # [f"{v:02d}" for v in base_views[3::4][:4]],
        #     # [f"{v:02d}" for v in base_views]
        # ]

        # 生成随机视角组
        base_views = list(range(0, 24))
        all_pairs = list(itertools.combinations(base_views, 2))
        valid_pairs = [(a, b) for a, b in all_pairs if abs(a - b) >= 3]

        self.supervise_view_groups = []
        for a, b in valid_pairs:
            a_relative = a + 24
            b_relative = b + 24

            group = [f"{a:02d}", f"{b:02d}", f"{a_relative:02d}", f"{b_relative:02d}"]
            self.supervise_view_groups.append(group)

        # 发现所有有效序列和帧
        self.valid_samples = self._discover_samples(min_frames)

        # 根据split划分数据
        if int(len(self.valid_samples)) > 10:
            if split == "train":
                self.sample_indices = self.valid_samples[:int(len(self.valid_samples) * 0.95)]
            else:  # val
                self.sample_indices = self.valid_samples[int(len(self.valid_samples) * 0.95):]
        else:
            if split == "train":
                self.sample_indices = self.valid_samples[:int(len(self.valid_samples))]
            else:  # val
                self.sample_indices = self.valid_samples[:int(len(self.valid_samples))]

        # 预加载索引列表并打乱顺序
        random.shuffle(self.sample_indices)
        self.current_index = 0

        logging.info(f"Loaded {len(self.sample_indices)} valid samples for {split}")

    def _discover_samples(self, min_frames):
        """发现所有有效的序列和帧组合"""
        valid_samples = []

        for seq_dir in sorted(os.listdir(self.data_root)):
            if not re.match(r"^\d{4}$", seq_dir):
                continue

            seq_path = osp.join(self.data_root, seq_dir)
            logging.debug(f"Processing sequence: {seq_dir}")

            # 获取所有有效视角
            camera_files = glob.glob(osp.join(seq_path, "cameras", "camera_*.json"))
            view_ids = sorted([f.split("_")[-1].split(".")[0] for f in camera_files])

            # 获取有效帧
            image_dir = osp.join(seq_path, "images", view_ids[0])
            if self.frame_numbers:
                frame_numbers = self.frame_numbers
            else:
                frame_numbers = sorted([
                    f.split("_")[1].split(".")[0]
                    for f in os.listdir(image_dir)
                    if f.startswith("frame_") and f.endswith(".png")
                ])

            # 验证帧完整性
            valid_frames = []
            for frame in frame_numbers:
                valid = True
                # 检查所有必须的视角组
                for view_group in self.view_groups:
                    for view_id in view_group:
                        required_files = {
                            "image": osp.join(seq_path, "images", view_id, f"frame_{frame}.png"),
                            "kp2d": osp.join(seq_path, "keypoints_2d", view_id, f"frame_{frame}.npy"),
                            "kp3d": osp.join(seq_path, "keypoints_3d", view_id, f"frame_{frame}.npy"),
                            "camera": osp.join(seq_path, "cameras", f"camera_{view_id}.json"),
                            "mask": osp.join(seq_path, "masks", view_id, f"frame_{frame}.png"),
                            "smplx": osp.join(seq_path, "smplx", f"frame_{frame}_smplx.json")
                        }
                        if self.load_depth and (int(view_id) - 1) % 6 == 0:
                            required_files.update({
                                "depth": osp.join(seq_path, "kinect", "depth", f"{((int(view_id) - 1) % 6):02d}",
                                                  f"frame_{frame}.npy"),
                                "depth_mask": osp.join(seq_path, "kinect", "mask", f"{((int(view_id) - 1) % 6):02d}",
                                                       f"frame_{frame}.png")
                            })

                        # 验证文件存在性
                        for f in required_files.values():
                            if not os.path.exists(f):
                                valid = False
                                break
                        if not valid: break
                    if not valid: break

                if valid:
                    valid_frames.append(frame)

            # 记录有效帧
            if len(valid_frames) >= min_frames:
                for frame in valid_frames:
                    valid_samples.append({
                        "seq_dir": seq_dir,
                        "frame": frame,
                        "view_ids": view_ids,
                    })

            self.frame_records = valid_samples

        logging.info(f"Discovered {len(valid_samples)} valid samples")
        return valid_samples

    def __iter__(self):
        """数据集迭代器"""
        # 获取worker信息（用于多进程环境）
        worker_info = get_worker_info()
        num_workers = 1
        worker_id = 0

        if worker_info is not None:
            # 多进程环境
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            total_samples = len(self.sample_indices)

            # 将样本划分为每个worker一份
            samples_per_worker = total_samples // num_workers
            start_idx = worker_id * samples_per_worker
            end_idx = start_idx + samples_per_worker

            # 如果是最后一个worker，取所有剩余样本
            if worker_id == num_workers - 1:
                end_idx = total_samples

            worker_samples = self.sample_indices[start_idx:end_idx]
            samples_to_iterate = worker_samples

            # 有限循环 - 处理分配到的样本
            for sample in worker_samples:
                yield self.load_sample(sample["seq_dir"], sample["frame"])
        else:
            # 单进程环境
            # 有限循环
            for sample in self.sample_indices:
                yield self.load_sample(sample["seq_dir"], sample["frame"])

    def __len__(self):
        """返回数据集的实际长度"""
        return len(self.sample_indices)

    def __getitem__(self, idx):
        record = self.frame_records[idx]
        seq_dir = record["seq_dir"]
        frame = record["frame"]

        # 随机选择视角组
        view_group = random.choice(self.view_groups)
        random.shuffle(view_group)

        # 初始化数据容器
        image_paths, mask_paths = [], []
        kp2ds, kp3ds = [], []
        intrinsics, distortions, extrinsics = [], [], []
        depths, depth_masks = [], []
        Ds, Ks = [], []

        # 加载组内所有视角数据
        base_path = osp.join(self.data_root, seq_dir)
        for view_id in view_group:
            # 加载图像数据
            img_path = osp.join(base_path, "images", view_id, f"frame_{frame}.png")
            image_paths.append(img_path)

            # 加载掩码
            mask_path = osp.join(base_path, "masks", view_id, f"frame_{frame}.png")
            mask_paths.append(mask_path)

            # 加载关键点
            # kp2ds.append(np.load(osp.join(base_path, "keypoints_2d", view_id, f"frame_{frame}.npy")))
            # kp3ds.append(np.load(osp.join(base_path, "keypoints_3d", view_id, f"frame_{frame}.npy")))

            # 加载相机参数
            camera = self._load_camera(osp.join(base_path, "cameras", f"camera_{view_id}.json"))

            # 用于undistort
            Ks.append(camera["K"])
            Ds.append(camera["D"])

            adjusted_intrinsic = adjust_intrinsic(camera["K"],
                                                  self.origin_width, self.origin_height,
                                                  self.target_size, self.target_size, self.image_load_mode)

            intrinsics.append(adjusted_intrinsic)
            extrinsics.append(camera["RT"])

            # 加载深度数据
            if self.load_depth:
                kinect_view_id = f"{int((int(view_id) + 5) / 6):02d}"
                depths.append(np.load(osp.join(base_path, "kinect", "depth", kinect_view_id, f"frame_{frame}.npy")))
                depth_mask = self._load_mask(
                    osp.join(base_path, "kinect", "mask", kinect_view_id, f"frame_{frame}.png"))
                depth_masks.append(depth_mask.astype(np.float32))

        # 将之后的外参矩阵调整为第一个矩阵的相对矩阵
        relatived_extrinsics = convert_extrinsics_to_relative(extrinsics)

        # 转换为numpy数组
        data = {
            "images": load_and_preprocess_images(image_paths, self.target_size, mask_path_list=None, Ks=Ks, Ds=Ds),
            # [S, 3, H, W]
            "images_wo_bg": load_and_preprocess_images(image_paths, self.target_size, mask_path_list=mask_paths, Ks=Ks,
                                                       Ds=Ds),  # [S, H, W]
            # "kp2d": np.stack(kp2ds, axis=0).astype(np.float32),  # [S, N, 2]
            # "kp3d": np.stack(kp3ds, axis=0).astype(np.float32),  # [S, N, 3]
            "camera": {
                "intrinsics": np.stack(intrinsics, axis=0).astype(np.float32),  # [S, 3, 3]
                "extrinsics": np.stack(relatived_extrinsics, axis=0).astype(np.float32)  # [S, 4, 4]
            },
            # "smplx": self._load_smplx(osp.join(base_path, "smplx", f"frame_{frame}_smplx.json"))
        }

        if self.load_depth:
            data.update({
                "depths": np.stack(depths, axis=0).astype(np.float32),  # [S, H, W]
                "depth_masks": np.stack(depth_masks, axis=0).astype(np.float32)  # [S, H, W]
            })

        return data

    def _load_image(self, path):
        img = cv2.imread(path)
        return img

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return mask

    def _load_camera(self, path):
        with open(path, "r") as f:
            params = json.load(f)
        return {
            "K": np.array(params["K"]),
            "D": np.array(params["D"]),
            "RT": np.array(params["RT"]),
        }

    def _load_smplx(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def load_sample(self, seq_dir, frame):
        """加载单个样本数据"""
        base_path = osp.join(self.data_root, seq_dir)

        # 随机选择视角组
        view_group = random.choice(self.view_groups)
        random.shuffle(view_group)

        # 初始化数据容器
        image_paths, mask_paths = [], []
        kp2ds, kp3ds = [], []
        intrinsics, distortions, extrinsics = [], [], []
        depths, depth_masks = [], []
        Ks, Ds = [], []

        # 加载组内所有视角数据
        for view_id in view_group:
            # 图像路径
            img_path = osp.join(base_path, "images", view_id, f"frame_{frame}.png")
            image_paths.append(img_path)

            # 掩码路径
            mask_path = osp.join(base_path, "masks", view_id, f"frame_{frame}.png")
            mask_paths.append(mask_path)

            # 加载关键点
            # kp2d = np.load(osp.join(base_path, "keypoints_2d", view_id, f"frame_{frame}.npy"))
            # kp3d = np.load(osp.join(base_path, "keypoints_3d", view_id, f"frame_{frame}.npy"))
            # kp2ds.append(kp2d)
            # kp3ds.append(kp3d)

            # 加载相机参数
            camera = self._load_camera(osp.join(base_path, "cameras", f"camera_{view_id}.json"))

            # 用于undistort
            Ks.append(camera["K"])
            Ds.append(camera["D"])

            adjusted_intrinsic = adjust_intrinsic(camera["K"],
                                                  self.origin_width, self.origin_height,
                                                  self.target_size, self.target_size, self.image_load_mode)

            intrinsics.append(adjusted_intrinsic)
            extrinsics.append(camera["RT"])

            # 加载深度数据
            if self.load_depth:
                kinect_view_id = f"{int((int(view_id) + 5) / 6):02d}"
                depths.append(np.load(osp.join(base_path, "kinect", "depth", kinect_view_id, f"frame_{frame}.npy")))
                depth_mask = self._load_mask(
                    osp.join(base_path, "kinect", "mask", kinect_view_id, f"frame_{frame}.png"))
                depth_masks.append(depth_mask.astype(np.float32))

        # 将之后的外参矩阵调整为第一个矩阵的相对矩阵
        relatived_extrinsics = convert_extrinsics_to_relative(extrinsics)

        # 预处理图像
        images = load_and_preprocess_images(image_paths, self.target_size, mask_path_list=None, Ks=Ks, Ds=Ds)
        masks = load_and_preprocess_images(mask_paths, self.target_size, mask_path_list=None, Ks=Ks, Ds=Ds)
        images_wo_bg = load_and_preprocess_images(image_paths, self.target_size, mask_paths, Ks=Ks, Ds=Ds)

        # 转换为numpy数组
        data = {
            "images": images,
            "masks": masks[:, 0:1, :, :],  # 只取第一通道
            "images_wo_bg": images_wo_bg,
            "intrinsics": torch.tensor(np.array(intrinsics), dtype=torch.float32),
            "extrinsics": torch.tensor(np.array(relatived_extrinsics), dtype=torch.float32),
            # "smplx": self._load_smplx(osp.join(base_path, "smplx", f"frame_{frame}_smplx.json"))
        }

        if self.load_depth:
            data.update({
                "depths": np.stack(depths, axis=0).astype(np.float32),
                "depth_masks": np.stack(depth_masks, axis=0).astype(np.float32)
            })

        if self.supervise_view_groups:

            # 随机选择另一个视角组，与之前的不同
            view_group_supervise = random.choice(self.supervise_view_groups)
            random.shuffle(view_group_supervise)

            # 初始化数据容器
            image_paths_supervise, mask_paths_supervise = [], []
            intrinsics_supervise, distortions_supervise, extrinsics_supervise = [], [], []
            Ks_s, Ds_s = [], []

            # 加载组内所有视角数据
            for view_id in view_group_supervise:
                # 图像路径
                img_path = osp.join(base_path, "images", view_id, f"frame_{frame}.png")
                image_paths_supervise.append(img_path)

                # 掩码路径
                mask_path = osp.join(base_path, "masks", view_id, f"frame_{frame}.png")
                mask_paths_supervise.append(mask_path)

                # 加载相机参数
                camera = self._load_camera(osp.join(base_path, "cameras", f"camera_{view_id}.json"))

                # 用于undistort
                Ks_s.append(camera["K"])
                Ds_s.append(camera["D"])

                adjusted_intrinsic = adjust_intrinsic(camera["K"],
                                                      self.origin_width, self.origin_height,
                                                      self.target_size, self.target_size, self.image_load_mode)

                intrinsics_supervise.append(adjusted_intrinsic)
                extrinsics_supervise.append(camera["RT"])

            # 将之后的外参矩阵调整为第一个矩阵的相对矩阵
            all_extrinsics = extrinsics + extrinsics_supervise
            relatived_all_extrinsics = convert_extrinsics_to_relative(all_extrinsics)
            relatived_extrinsics_supervise = relatived_all_extrinsics[-len(extrinsics_supervise):]

            # 预处理图像
            images_supervise = load_and_preprocess_images(image_paths_supervise, self.target_size, mask_path_list=None,
                                                          Ks=Ks_s, Ds=Ds_s)
            masks_supervise = load_and_preprocess_images(mask_paths_supervise, self.target_size, mask_path_list=None,
                                                         Ks=Ks_s, Ds=Ds_s)
            images_wo_bg_supervise = load_and_preprocess_images(image_paths_supervise, self.target_size,
                                                                mask_paths_supervise, Ks=Ks_s, Ds=Ds_s)

            data.update({
                "supervise_images_wo_bg": images_wo_bg_supervise,
                # "supervise_images": images_supervise,
                "supervise_masks": masks_supervise[:, 0:1, :, :],  # 只取第一通道
                "supervise_intrinsics": torch.tensor(np.array(intrinsics_supervise), dtype=torch.float32),
                "supervise_extrinsics": torch.tensor(np.array(relatived_extrinsics_supervise), dtype=torch.float32)
            })

        return data


# 使用示例
if __name__ == "__main__":
    dataset = StreamingDnaRenderingDataset(
        data_root="/home/china/lab/VGGT_human/dna_rendering/data_example",
        min_frames=24,
        target_size=518,
        load_depth=True,
    )

    sample = dataset[0]
    print("Loaded keys:", sample.keys())
    print("Images shape:", sample["images"].shape)  # (4, 3, 512, 512)
    print("Camera K shape:", sample["camera"]["K"].shape)  # (4, 3, 3)
