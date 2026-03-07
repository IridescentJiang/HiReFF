# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
from typing import List
import cv2
import concurrent.futures
import os


def adjust_intrinsic_batch(K: torch.Tensor, orig_w: int, orig_h: int, target_w: int, target_h: int, type: str = "crop") -> torch.Tensor:
    """
    调整内参矩阵以适应缩放和裁剪操作

    Args:
        K: 原始内参矩阵 [N, 3, 3]
        orig_w: 原始图像宽度
        orig_h: 原始图像高度
        target_w: 目标图像宽度
        target_h: 目标图像高度
        type: 处理类型，"crop" 或 "pad"
            - "crop": 设置宽度为 target_w，并根据宽度缩放高度，保持纵横比
            - "pad": 保持原始纵横比，最大维度为 target_w，最小维度填充到正方形

    Returns:
        调整后的内参矩阵，形状为 [N, 3, 3]
    """
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise ValueError("Input K must be a tensor of shape [N, 3, 3]")

    if type == "crop":
        # 计算缩放比例（基于宽度）
        scale_w = target_w / orig_w

        # 实际缩放后的中间尺寸
        scaled_h = round(orig_h * scale_w / 14) * 14
        scale_h = scaled_h / orig_h

        # 计算裁剪量（上下裁剪）
        crop_amount = scaled_h - target_h
        crop_top = crop_amount / 2.0
    else:
        # 计算缩放比例
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        crop_top = 0

    # 调整内参
    adjusted_K = K.clone()
    adjusted_K[:, 0, 0] *= scale_w
    adjusted_K[:, 0, 2] *= scale_w
    adjusted_K[:, 1, 1] *= scale_h
    adjusted_K[:, 1, 2] = adjusted_K[:, 1, 2] * scale_h - crop_top

    return adjusted_K


def adjust_intrinsic(K, orig_w, orig_h, target_w, target_h, type="crop"):
    """
    调整内参矩阵以适应缩放和裁剪操作

    Args:
        K: 原始内参矩阵 [3x3]
        orig_w: 原始图像宽度
        orig_h: 原始图像高度
        target_w: 目标图像宽度
        target_h: 目标图像高度
        type: 处理类型，"crop" 或 "pad"
            - "crop": 设置宽度为 target_w，并根据宽度缩放高度，保持纵横比
            - "pad": 保持原始纵横比，最大维度为 target_w，最小维度填充到正方形

    Returns:
        调整后的内参矩阵
    """

    if type == "crop":
        # 计算缩放比例（基于宽度）
        scale_w = target_w / orig_w

        # 实际缩放后的中间尺寸
        scaled_h = round(orig_h * scale_w / 14) * 14  # 注意是 orig_h * scale_w
        scale_h = scaled_h / orig_h

        # 计算裁剪量（上下裁剪）
        crop_amount = scaled_h - target_h  # 垂直方向裁剪的总像素
        crop_top = crop_amount / 2.0  # 假设上下对称裁剪
    else:
        # 计算缩放比例
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        crop_top = 0

    # 调整内参
    adjusted_K = np.array([
        [K[0][0] * scale_w, 0, K[0][2] * scale_w],  # fx 和 cx 只需宽度比例
        [0, K[1][1] * scale_h, K[1][2] * scale_h - crop_top],  # fy 用宽度比例，cy 需减去顶部裁剪量
        [0, 0, 1]
    ])

    return adjusted_K


def normalize_extrinsic(extrinsic, rat=3):
    """归一化外参矩阵到[-1,1]范围"""
    # 分离旋转和平移
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # 旋转矩阵归一化（本身应在[-1,1]）
    # 平移向量归一化（假设最大平移量4米）
    t_normalized = t / rat

    return np.vstack([
        np.hstack([R, t_normalized.reshape(3, 1)]),
        [0, 0, 0, 1]
    ])


def convert_extrinsics_to_relative_tensor(
    extrinsics_list: List[torch.Tensor] | torch.Tensor,
    scale_factor: float = 1  # 明确缩放参数
) -> torch.Tensor:
    """
    将绝对坐标系外参矩阵（相机到世界的变换）转换为相对于第一个相机的变换矩阵

    参数:
        extrinsics_list: 相机到世界的外参矩阵列表 [4x4]

    返回:
        相对变换矩阵列表：每个矩阵表示从基准相机到当前相机的变换 [4x4]
    """
    if isinstance(extrinsics_list, torch.Tensor):
        extrinsics_tensor = extrinsics_list
        if extrinsics_tensor.ndim == 2:
            extrinsics_tensor = extrinsics_tensor.unsqueeze(0)
        if extrinsics_tensor.ndim != 3 or extrinsics_tensor.shape[-2:] != (4, 4):
            raise ValueError("extrinsics_list tensor must have shape [N, 4, 4] or [4, 4]")
    else:
        if len(extrinsics_list) == 0:
            return torch.empty((0, 3, 4), dtype=torch.float32)
        extrinsics_tensor = torch.stack(extrinsics_list, dim=0)

    if extrinsics_tensor.shape[0] == 0:
        return torch.empty((0, 3, 4), dtype=extrinsics_tensor.dtype, device=extrinsics_tensor.device)

    # 获取基准相机（世界->相机）
    base_inv = torch.linalg.inv(extrinsics_tensor[0])
    R_base = base_inv[:3, :3]
    t_base = base_inv[:3, 3]

    relative_list = []

    for ext in extrinsics_tensor:
        # 当前相机（世界->相机）
        cur_inv = torch.linalg.inv(ext)
        R_cur = cur_inv[:3, :3]
        t_cur = cur_inv[:3, 3]

        # 计算相对变换：δ = T_current_w2c * T_base_w2c^{-1}
        R_rel = R_cur @ R_base.T
        t_rel = t_cur - R_rel @ t_base

        # 构建矩阵并缩放平移量
        rel_mat = torch.eye(4, dtype=ext.dtype, device=ext.device)
        rel_mat[:3, :3] = R_rel
        rel_mat[:3, 3] = t_rel * scale_factor

        relative_list.append(rel_mat[:3, :4])

    return torch.stack(relative_list, dim=0)


def convert_extrinsics_to_relative(
        extrinsics_list: List[np.ndarray],
        scale_factor: float = 1  # 明确缩放参数
) -> List[np.ndarray]:
    """
    将绝对坐标系外参矩阵（相机到世界的变换）转换为相对于第一个相机的变换矩阵

    参数:
        extrinsics_list: 相机到世界的外参矩阵列表 [4x4]

    返回:
        相对变换矩阵列表：每个矩阵表示从基准相机到当前相机的变换 [4x4]
    """
    if not extrinsics_list:
        return []

    # 获取基准相机（世界->相机）
    base_inv = np.linalg.inv(extrinsics_list[0])
    R_base = base_inv[:3, :3]
    t_base = base_inv[:3, 3]

    relative_list = []

    for ext in extrinsics_list:
        # 当前相机（世界->相机）
        cur_inv = np.linalg.inv(ext)
        R_cur = cur_inv[:3, :3]
        t_cur = cur_inv[:3, 3]

        # 计算相对变换：δ = T_current_w2c * T_base_w2c^{-1}
        R_rel = R_cur @ R_base.T
        t_rel = t_cur - R_rel @ t_base

        # 构建矩阵并缩放平移量
        rel_mat = np.eye(4)
        rel_mat[:3, :3] = R_rel
        rel_mat[:3, 3] = t_rel * scale_factor

        relative_list.append(rel_mat[:3, :4])

    return relative_list


def load_masked_image(image_path, mask_path, threshold=128):
    """
    加载带有mask的图像，返回只保留mask区域的图像
    Args:
        image_path: 原始图像路径
        mask_path: mask图像路径
        threshold: mask二值化阈值(0-255)
    Returns:
        masked_image: PIL.Image对象，只保留mask区域的内容
    """
    # 加载原始图像和mask
    img = Image.open(image_path).convert('RGBA')  # 必须转换为RGBA格式
    mask = Image.open(mask_path).convert('L')  # 转换为灰度图

    if img.size != mask.size:
        mask = mask.resize(img.size, Image.BILINEAR)

    # 转换为numpy数组处理
    img_array = np.array(img)
    mask_array = np.array(mask)

    # 二值化mask（True表示保留区域）
    binary_mask = (mask_array > threshold)

    # 创建全透明背景（alpha=0）
    transparent_bg = np.zeros_like(img_array)
    transparent_bg[..., 3] = 0  # alpha通道全为0

    # 合并图像：保留mask区域的像素，其他区域透明
    masked_array = np.where(
        binary_mask[..., None],  # 增加维度以匹配RGBA
        img_array,  # True时保留原像素
        transparent_bg  # False时设为透明
    )

    return Image.fromarray(masked_array)


def gen_mask_image(images, masks, bg=torch.tensor([1.0, 1.0, 1.0])):
    # 统一掩码形状
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    bg = bg.view(1, 3, 1, 1).to(images.device)

    # 应用掩码
    return images * masks + bg * (1 - masks)


def load_and_preprocess_images(image_path_list, target_size=518, mask_path_list=None, Ks=None, Ds=None, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for i, image_path in enumerate(image_path_list):

        # Open image
        if mask_path_list:
            mask_path = mask_path_list[i]
            img = load_masked_image(image_path, mask_path)
        else:
            img = Image.open(image_path)

        if Ks is not None and Ds is not None:
            img_np = np.array(img)
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # 获取当前图像尺寸
            height, width = img_np.shape[:2]

            # 计算理想主点位置（图像中心）
            ideal_cx = width / 2.0
            ideal_cy = height / 2.0

            # 获取当前内参的实际主点
            current_cx = Ks[i][0, 2]
            current_cy = Ks[i][1, 2]

            # 计算需要平移的量（以补偿不在中心的偏移）
            tx = current_cx - ideal_cx
            ty = current_cy - ideal_cy

            # 构建新内参矩阵（修正主点偏移）
            new_K = Ks[i].copy()
            new_K[0, 2] = ideal_cx  # 水平主点设为图像中心
            new_K[1, 2] = ideal_cy  # 垂直主点设为图像中心

            # 执行畸变矫正并同时平移图像
            img_np = cv2.undistort(img_np, Ks[i], Ds[i], newCameraMatrix=new_K)

            img = Image.fromarray(img_np)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
