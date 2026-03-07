# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# DIRTY VERSION, TO BE CLEANED UP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import threading
import weakref
from math import ceil, floor
from einops import rearrange
from torchvision.models import vgg16
from torchvision.ops import masks_to_boxes
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from functools import lru_cache
from vggt.training.lpips.lpips import LPIPS


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def camera_loss(pred_pose_enc_list, batch, with_auc=False, loss_type="l1", gamma=0.6,
                pose_encoding_type="absT_quaR_FoV", weight_T=1.0, weight_R=1.0, weight_fl=0.5):
    # Extract predicted and ground truth components
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic, gt_intrinsic, image_size_hw,
                                                    pose_encoding_type=pose_encoding_type).to(pred_pose_enc_list.device)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i]
        cur_gt_pose_encoding = gt_pose_encoding[i]

        torch.set_printoptions(
            precision=4,  # 显示4位小数
            sci_mode=False  # 禁用科学计数法（避免显示为1e-4等形式）
        )

        loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc.clone(), cur_gt_pose_encoding.clone(),
                                                           loss_type=loss_type)

        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_camera_dict = {
        "loss_camera": loss_camera.item(),
        "loss_T": loss_T.item(),
        "loss_R": loss_R.item(),
        "loss_fl": loss_fl.item()
    }

    if with_auc:
        with torch.no_grad():
            # compute auc
            last_pred_pose_enc = pred_pose_enc_list[-1]

            last_pred_extrinsic, _ = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw,
                                                                  pose_encoding_type=pose_encoding_type,
                                                                  build_intrinsics=False)

            rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(last_pred_extrinsic.float(), gt_extrinsic.float(),
                                                               gt_extrinsic.device)

            if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
                rel_rangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)
                rel_tangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)

            thresholds = [5, 15]
            for threshold in thresholds:
                loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
                loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

            _, normalized_histogram = calculate_auc(
                rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
            )

            auc_thresholds = [30, 10, 5, 3]
            for auc_threshold in auc_thresholds:
                cur_auc = torch.cumsum(
                    normalized_histogram[:auc_threshold], dim=0
                ).mean()
                loss_dict[f"Auc_{auc_threshold}"] = cur_auc

    return loss_camera, loss_camera_dict


def huber_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Huber loss function implementation for PyTorch

    Args:
        input (torch.Tensor): predicted values of shape (N, *)
        target (torch.Tensor): ground truth values of shape (N, *)
        delta (float, optional): threshold where loss transitions from L2 to L1. Default: 1.0
        reduction (str): reduction method ['none', 'mean', 'sum']. Default: 'mean'

    Returns:
        torch.Tensor: computed loss value
    """
    # 输入维度校验
    assert input.shape == target.shape, "Input and target shapes must match"
    assert reduction in ['none', 'mean', 'sum'], "Invalid reduction method"

    # 设备校验
    assert input.device == target.device, "Input and target must at the same device."

    # 计算绝对误差
    error = input - target
    abs_error = torch.abs(error)

    # 计算两种情况的损失
    quadratic = torch.min(abs_error, torch.tensor(delta, device=error.device))
    linear = abs_error - quadratic

    # 组合Huber损失
    loss = 0.5 * quadratic ** 2 + delta * linear

    # 根据reduction参数处理输出
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
        loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
        loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


@lru_cache(maxsize=None)
def get_perceptual_loss_model(device='cuda'):
    """进程安全的模型获取方法 (lru_cache保证单例)"""
    vgg = vgg16(pretrained=True).features[:5]
    vgg = vgg.half().to(device)
    vgg.requires_grad_(False)

    return vgg


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda', patch_size=256):
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target):
        # 合并两次前向传播为一次计算
        concatenated = torch.cat([input, target], dim=0)
        features = self.vgg(concatenated)

        # 分离特征结果
        vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

        # 确保梯度只来自input分支
        return self.l1_loss(vgg_input, vgg_target.detach())


class PerceptualLossPatch(nn.Module):
    def __init__(self, device='cuda', patch_size=256):
        """
        基于有效区域的感知损失（简化版）

        参数:
            device: 计算设备
            patch_size: 采样块大小
        """
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()
        self.patch_size = patch_size

    def find_valid_region(self, image):
        """
        检测图像中的有效像素区域（非零区域）

        返回:
            bbox: (x_min, y_min, x_max, y_max) 有效区域的边界框
        """
        # 创建二值掩码（非零像素）
        mask = (image != 0).any(dim=0)  # [H, W]

        # 如果没有有效像素，返回整个图像
        if not mask.any():
            return (0, 0, image.shape[1], image.shape[2]), -1

        # 找到有效像素的坐标
        coords = torch.nonzero(mask)

        # 计算边界框
        x_min = coords[:, 0].min().item()
        x_max = coords[:, 0].max().item()
        y_min = coords[:, 1].min().item()
        y_max = coords[:, 1].max().item()

        area_size = (x_max - x_min + 1) * (y_max - y_min + 1)

        return (x_min, y_min, x_max, y_max), area_size

    def sample_patch(self, image, target, bbox):
        """
        在有效区域的左上部分随机采样一个块

        参数:
            image: 输入图像 [C, H, W]
            bbox: 有效区域的边界框 (x_min, y_min, x_max, y_max)

        返回:
            patch: 采样的图像块 [C, patch_size, patch_size]
        """
        x_min, y_min, x_max, y_max = bbox

        # 计算有效区域的大小
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        # 计算可采样区域
        if bbox_width > self.patch_size:
            max_x = x_min + (bbox_width - self.patch_size)
            start_x = torch.randint(x_min, max_x + 1, (1,)).item()
            end_x = start_x + self.patch_size
        else:
            start_x = x_min
            end_x = start_x + bbox_width

        if bbox_height > self.patch_size:
            max_y = y_min + (bbox_height - self.patch_size)
            start_y = torch.randint(y_min, max_y + 1, (1,)).item()
            end_y = start_y + self.patch_size
        else:
            start_y = y_min
            end_y = start_y + bbox_height

        # 提取图像块
        patch_img = image[:, start_x:end_x, start_y:end_y]
        patch_target = target[:, start_x:end_x, start_y:end_y]

        return patch_img, patch_target

    def forward(self, input, target):
        """
        基于有效区域的感知损失（简化版）

        参数:
            input: 输入图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]

        返回:
            感知损失值
        """
        # 确保输入和目标形状相同
        assert input.shape == target.shape, "输入和目标形状不一致"

        total_loss = 0
        batch_size, C, H, W = input.shape

        # 处理每张图像
        for i in range(batch_size):
            input_img = input[i]  # [C, H, W]
            target_img = target[i]  # [C, H, W]

            # 查找有效区域
            input_bbox, area_size = self.find_valid_region(input_img)

            if area_size == -1:
                continue
            elif area_size > (H * W) / 2:
                # print("Inaccurate mask, skipping LPIP.")
                # print(input_bbox, area_size)
                continue
            elif area_size < 50:
                continue

            # 采样左上角块
            input_patch, target_patch = self.sample_patch(input_img, target_img, input_bbox)

            if input_patch.shape != target_patch.shape:
                print("Patch shapes do not match, skipping LPIP.")
                continue

            # 合并块
            concatenated = torch.cat([input_patch.unsqueeze(0), target_patch.unsqueeze(0)], dim=0)

            # 计算特征
            features = self.vgg(concatenated)

            # 分离特征
            vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

            # 计算当前块的损失
            patch_loss = self.l1_loss(vgg_input, vgg_target.detach())

            # 累积损失
            total_loss += patch_loss / batch_size

        return total_loss


class PerceptualLossPatch_fix(nn.Module):
    def __init__(self, device='cuda', patch_size=256, min_nonbg=100, max_attempts=20):
        """
        基于有效区域的感知损失（改进版）

        参数:
            device: 计算设备
            patch_size: 采样块大小
            min_nonzero: 最小非零像素数量
            max_attempts: 最大尝试次数
        """
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()
        self.patch_size = patch_size
        self.min_nonbg = min_nonbg
        self.max_attempts = max_attempts

    def sample_patch(self, image, target, bbox):
        x_min, y_min, x_max, y_max = bbox

        # 提取图像块
        patch_img = image[:, x_min:x_max, y_min:y_max]
        patch_target = target[:, x_min:x_max, y_min:y_max]

        return patch_img, patch_target

    def count_nonbg_pixels(self, patch):
        """
        计算图像块中的非背景像素数量

        参数:
            patch: 图像块 [C, H, W]

        返回:
            非背景像素数量
        """
        # 创建掩码：至少有一个通道不为1（背景）
        mask = (patch != 0).any(dim=0)  # [H, W]
        return mask.sum().item()

    def generate_random_patch_coords(self, H, W):
        """
        生成随机采样块的坐标

        参数:
            H: 图像高度
            W: 图像宽度

        返回:
            (x_min, y_min, x_max, y_max)
        """
        # 计算可能的采样位置数量
        n_H = max(1, int(H / (self.patch_size * 0.75)))
        n_W = max(1, int(W / (self.patch_size * 0.75)))

        # 随机选择位置
        random_n_H = torch.randint(0, n_H + 1, (1,)).item()
        random_n_W = torch.randint(0, n_W + 1, (1,)).item()

        # 计算坐标
        x_min = int(random_n_H * self.patch_size * 0.75)
        y_min = int(random_n_W * self.patch_size * 0.75)
        x_max = min(x_min + self.patch_size, H)
        y_max = min(y_min + self.patch_size, W)

        return (x_min, y_min, x_max, y_max)

    def forward(self, input, target):
        """
        基于有效区域的感知损失（改进版）

        参数:
            input: 输入图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]

        返回:
            感知损失值
        """
        # 确保输入和目标形状相同
        assert input.shape == target.shape, "输入和目标形状不一致"

        total_loss = 0
        batch_size, C, H, W = input.shape

        # 处理每张图像
        for i in range(batch_size):
            input_img = input[i]  # [C, H, W]
            target_img = target[i]  # [C, H, W]

            attempts = 0
            valid_patch_found = False

            # 尝试采样有效块
            while attempts < self.max_attempts and not valid_patch_found:
                # 生成随机块坐标
                patch_coords = self.generate_random_patch_coords(H, W)

                # 采样块
                input_patch, target_patch = self.sample_patch(input_img, target_img, patch_coords)

                # 检查块形状是否匹配
                if input_patch.shape != target_patch.shape:
                    attempts += 1
                    continue

                # 计算非背景像素数量
                nonbg_count = self.count_nonbg_pixels(input_patch)

                # 检查是否满足最小非背景像素要求
                if nonbg_count >= self.min_nonbg:
                    valid_patch_found = True
                else:
                    attempts += 1

            # 如果未找到有效块，使用最后一次采样的块
            if not valid_patch_found:
                print(f"Warning: No matching blocks found for image {i}, using last sample")

            # 合并块
            concatenated = torch.cat([input_patch.unsqueeze(0), target_patch.unsqueeze(0)], dim=0)

            # 计算特征
            features = self.vgg(concatenated)

            # 分离特征
            vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

            # 计算当前块的损失
            patch_loss = self.l1_loss(vgg_input, vgg_target.detach())

            # 累积损失
            total_loss += patch_loss / batch_size

        return total_loss


def extract_mask_from_image(image, background_value=1.0, tolerance=0.2):
    """
    从图像中提取掩膜（背景为指定值的区域），不需要梯度

    Args:
        image: 输入图像 [C, H, W]
        background_value: 背景值 (白色通常为1.0)
        tolerance: 容差范围

    Returns:
        二值掩膜 [1, H, W]，前景为1，背景为0（不需要梯度）
    """
    with torch.no_grad():
        # 计算每个像素与背景值的差异
        diff = torch.abs(image - background_value)

        # 对于RGB图像，取所有通道的最大差异
        if image.dim() == 3 and image.size(0) > 1:
            diff = torch.max(diff, dim=0, keepdim=True)[0]

        # 创建掩膜（背景区域为0，前景区域为1）
        mask = torch.where(diff > tolerance, 1.0, 0.0)

        # 确保掩膜是二维或三维的
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

    return mask.detach()


def get_largest_connected_component(mask, ksize=3):
    """
    高效GPU加速的最大连通域提取

    使用形态学操作+区域增长代替传统连通域分析
    适用于PyTorch GPU计算环境

    Args:
        mask: 输入掩膜 [1, H, W] (0/1值)
        ksize: 形态学操作核大小 (默认3)

    Returns:
        仅包含最大连通域的掩膜 [1, H, W]
    """
    with torch.no_grad():
        # 确保是单通道
        if mask.shape[0] > 1:
            mask = mask[:1]

        # 1. 形态学开运算去噪
        kernel = torch.ones(ksize, ksize, device=mask.device) / (ksize ** 2)
        kernel = kernel.view(1, 1, ksize, ksize)

        # 膨胀和腐蚀操作
        padding = ksize // 2
        smoothed_mask = F.conv2d(mask.unsqueeze(0), kernel, padding=padding).squeeze(0)
        smoothed_mask = (smoothed_mask > 0.5).float()

        # 2. 提取最亮像素作为种子点
        max_val = torch.max(smoothed_mask)
        seed_points = (smoothed_mask == max_val).float()

        # 3. GPU加速的区域增长算法
        current_mask = seed_points
        max_iter = 50  # 安全边界

        for _ in range(max_iter):
            # 扩散到邻接区域
            expanded = F.conv2d(current_mask.unsqueeze(0),
                                torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]],
                                             dtype=torch.float32, device=mask.device),
                                padding=1).squeeze(0)

            # 限制在原始二值掩膜内
            new_mask = torch.where(expanded > 0, smoothed_mask, torch.zeros_like(expanded))

            # 检查是否收敛
            if torch.all(new_mask == current_mask):
                break

            current_mask = new_mask

        # 4. 优化边界
        return current_mask * mask


def label_connected_components(binary_mask):
    """
    使用PyTorch实现连通域标记（不需要梯度）

    Args:
        binary_mask: 二值掩膜 [H, W]

    Returns:
        labeled: 标记的连通域 [H, W]
        num_labels: 连通域数量
    """
    from queue import Queue

    with torch.no_grad():
        # 使用4邻域连通性
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # 初始化
        h, w = binary_mask.shape
        labeled = torch.zeros((h, w), dtype=torch.int32)
        current_label = 1
        queue = Queue()

        # 遍历每个像素
        for i in range(h):
            for j in range(w):
                # 跳过背景或已标记
                if binary_mask[i, j] == 0 or labeled[i, j] > 0:
                    continue

                # 新连通域
                queue.put((i, j))
                labeled[i, j] = current_label

                # 广度优先搜索
                while not queue.empty():
                    x, y = queue.get()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < h and 0 <= ny < w and
                                binary_mask[nx, ny] == 1 and labeled[nx, ny] == 0):
                            labeled[nx, ny] = current_label
                            queue.put((nx, ny))

                current_label += 1

        return labeled, current_label - 1


def align_foreground_objects(pred_img, tgt_img,
                             background_value=1.0,
                             min_crop_size=128,
                             output_size=128,
                             pad_value=1.0,
                             large_bbox_threshold=0.5):
    """
    对齐前景对象（仅考虑最大连通域），但当对象边界框占图像超过阈值时跳过对齐

    注意：所有图像操作保持梯度，掩膜操作不需要梯度

    Args:
        pred_img: 预测图像 [C, H, W]（有梯度）
        tgt_img: 目标图像 [C, H, W]（有梯度）
        background_value: 背景值
        min_crop_size: 最小裁剪尺寸
        output_size: 输出尺寸
        pad_value: 填充值
        large_bbox_threshold: 大边界框面积判定阈值 (默认2/3=0.66)

    Returns:
        aligned_pred: 对齐后的预测图像 [C, output_size, output_size]（有梯度）
        aligned_tgt: 对齐后的目标图像 [C, output_size, output_size]（有梯度）
        combined_mask: 结合的前景掩膜 [1, output_size, output_size]（不需要梯度）
    """
    device = pred_img.device

    # 获取原始图像尺寸
    _, h, w = pred_img.shape
    img_area = h * w

    # 1. 提取掩膜并获取最大连通域（不需要梯度）
    with torch.no_grad():
        # 提取掩膜
        pred_mask = extract_mask_from_image(pred_img, background_value)
        tgt_mask = extract_mask_from_image(tgt_img, background_value)

        # 提取最大连通域
        pred_mask_cc = get_largest_connected_component(pred_mask)
        tgt_mask_cc = get_largest_connected_component(tgt_mask)

        # 如果掩膜全零（无前景），则使用整个图像
        if pred_mask_cc.sum() == 0:
            pred_mask = torch.ones_like(pred_mask)
        else:
            pred_mask = pred_mask_cc

        if tgt_mask_cc.sum() == 0:
            tgt_mask = torch.ones_like(tgt_mask)
        else:
            tgt_mask = tgt_mask_cc

        # 获取边界框
        pred_box = masks_to_boxes(pred_mask)
        tgt_box = masks_to_boxes(tgt_mask)

        # 计算边界框面积
        def bbox_area(bbox):
            if bbox.numel() > 0:
                x1, y1, x2, y2 = bbox[0]
                return (x2 - x1) * (y2 - y1)
            return 0.0

        pred_bbox_area = bbox_area(pred_box)
        tgt_bbox_area = bbox_area(tgt_box)

        # 判断是否为大型边界框
        is_large_pred = pred_bbox_area > large_bbox_threshold * img_area
        is_large_tgt = tgt_bbox_area > large_bbox_threshold * img_area

        # 如果任意一个边界框大于阈值，则跳过对齐
        if is_large_pred or is_large_tgt:
            # 直接调整原始图像大小（保持梯度）
            aligned_pred = adaptive_resize_and_pad(pred_img, output_size, pad_value)
            aligned_tgt = adaptive_resize_and_pad(tgt_img, output_size, pad_value)

            # 提取掩膜（不需要梯度）
            with torch.no_grad():
                aligned_pred_mask = extract_mask_from_image(aligned_pred, background_value)
                aligned_tgt_mask = extract_mask_from_image(aligned_tgt, background_value)
                combined_mask = torch.max(aligned_pred_mask, aligned_tgt_mask)

            return aligned_pred, aligned_tgt, combined_mask

        # 计算结合两个边界框的最小区域
        if pred_box.shape[1] > 0 and tgt_box.shape[1] > 0:
            combined_box = torch.cat([pred_box, tgt_box])
            min_coords = torch.min(combined_box[:, :2], dim=0)[0]
            max_coords = torch.max(combined_box[:, 2:], dim=0)[0]
        else:
            # 如果某个掩膜没有边界框，使用整个图像
            min_coords = torch.tensor([0, 0], device=device)
            max_coords = torch.tensor([w, h], device=device)

        # 确保最小尺寸
        w_size, h_size = max_coords - min_coords
        if w_size < min_crop_size or h_size < min_crop_size:
            center = (min_coords + max_coords) / 2
            half_size = max(min_crop_size / 2, max(w_size, h_size) / 2)
            min_coords = torch.clamp(center - half_size, 0)
            max_coords = torch.clamp(center + half_size, 0, pred_img.shape[-1])

        # 转换为整数坐标
        min_coords = torch.floor(min_coords).int()
        max_coords = torch.ceil(max_coords).int()

        # 确保有效的裁剪区域
        x1, y1 = min_coords[0].clamp(0), min_coords[1].clamp(0)
        x2, y2 = max_coords[0].clamp(0, w), max_coords[1].clamp(0, h)

        # 如果裁剪区域无效，使用整个图像
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h

    # 2. 裁剪图像（保持梯度）
    # 注意：切片操作保持原始梯度
    cropped_pred = pred_img[:, y1:y2, x1:x2]
    cropped_tgt = tgt_img[:, y1:y2, x1:x2]

    # 3. 调整并填充到统一大小（保持梯度）
    aligned_pred = adaptive_resize_and_pad(cropped_pred, output_size, pad_value)
    aligned_tgt = adaptive_resize_and_pad(cropped_tgt, output_size, pad_value)

    # 4. 创建结合的前景掩膜（不需要梯度）
    with torch.no_grad():
        aligned_pred_mask = extract_mask_from_image(aligned_pred, background_value)
        aligned_pred_mask = get_largest_connected_component(aligned_pred_mask)

        aligned_tgt_mask = extract_mask_from_image(aligned_tgt, background_value)
        aligned_tgt_mask = get_largest_connected_component(aligned_tgt_mask)

        combined_mask = torch.max(aligned_pred_mask, aligned_tgt_mask)

    return aligned_pred, aligned_tgt, combined_mask


def adaptive_resize_and_pad(image, output_size, pad_value=1.0):
    """
    保持宽高比的调整大小和填充（保持梯度）

    Args:
        image: 输入图像 [C, H, W]（有梯度）
        output_size: 目标输出大小
        pad_value: 填充值

    Returns:
        调整大小和填充后的图像 [C, output_size, output_size]（有梯度）
    """
    # 如果图像为空，直接返回
    if image.numel() == 0:
        return torch.full((image.shape[0], output_size, output_size), pad_value, device=image.device)

    # 获取原始尺寸
    _, h, w = image.shape

    # 计算缩放比例
    scale = min(output_size / h, output_size / w)

    # 计算新尺寸
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # 调整大小（保持梯度）
    resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                            mode='bilinear', align_corners=False).squeeze(0)

    # 计算填充
    pad_top = (output_size - new_h) // 2
    pad_bottom = output_size - new_h - pad_top
    pad_left = (output_size - new_w) // 2
    pad_right = output_size - new_w - pad_left

    # 应用填充（保持梯度）
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom),
                   value=pad_value)

    return padded


def pts_trans_loss(pts_trans, target, radio, lambda_l1=1.0):
    """
    计算点云转换损失
    利用target中相机外参的平移矩阵，计算点云转换损失
    """
    device = pts_trans.device

    transl = pts_trans[:, :, :3]
    extrinsics = target["extrinsics"]

    T = extrinsics[:, :, :3, 3]  # BxSx3
    target_transl = T * (radio - 1)

    l1_criterion = nn.L1Loss(reduction='none')
    l1_loss_map = l1_criterion(transl, target_transl)
    l1_loss = l1_loss_map.mean()  # 平均每个点的损失

    losses = {}
    losses['l1'] = l1_loss

    total_loss = torch.tensor(0.0, device=device)
    total_loss += lambda_l1 * l1_loss

    return total_loss, losses


class RenderLoss:
    """
    改进的渲染损失函数：对掩码区域应用更高权重
    """

    def __init__(self, device):
        self.lpips = LPIPS(net='vgg').half().to(device)
        self.lpips.requires_grad_(False)
        self.bg_color_cuda = torch.tensor([1.0, 1.0, 1.0], device=device).to(torch.float32)

    def compute_lpips_loss(self, image, gt_image):
        """
            image: [N, C, H ,W]
        """
        assert image.shape[2] == image.shape[3] and gt_image.shape[2] == gt_image.shape[3]
        lpips_loss = self.lpips.forward(
            image[:, [2, 1, 0], :, :],
            gt_image[:, [2, 1, 0], :, :],
            normalize=True
        ).mean()
        return lpips_loss

    def crop_image(self, gt_mask, patch_size, randomly, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """
        mask_uv = torch.argwhere(gt_mask > 0.)
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        if randomly and max_size > patch_size:
            random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
            random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        for image in args:
            cropped_image = self.bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype=image.dtype,
                                                                           device=image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

            if randomly and max_size > patch_size:
                cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
            else:
                cropped_image = F.interpolate(cropped_image[None], size=(patch_size, patch_size), mode='bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imshow('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imshow('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.waitKey(0)

        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]

    def forward(self, images, masks, target_images, loss_type="perceptual+l1",
                lambda_perceptual=0.1, lambda_l1=1.0, lambda_mse=0.5,
                mask_weight_factor=2.0, edge_weight_factor=3.0,
                min_crop_size=128, output_size=518,
                background_value=1.0, algining=False,
                random_patch=False, patch_size=256):
        """
        Args:
            images: 预测图像 [B*V, C, H, W]
            masks: 预测图像掩膜 [B, V, H, W, C]
            target: 包含目标图像的字典，需有 "images" 键
            loss_type: 损失组合方式 ("perceptual", "l1", "perceptual+l1")
            lambda_perceptual: 感知损失权重
            lambda_l1: L1损失权重
            mask_weight_factor: 掩码区域损失权重倍数
            edge_weight_factor: 边缘区域额外权重倍数

        Returns:
            组合后的损失值及各分量损失``
        """
        # 重组目标图像张量
        target_images = rearrange(target_images, "b v c h w -> (b v) c h w").contiguous()
        masks = rearrange(masks, "b v h w c -> (b v) c h w").contiguous()

        N, C, H, W = target_images.shape
        masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)

        device = images.device

        if algining:
            # 对每对图像进行对齐
            aligned_preds = []
            aligned_targets = []
            aligned_masks = []

            for i in range(len(images)):
                pred_img = images[i]
                tgt_img = target_images[i]

                # 对齐前景对象
                aligned_pred, aligned_tgt, combined_mask = align_foreground_objects(
                    pred_img, tgt_img,
                    background_value=background_value,
                    min_crop_size=min_crop_size,
                    output_size=output_size,
                    pad_value=background_value
                )

                aligned_preds.append(aligned_pred)
                aligned_targets.append(aligned_tgt)
                aligned_masks.append(combined_mask)

            # 转换为张量
            images = torch.stack(aligned_preds)
            target_images = torch.stack(aligned_targets)
            masks = torch.stack(aligned_masks)

        # 创建加权图：对掩码区域应用更高权重
        if mask_weight_factor > 1.0:
            # 标准化掩码形状 (B*V, 1, H, W)
            if masks.dim() > 2 and masks.size(1) == 3:  # RGB掩码
                masks = masks.mean(dim=1, keepdim=True)  # 转换为单通道

            # 基础权重图 (背景为1，前景为mask_weight_factor)
            base_weights = torch.ones_like(masks)  # 背景权重=1
            base_weights[masks > 0.5] = mask_weight_factor  # 部分权重区域

            # 边缘增强 (可选)
            if edge_weight_factor > mask_weight_factor:
                from torch.nn.functional import conv2d
                kernel = torch.tensor([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=torch.float32, device=device)
                kernel = kernel.view(1, 1, 3, 3) / kernel.sum()

                # 检测边界区域
                with torch.no_grad():
                    dilated = conv2d(masks, kernel, padding=1)
                    edges = (dilated > 0.2) & (dilated < 0.8)
                    base_weights[edges] = edge_weight_factor

            # 扩展权重图以匹配所有通道
            weights = base_weights.expand_as(images)  # 复制到RGB通道
        else:
            weights = torch.ones_like(images)  # 不应用额外权重

        # 初始化损失模块
        perceptual_criterion = PerceptualLossPatch(device=device, patch_size=patch_size)
        l1_criterion = nn.L1Loss(reduction='none')  # 设置为none以便加权

        # 计算各分量损失
        losses = {}

        # L1损失（像素级，可加权）
        if "l1" in loss_type:
            l1_loss_map = l1_criterion(images, target_images)
            weighted_l1 = (l1_loss_map * weights).mean()
            losses['l1'] = weighted_l1

        # MSE损失（像素级，可加权）
        if "mse" in loss_type:
            mse_loss_map = F.mse_loss(images, target_images, reduction='none')
            weighted_mse = (mse_loss_map * weights).mean()
            losses['mse'] = weighted_mse

        # 感知损失（特征级，通常不直接加权）
        if "perceptual" in loss_type:
            # 感知损失不直接加权，但可通过ROI提取
            if mask_weight_factor > 1.0:
                # 提取前景区域
                with torch.no_grad():
                    mask_roi = (masks > 0.5).float()

                # 分别计算前景和背景感知损失

                # lpip from animatable gaussian
                N = mask_roi.shape[0]
                image_list = []
                gt_image_list = []
                for i in range(N):
                    if mask_roi[i, 0].sum() < 100:
                        # 全背景，跳过
                        continue
                    image, gt_image = self.crop_image(mask_roi[i, 0], patch_size, random_patch, images[i],
                                                      target_images[i])
                    image_list.append(image)
                    gt_image_list.append(gt_image)
                image_list = torch.stack(image_list, dim=0)
                gt_image_list = torch.stack(gt_image_list, dim=0)
                fg_perceptual = self.compute_lpips_loss(image_list, gt_image_list)

                # fg_perceptual = perceptual_criterion(images * mask_roi, target_images * mask_roi)
                # bg_perceptual = perceptual_criterion(images * (1 - mask_roi), target_images * (1 - mask_roi))

                # 对前景应用更高权重
                # losses['perceptual'] = (fg_perceptual * mask_weight_factor + bg_perceptual) / 2

                losses['perceptual'] = fg_perceptual
            else:
                # losses['perceptual'] = self.compute_lpips_loss(images, target_images)
                losses['perceptual'] = perceptual_criterion(images, target_images)

        # 加权求和
        total_loss = torch.tensor(0.0, device=device)
        if 'perceptual' in losses:
            total_loss += lambda_perceptual * losses['perceptual']
        if 'l1' in losses:
            total_loss += lambda_l1 * losses['l1']
        if 'mse' in losses:
            total_loss += lambda_mse * losses['mse']  # 使用与L1相同的权重

        # 可选：添加前景存在性正则化，防止全黑预测
        if mask_weight_factor > 1.0:
            foreground_coverage = masks.mean()  # 平均覆盖度
            # 确保至少有一定比例的前景
            coverage_constraint = torch.maximum(torch.tensor(0.05, device=device) - foreground_coverage,
                                                torch.tensor(0.0, device=device))
            losses['coverage'] = coverage_constraint * mask_weight_factor * 10.0
            total_loss += losses['coverage']

        return total_loss, {k: v.item() for k, v in losses.items()}


def chamfer_distance(pred, target):
    """
    双向Chamfer距离实现
    """
    # 预测->目标

    dist_pred_target = torch.cdist(pred, target)  # [B, N, M]
    min_dist1, _ = dist_pred_target.min(2)  # [B, N]

    # 目标->预测
    min_dist2, _ = dist_pred_target.min(1)  # [B, M]

    return 0.5 * (min_dist1.mean() + min_dist2.mean())


def estimate_normals(points):
    """
    估计点云的法向量
    使用简单的k-NN方法来估计法向量
    """
    k = 16  # 最近邻数目
    B, N, _ = points.shape

    # 构建k-NN图
    dists = torch.cdist(points, points)  # [B, N, N]
    _, topk_idx = torch.topk(dists, k + 1, largest=False)  # [B, N, k+1]
    nn_points = torch.gather(points.unsqueeze(2), 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # [B, N, k+1, 3]

    # 平均法向量
    normals = nn_points[:, :, 1:] - nn_points[:, :, :1]  # [B, N, k, 3]
    normals = normals.mean(dim=2)  # [B, N, 3]

    return normals / (normals.norm(dim=-1, keepdim=True) + 1e-6)  # 单位化法向量


def normal_consistency_loss(pred, target):
    """
    预测点云与伪标签点云的法向量一致性
    """
    pred_normals = estimate_normals(pred)  # [B, N, 3]
    target_normals = estimate_normals(target)  # [B, M, 3]

    # 找到最近邻的法向量
    dist_matrix = torch.cdist(pred, target)
    _, nn_idx = dist_matrix.min(2)  # [B, N]

    # 收集对应的法向量
    nn_normals = torch.gather(
        target_normals,
        1,
        nn_idx.unsqueeze(-1).expand(-1, -1, 3)
    )

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(pred_normals, nn_normals, dim=-1)
    return 1.0 - cos_sim.mean()


def uniform_point_loss(points, k=16):
    """
    惩罚点密度不均匀分布
    """
    # 1. 构建k-NN图
    dists = torch.cdist(points, points)  # [B, N, N]

    # 2. 计算局部密度方差
    _, topk_idx = torch.topk(dists, k + 1, largest=False)  # [B, N, k+1]
    nn_dists = torch.gather(dists, 2, topk_idx)[..., 1:]  # 排除自身

    # 3. 计算局部区域密度方差
    density_vars = torch.var(nn_dists, dim=-1)  # [B, N]
    return density_vars.mean()


def distill_geometry_loss(points, masks, pseudo_label_points, loss_type="chamfer+uniform", lambda_chamfer=0.5,
                          lambda_uniform=0.2, downsample_ratio=0.01):
    """
    通过点云和伪标签点计算几何损失

    Args:
        points: 预测点 [B, V, H, W, C]
        target: 包含目标点云的字典，需有 "pseudo_label_points" 键。target["pseudo_label_points"]：[B, V, H, W, C]
        masks: 预测图像掩膜 [B, V, H, W, C]
        loss_type: 损失组合方式 ("chamfer", "normal", "uniform")

    Returns:
        组合后的损失值
    """

    # 重组目标点云张量
    target_points = rearrange(pseudo_label_points, "b v h w c -> (b v) (h w) c").contiguous()
    points = rearrange(points, "b v h w c -> (b v) (h w) c").contiguous()
    masks = rearrange(masks, "b v h w c -> (b v) (h w) c").contiguous()

    device = points.device

    # 仅保留有效点
    valid_mask = masks > 0.5  # 假设掩膜值大于0.5表示有效点
    valid_mask = valid_mask.expand_as(points)  # [B*V, H*W, C]
    points = points * valid_mask
    target_points = target_points * valid_mask

    # 降采样点云
    target_num = int(points.shape[1] * downsample_ratio)

    points = downsample_pointcloud(points, target_num, method='random')
    target_points = downsample_pointcloud(target_points, target_num, method='random')

    # 修改成使用chamfer+uniform Loss
    losses = {}
    if "chamfer" in loss_type:
        losses['chamfer'] = chamfer_distance(points, target_points)
    if "uniform" in loss_type:
        losses['uniform'] = uniform_point_loss(points)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    # 加权求和
    total_loss = torch.tensor(0.0, device=device)
    if 'chamfer' in losses:
        total_loss += lambda_chamfer * losses['chamfer']
    if 'uniform' in losses:
        total_loss += lambda_uniform * losses['uniform']

    return total_loss, {k: v.item() for k, v in losses.items()}


def depth_consist_loss(rendered_depth, masks, target_depth, loss_type):
    """
    计算深度一致性损失

    参数:
        rendered_depth: 渲染深度图 [B*V, 1, H, W]
        masks: 有效掩码 [B, V, 1, H, W]
        target_depth: 目标深度图 [B, V, 1, H, W]
        loss_type: 损失类型 ('MSE')

    返回:
        depth_loss: 深度一致性损失
    """
    B, V, H, W, C = target_depth.shape

    all_views = rendered_depth.shape[0] // B
    rendered_depth = rendered_depth.view(B, all_views, H, W, C)[:, :V]

    masks = rearrange(masks[:, :V], "b v c h w -> b v h w c").contiguous()
    valid_mask = (rendered_depth > 1e-5) & (masks > 0.5)

    if 'MSE' in loss_type:
        depth_loss = F.mse_loss(rendered_depth, target_depth, reduction='none')[valid_mask].mean()
    else:
        depth_loss = None

    loss = torch.nan_to_num(depth_loss, nan=0.0)
    return loss, {"mse_loss": loss.item()}


def distill_depth_loss(pred_depth, masks, target_depth):
    """
    深度蒸馏损失函数 - 仅使用MSE损失

    Args:
        pred_depth: 模型预测的深度图 [B, V, H, W]
        target_depth: 目标深度图（真值或伪标签）[B, V, H, W]
        masks: 有效区域掩码 [B, V, H, W]

    Returns:
        MSE损失值
    """

    # 应用有效掩码
    valid_mask = masks > 0.5  # 假设掩膜值大于0.5表示有效点
    valid_mask = valid_mask.expand_as(pred_depth)  # 确保维度匹配

    # 计算有效像素数量（避免除零）
    num_valid = torch.sum(valid_mask).clamp(min=1)

    # 计算掩码区域的均方误差 (MSE)
    diff = pred_depth - target_depth
    squared_diff = diff * diff
    masked_squared_diff = squared_diff * valid_mask.float()

    # 平均损失
    loss = torch.sum(masked_squared_diff) / num_valid

    return loss, {"mse_loss": loss.item()}


def distill_transformer_loss(feature_list, feature_list_label):
    """
    多特征蒸馏损失函数 - 标准MSE实现

    Args:
        feature_list: 模型预测的特征列表 [B, V, H, W] * N
        feature_list_label: 目标特征列表（真值或伪标签）[B, V, H, W] * N

    Returns:
        平均MSE损失值
    """
    total_loss = 0.0
    num_features = len(feature_list)

    # 确保特征列表长度一致
    assert len(feature_list) == len(feature_list_label), \
        "特征列表和目标列表长度不一致"

    # 计算每个特征对的损失
    losses = []
    for feature, feature_label in zip(feature_list, feature_list_label):
        # 检查特征形状是否匹配
        assert feature.shape == feature_label.shape, \
            f"特征形状不匹配: {feature.shape} vs {feature_label.shape}"

        # 计算有效元素数量
        num_elements = feature.numel()

        # 计算单个特征的MSE损失
        loss = F.mse_loss(feature, feature_label, reduction='sum') / num_elements
        losses.append(loss)
        total_loss += loss

    # 计算平均损失
    avg_loss = total_loss / num_features

    # 收集损失指标
    metrics = {"avg_mse_loss": avg_loss.item()}
    for i, loss in enumerate(losses):
        metrics[f"mse_loss_{i}"] = loss.item()

    return avg_loss, metrics


def mask_loss(masks, target_masks, loss_type="Dice+BCE", lambda_BCE=0.5, lambda_Dice=0.5):
    """
    组合损失函数

    Args:
        masks: 预测掩膜 [B, V, H, W, C]
        target: 包含目标图像的字典，需有 "mask" 键。target["masks"]：[B, V, C, H, W]
        loss_type: 损失组合方式 ("Dice", "BCE", "Dice+BCE")

    Returns:
        组合后的损失值
    """

    # 重组目标图像张量
    target_masks = rearrange(target_masks, "b v c h w -> (b v) c h w").contiguous()
    masks = rearrange(masks, "b v h w c -> (b v) c h w").contiguous()

    device = masks.device

    # 计算各分量损失
    losses = {}
    if "BCE" in loss_type:
        losses['BCE'] = nn.BCEWithLogitsLoss()(masks, target_masks)
    if "Dice" in loss_type:
        intersection = (masks * target_masks).sum()
        union = masks.sum() + target_masks.sum()
        epsilon = 1e-5
        losses['Dice'] = 1 - (2 * intersection + epsilon) / (union + epsilon)  # +1避免除零

    # 加权求和
    total_loss = torch.tensor(0.0, device=device)
    if 'BCE' in losses:
        total_loss += lambda_BCE * losses['BCE']
    if 'Dice' in losses:
        total_loss += lambda_Dice * losses['Dice']

    return total_loss, {k: v.item() for k, v in losses.items()}


def foreground_region_loss(rendered_images, target_masks, loss_type="Dice+BCE", lambda_BCE=0.5, lambda_Dice=0.5):
    """
    组合损失函数

    Args:
        rendered_images: 预测图像 [B*V, C, H, W]
        target_masks: 包含目标图像的字典，需有 "mask" 键。target["masks"]：[B, V, C, H, W]
        loss_type: 损失组合方式 ("Dice", "BCE", "Dice+BCE")

    Returns:
        组合后的损失值
    """

    target_masks = rearrange(target_masks, "b v c h w -> (b v) c h w").contiguous()
    N, C, H, W = rendered_images.shape
    target_masks = F.interpolate(target_masks, size=(H, W), mode='bilinear', align_corners=True)

    intensity = rendered_images.abs().mean(dim=1, keepdim=True)

    foreground_prob = torch.sigmoid((1 - intensity) * 1000) * 2 - 1  # 使用sigmoid近似阶跃函数

    device = rendered_images.device

    # 计算各分量损失
    losses = {}
    if "BCE" in loss_type:
        losses['BCE'] = nn.BCEWithLogitsLoss()(foreground_prob, target_masks)
    if "Dice" in loss_type:
        intersection = (foreground_prob * target_masks).sum()
        union = foreground_prob.sum() + target_masks.sum()
        epsilon = 1e-5
        losses['Dice'] = 1 - (2 * intersection + epsilon) / (union + epsilon)  # +1避免除零

    # 加权求和
    total_loss = torch.tensor(0.0, device=device)
    if 'BCE' in losses:
        total_loss += lambda_BCE * losses['BCE']
    if 'Dice' in losses:
        total_loss += lambda_Dice * losses['Dice']

    return total_loss, {k: v.item() for k, v in losses.items()}


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False,
               affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):
    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth = gt_depth[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                                   batch, normalize_pred=False, normalize_gt=False,
                                   gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss,
                                   valid_range=valid_range, postfix="_depth", disable_conf=disable_conf,
                                   all_mean=all_mean)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False,
               gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False,
               conf_loss_type="v1", **kwargs):
    """
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    """
    # gt_pts3d: B, S, H, W, 3
    gt_pts3d = batch['world_points']
    # valid_mask: B, S, H, W
    valid_mask = batch['point_masks']
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")

    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask,
                                  batch, normalize_pred=normalize_pred, gamma=gamma, alpha=alpha, affine_inv=affine_inv,
                                  gradient_loss=gradient_loss, valid_range=valid_range,
                                  camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean)

    return conf_loss_dict


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        # too small, just return
        return loss_tensor

    # Randomly sample if tensor is too large
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch, normalize_gt=True, normalize_pred=True, gamma=1.0,
              alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1,
              disable_conf=False, all_mean=False, postfix=""):
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(pts3d,
                                                                                                          gt_pts3d,
                                                                                                          valid_mask,
                                                                                                          gradient_loss=gradient_loss)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(
            first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[
            other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])

    if conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        if valid_range > 0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss", batch["seq_name"])

    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames

    # Verified that the loss is the same

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None):
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask],
                                      dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask],
                                       dim=-1)

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0  # hard coded
            alpha = 0.2  # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Direction vectors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def downsample_pointcloud(points, target_num, method='random'):
    """
    点云降采样

    参数:
        points: [B, N, 3] 输入点云
        target_num: 目标点数
        method: 'random', 'farthest_point' 或 'voxel'

    返回:
        降采样后的点云 [B, K, 3] 其中 K=target_num
    """
    B, N, D = points.shape

    # 随机采样 (最快)
    if method == 'random':
        indices = torch.randperm(N)[:target_num]
        return points[:, indices]

    # 最远点采样 (保持点云分布)
    elif method == 'farthest_point':
        downsampled = []
        for i in range(B):
            samples = []
            remaining = points[i].clone()

            # 随机选择第一个点
            first_idx = torch.randint(0, N, (1,))
            samples.append(points[i, first_idx])
            remaining = torch.cat([points[i, :first_idx], points[i, first_idx + 1:]], dim=0)

            # 迭代选择最远点
            for _ in range(1, target_num):
                dists = torch.cdist(samples[-1:], remaining)[0]  # [1, M]
                max_idx = torch.argmax(dists)
                samples.append(remaining[max_idx])
                remaining = torch.cat([remaining[:max_idx], remaining[max_idx + 1:]], dim=0)

            downsampled.append(torch.stack(samples))

        return torch.stack(downsampled)

    # 体素化采样 (最均匀)
    elif method == 'voxel':
        import numpy as np

        voxel_size = 0.05  # Define voxel size
        downsampled_points = []

        for i in range(B):
            # Convert points to NumPy array
            points_np = points[i].cpu().numpy()

            # Compute voxel grid indices
            voxel_indices = np.floor(points_np / voxel_size).astype(int)

            # Use unique voxel indices to downsample
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            downsampled_points.append(points_np[unique_indices[:target_num]])

        downsampled_points = torch.tensor(downsampled_points, device=points.device)
        return downsampled_points

    else:
        raise ValueError(f"不支持的采样方法: {method}")


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn=gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def torch_quantile(
        input: torch.Tensor,
        q: float | torch.Tensor,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        interpolation: str = "nearest",
        out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out
