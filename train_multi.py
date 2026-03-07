import gc
import os
import time
import numpy
from math import sqrt

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from vggt.training.data.datasets.dna_rendering_multi import StreamingDnaRenderingDataset
from vggt.models.vggt import VGGT
from vggt.training.loss import camera_loss, RenderLoss, mask_loss, distill_geometry_loss, distill_depth_loss, \
    foreground_region_loss, distill_transformer_loss, depth_consist_loss
from vggt.rendering.render_image import render_images_infer, adding_mapped_poses, encode_poses, \
    save_rendered_images, adjust_transl, batch_render_images
from vggt.utils.visualization import vis_depth_map

torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 配置参数
class TrainingConfig:
    # 数据参数
    train_type = "frames"  # single_frame or frames

    if train_type == "single_frame":
        frame_numbers = ['000000']
    elif train_type == "frames":
        frame_numbers = None
    else:
        raise ValueError("train_type must be 'single_frame' or 'frames'.")

    if frame_numbers is not None and len(frame_numbers) <= 10:
        frame_numbers = frame_numbers * int(100 // len(frame_numbers))  # 重复以增加数据量

    data_root = "/home/user/dataset/data_example/"
    render_images_path = "render_images/dna_rendering_multi"

    gpus_num = torch.cuda.device_count()
    device = None

    batch_size = 1 * gpus_num
    num_workers = 4 * batch_size

    img_size = 518  # 1036
    sr_img_size = 518  # 2072
    lpip_patch_size = 518  # lpip监督patch大小
    random_patch = False  # 是否使用随机patch计算渲染损失，如果为False，则将全图插值到lpip_patch_size大小计算渲染损失
    random_patch_epoch = 1000  # 在第二阶段使用，表示在多少epoch后开始使用随机patch计算渲染损失

    multiview_supervise = 2  # 是否使用多视图监督，0表示不使用，N表示使用N对视图

    # 模型参数
    load_VGGT = True  # True 加载 VGGT 模型，False 加载 checkpoint
    model_name = "facebook/VGGT-1B"
    checkpoint = "checkpoints/finetuned_model.pt"
    only_load_mask_head = True  # 是否加载 mask_head

    # 渲染设置 gsplat或者mipsplat
    render_mode = "gsplat"
    # render_mode = "mipsplat"

    # 训练参数
    lr = 1e-5 * sqrt(gpus_num)
    epochs = 1000
    weight_decay = lr // 10
    warmup_epochs = 5 if load_VGGT else 1
    save_interval = epochs // 60
    val_interval = epochs // 60

    # 数据采样
    data_sample_rate = None  # None or n, 均匀抽取1/n数据进行训练

    # 模块激活情况
    camera_head_activate = False  # 是否激活 camera_head
    depth_head_activate = True  # 是否激活 depth_head
    gs_para_head_activate = True  # 是否激活 gs_para_head
    aggregator_activate = False  # 是否激活 aggregator
    mask_head_activate = True  # 是否激活 mask_head

    # Loss 激活情况
    camera_loss_activate = False  # 是否使用姿态编码损失
    distill_depth_loss_activate = True  # 是否使用深度蒸馏损失
    render_loss_activate = True  # 是否使用渲染损失
    mask_loss_activate = True  # 是否使用掩膜损失
    foreground_region_loss_activate = False  # 是否使用前景范围损失
    depth_consist_loss_activate = False  # 是否使用深度一致性损失
    distill_geo_loss_activate = False  # 是否使用几何蒸馏损失

    # 设备配置
    amp_enabled = True  # 自动混合精度
    grad_clip = 1.0  # 梯度裁剪


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def build_dataloaders_distributed(config, rank, world_size):
    """构建分布式训练和验证数据加载器"""
    # 训练集
    train_set = StreamingDnaRenderingDataset(
        data_root=config.data_root,
        min_frames=1,
        target_size=config.img_size,
        sr_target_size=config.sr_img_size,
        load_depth=False,
        load_mask=True,
        frame_numbers=config.frame_numbers,
        split="train",
        n_views_supervise=config.multiview_supervise
    )

    # 均匀抽取1/n数据
    if config.data_sample_rate is not None:
        total_size = len(train_set)
        subset_size = total_size // config.data_sample_rate
        indices = numpy.linspace(0, total_size - 1, subset_size, dtype=numpy.int64).tolist()

        # 创建子集
        train_set = Subset(train_set, indices)

    # 验证集
    val_set = StreamingDnaRenderingDataset(
        data_root=config.data_root,
        min_frames=1,
        target_size=config.img_size,
        sr_target_size=config.sr_img_size,
        load_depth=False,
        load_mask=True,
        frame_numbers=config.frame_numbers,
        split="val",
        n_views_supervise=config.multiview_supervise
    )

    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size // world_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size // world_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_sampler, val_sampler


def unfreeze_last_n_layers(module_list, n=3):
    """解冻模块列表中最后 n 层的参数"""
    if n == 0:
        return
    total_layers = len(module_list)
    for i in range(max(0, total_layers - n), total_layers):
        for param in module_list[i].parameters():
            param.requires_grad = True


def freeze_module(module, optimizer):
    """冻结模块参数并从优化器中移除"""
    # 禁用参数梯度
    for param in module.parameters():
        param.requires_grad = False

    # 从优化器中移除冻结参数
    groups_to_remove = []

    # 遍历所有参数组
    for group_idx, param_group in enumerate(optimizer.param_groups):
        # 创建新的参数列表（仅保留需要梯度的参数）
        new_params = []
        for param in param_group['params']:
            if param.requires_grad:
                new_params.append(param)

        if new_params:
            # 更新参数组
            param_group['params'] = new_params
        else:
            # 标记为空组，待移除
            groups_to_remove.append(group_idx)

    # 从后往前移除空组（避免索引变化）
    for group_idx in sorted(groups_to_remove, reverse=True):
        del optimizer.param_groups[group_idx]


def unfreeze_module(module, optimizer, lr_scale=1.0, param_group_idx=None):
    """解冻模块参数并添加到优化器"""
    # 启用参数梯度
    for param in module.parameters():
        param.requires_grad = True

    # 查找或创建参数组
    if param_group_idx is not None:
        # 使用现有参数组
        param_group = optimizer.param_groups[param_group_idx]
        # 添加参数
        param_group['params'] = list(param_group['params']) + list(module.parameters())
    else:
        # 创建新参数组
        new_group = {
            "params": list(module.parameters()),
            "lr": optimizer.param_groups[0]["lr"] * lr_scale
        }
        optimizer.param_groups.append(new_group)


def initialize_model(config, rank):
    """初始化模型和优化器"""
    # 在模型初始化后添加参数冻结逻辑
    model = VGGT.from_pretrained(config.model_name) if config.load_VGGT else VGGT.from_checkpoint(config.checkpoint)
    model = model.to(rank)

    if config.checkpoint and config.only_load_mask_head:
        checkpoint = torch.load(config.checkpoint, map_location='cpu')

        checkpoint_state = checkpoint["model_state"]

        mask_head_dict = {}
        for key, value in checkpoint_state.items():
            if key.startswith("mask_head."):
                # 移除"mask_head."前缀
                new_key = key.replace("mask_head.", "")
                mask_head_dict[new_key] = value

        # 加载整个mask_head
        model.mask_head.load_state_dict(mask_head_dict)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'aggregator'):
        if model.aggregator is not None:
            if config.aggregator_activate:
                for name, param in model.aggregator.named_parameters():
                    if any(m in name for m in ["global", "frame"]):
                        param.requires_grad = True

    # 解冻camera_head的所有参数
    if hasattr(model, 'camera_head'):
        if model.camera_head is not None:
            if config.camera_head_activate:
                for param in model.camera_head.parameters():
                    param.requires_grad = True

    if hasattr(model, 'activate_depth_head'):
        if model.activate_depth_head is not None:
            # 将 depth_head 的状态字典复制到 activate_depth_head
            if hasattr(model, 'depth_head') and config.load_VGGT == True:
                model.activate_depth_head.load_state_dict(model.depth_head.state_dict())

            # 解冻activate_depth_head的所有参数
            if config.depth_head_activate:
                for param in model.activate_depth_head.parameters():
                    param.requires_grad = True

    # if hasattr(model, 'point_offset_head'):
    #     if model.point_offset_head is not None:
    #         # 解冻point_offset_head的所有参数
    #         for param in model.point_offset_head.parameters():
    #             param.requires_grad = True

    if hasattr(model, 'gs_para_head'):
        if config.gs_para_head_activate:
            for param in model.gs_para_head.parameters():
                param.requires_grad = True

    # 解冻mask_head的所有参数
    if hasattr(model, 'mask_head'):
        if config.mask_head_activate:
            for param in model.mask_head.parameters():
                param.requires_grad = True

    lr_weights = {
        'default': 1.0,
        'aggregator': 1e-4,
        'camera': 1e-3,
        'activate_point_head': 1e-3,
        'activate_depth_head': 1e-1,
        'point_offset': 1.0,
        'gs_para': 1.0,
        'mask': 1.0
    }

    optimizer = AdamW(
        [
            {"params": model.aggregator.parameters(), "lr": config.lr * lr_weights["aggregator"]},
            {"params": model.camera_head.parameters(), "lr": config.lr * lr_weights["camera"]},
            {"params": model.activate_depth_head.parameters(), "lr": config.lr * lr_weights["activate_depth_head"]},
            # {"params": model.point_offset_head.parameters(), "lr": config.lr * lr_weights["point_offset"]},
            {"params": model.gs_para_head.parameters(), "lr": config.lr * lr_weights["gs_para"]},
            {"params": model.mask_head.parameters(), "lr": config.lr * lr_weights["mask"]},
        ],
        weight_decay=config.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs - config.warmup_epochs,
        eta_min=config.lr * 0.5
    )

    scaler = GradScaler(enabled=config.amp_enabled)
    return model, optimizer, scheduler, scaler


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self, device):
        super().__init__()
        self.camera_loss = camera_loss
        self.render_loss = RenderLoss(device)
        self.mask_loss = mask_loss
        self.distill_depth_loss = distill_depth_loss
        self.distill_geo_loss = distill_geometry_loss
        self.foreground_region_loss = foreground_region_loss
        self.distill_transformer_loss = distill_transformer_loss
        self.depth_consist_loss = depth_consist_loss

    def forward(self, preds, images, gs_depth, target, config):
        # 姿态编码损失
        if config.camera_loss_activate:
            camera_loss, camera_loss_dict = self.camera_loss(
                preds["pose_enc_pre"], target, with_auc=False, loss_type="huber"
            )
        else:
            camera_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 渲染损失
        if config.render_loss_activate:
            if target["sr_images"] is not None:
                target_images = target["sr_images"]
            else:
                target_images = target["images"]
            render_loss, render_loss_dict = self.render_loss.forward(
                images, preds["render_masks"], target_images, loss_type="perceptual+l1", mask_weight_factor=3.0,
                edge_weight_factor=2.0, random_patch=config.random_patch, patch_size=config.lpip_patch_size
            )
        else:
            render_loss = torch.tensor(0.0, device=preds["images"].device)

        # 掩膜损失
        if config.mask_loss_activate:
            mask_loss, mask_loss_dict = self.mask_loss(
                preds["masks"], target["masks"], loss_type="Dice+BCE"
            )
        else:
            mask_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 前景范围损失
        if config.foreground_region_loss_activate:
            foreground_region_loss, foreground_region_loss_dict = self.foreground_region_loss(
                images, target["masks_sp"], loss_type="Dice+BCE"
            )
        else:
            foreground_region_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 前景范围损失
        if config.depth_consist_loss_activate:
            depth_consist_loss, depth_consist_loss_dict = self.depth_consist_loss(
                gs_depth, target["masks_sp"], preds["depth"], loss_type="MSE"
            )
        else:
            depth_consist_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 蒸馏深度损失
        if config.distill_depth_loss_activate:
            distill_depth_loss, distill_depth_loss_dict = self.distill_depth_loss(
                preds["depth"], preds["masks"], preds["pseudo_label_depth"]
            )
            # 避免深度蒸馏损失过大影响训练稳定性
            if distill_depth_loss > 1e-3:
                distill_depth_loss = torch.tensor(0.0, device=preds["masks"].device)
        else:
            distill_depth_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 蒸馏几何损失
        if config.distill_geo_loss_activate:
            distill_geo_loss, distill_geo_loss_dict = self.distill_geo_loss(
                preds["world_points"], preds["masks"], preds["pseudo_label_points"], loss_type="chamfer+uniform"
            )
        else:
            distill_geo_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 总损失加权求和
        camera_w = 1e3
        render_w = 1.0
        mask_w = 5e-2
        foreground_region_w = 1e-2
        distill_depth_w = 1e1
        distill_geometry_w = 1e1
        depth_consist_w = 5e0

        total_loss = camera_loss * camera_w + render_loss * render_w + mask_loss * mask_w + \
                     foreground_region_loss * foreground_region_w + \
                     + distill_depth_loss * distill_depth_w + distill_geo_loss * distill_geometry_w + \
                     depth_consist_loss * depth_consist_w

        # 记录各项损失（不带梯度）
        loss_dict = {
            "camera": camera_loss.item() * camera_w,
            "render": render_loss.item() * render_w,
            "mask": mask_loss.item() * mask_w,
            "fore_reg": foreground_region_loss.item() * foreground_region_w,
            "dis_depth": distill_depth_loss.item() * distill_depth_w,
            # "dis_geo": distill_geo_loss.item() * distill_geometry_w,
            "depth_consist": depth_consist_loss.item() * depth_consist_w
        }

        return total_loss, loss_dict


def train_epoch(model, loader, optimizer, scaler, criterion, config, epoch, sampler, rank):
    """单个训练epoch"""
    model.train()
    sampler.set_epoch(epoch)  # 设置采样器epoch
    total_loss = 0.0
    metric_logger = {"camera": 0.0, "render": 0.0, "mask": 0.0, "fore_reg": 0.0, "dis_depth": 0.0,
                     "dis_geo": 0.0, "depth_consist": 0.0}  # 完整指标初始化

    for batch_idx, batch in enumerate(loader):

        # 数据加载到设备
        device = rank

        # 确保所有目标张量在正确设备
        targets = ({
            "extrinsics": batch["extrinsics"].to(device, non_blocking=True, dtype=torch.float32),
            "intrinsics": batch["intrinsics"].to(device, non_blocking=True, dtype=torch.float32),
            "images": batch["images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
            "sr_images": batch["sr_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
            "images_origin": batch["images"].to(device, non_blocking=True, dtype=torch.float32),
            "masks": batch["masks"].to(device, non_blocking=True, dtype=torch.float32),
            "bg_color": batch["bg_color"],
        })

        if config.multiview_supervise > 0:
            targets.update({
                "supervise_images": batch["supervise_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
                "supervise_masks": batch["supervise_masks"].to(device, non_blocking=True, dtype=torch.float32),
                "sr_supervise_images": batch["sr_supervise_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32)
            })

        images = targets["images_origin"]

        # 混合精度前向
        with autocast(enabled=config.amp_enabled):
            # preds = model(images, targets["extrinsics"], targets["intrinsics"])
            preds = model(images)

            if config.multiview_supervise > 0:
                adjusted_extrinsics, adjusted_supervise_extrinsics = adjust_transl(preds["pose_enc_pre"],
                                                                                   batch["extrinsics"],
                                                                                   batch["supervise_extrinsics"])
            else:
                adjusted_extrinsics, _ = adjust_transl(preds["pose_enc_pre"], batch["extrinsics"], None)

            targets["extrinsics"] = adjusted_extrinsics.to(device, non_blocking=True, dtype=torch.float32)

            if config.multiview_supervise > 0:
                combined_images = torch.cat([targets["images"], targets["supervise_images"]], dim=1)
                del targets["images"]
                targets["images"] = combined_images

                combined_sr_images = torch.cat([targets["sr_images"], targets["sr_supervise_images"]], dim=1)
                del targets["sr_images"]
                targets["sr_images"] = combined_sr_images

                combine_intrinsics = torch.cat([batch["intrinsics"], batch["supervise_intrinsics"]], dim=1)
                combine_extrinsics = torch.cat([adjusted_extrinsics, adjusted_supervise_extrinsics], dim=1)

                _, _, C, H, W = preds["images"].shape
                preds["pose_enc"] = encode_poses(combine_intrinsics, combine_extrinsics, H, W, device)

                preds["render_masks"] = torch.cat(
                    [preds["masks"], targets["supervise_masks"].permute(0, 1, 3, 4, 2).contiguous()], dim=1)

                targets["masks_sp"] = torch.cat([targets["masks"], targets["supervise_masks"].contiguous()], dim=1)

                del combine_intrinsics, combine_extrinsics
            else:
                _, _, C, H, W = preds["images"].shape
                preds["pose_enc"] = encode_poses(batch["intrinsics"], adjusted_extrinsics, H, W, device)
                targets["masks_sp"] = targets["masks"]
                preds["render_masks"] = preds["masks"]

            rendered_images, render_depths = batch_render_images(preds, wo_bg=True, sr_image_size=config.sr_img_size,
                                                                 render_depth=config.depth_consist_loss_activate,
                                                                 bg_color=targets["bg_color"], render_mode=config.render_mode)

            batch_loss, loss_components = criterion(preds, rendered_images, render_depths, targets, config)

        if batch_idx > config.random_patch_epoch:
            config.random_patch = True

        if rank == 0 and batch_idx % 20 == 0:
            if config.depth_consist_loss_activate:
                vis_depth = vis_depth_map(render_depths, preds["render_masks"])
                combined_output = torch.cat([rendered_images, vis_depth], dim=-1)
                save_rendered_images(combined_output.detach().cpu().clone(), f"{config.render_images_path}/train", epoch,
                                     batch_idx)
            else:
                save_rendered_images(rendered_images.detach().cpu().clone(), f"{config.render_images_path}/train", epoch,
                                     batch_idx)

        del images, preds, rendered_images, targets, batch, render_depths

        # 反向传播
        scaler.scale(batch_loss).backward()

        # 在反向传播后添加
        for param in model.parameters():
            if param.grad is not None:
                # 确保梯度是连续的
                param.grad = param.grad.contiguous()

        # 梯度裁剪
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),  # 只裁剪可训练参数
                max_norm=config.grad_clip,
                error_if_nonfinite=False  # 捕捉梯度异常
            )

        # 参数更新
        scaler.step(optimizer)
        scaler.update()

        # 梯度清零
        optimizer.zero_grad(set_to_none=True)

        value_loss = batch_loss.item()

        # 记录损失
        total_loss += value_loss
        for k in metric_logger:
            metric_logger[k] += loss_components.get(k, 0.0)  # 处理可能缺失的指标

        # 优化日志输出（只在主进程）
        if rank == 0 and batch_idx % 10 == 0:
            log_msg = f"Epoch {epoch} | Batch {batch_idx}/{len(loader)}"
            log_msg += f" | Loss: {value_loss:.4f}"
            if isinstance(loss_components, dict):
                log_msg += " | " + " | ".join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
            print(log_msg)

        del batch_loss

    # 计算平均损失
    num_batches = max(len(loader), 1)  # 处理空loader情况
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metric_logger.items()}

    del loader

    # # 强制垃圾回收
    # gc.collect()
    # torch.cuda.empty_cache()

    return avg_loss, avg_metrics


def validate(model, loader, criterion, config, epoch, rank):
    """验证循环"""
    model.eval()
    total_loss = 0.0
    metric_logger = {"camera": 0.0, "render": 0.0, "mask": 0.0, "fore_reg": 0.0, "dis_depth": 0.0,
                     "dis_geo": 0.0, "depth_consist": 0.0}  # 完整指标初始化

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # 数据加载到设备（强制设备转换）
            device = rank

            # 确保所有目标张量在正确设备
            targets = {
                "extrinsics": batch["extrinsics"].to(device, non_blocking=True, dtype=torch.float32),
                "intrinsics": batch["intrinsics"].to(device, non_blocking=True, dtype=torch.float32),
                "images": batch["images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
                "sr_images": batch["sr_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
                "images_origin": batch["images"].to(device, non_blocking=True, dtype=torch.float32),
                "masks": batch["masks"].to(device, non_blocking=True, dtype=torch.float32),
                "bg_color": batch["bg_color"],
            }

            if config.multiview_supervise > 0:
                targets.update({
                    "supervise_images": batch["supervise_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32),
                    "supervise_masks": batch["supervise_masks"].to(device, non_blocking=True, dtype=torch.float32),
                    "sr_supervise_images": batch["sr_supervise_images_wo_bg"].to(device, non_blocking=True, dtype=torch.float32)
                })

            images = targets["images_origin"]

            # 混合精度前向（保持与训练一致）
            with torch.cuda.amp.autocast(enabled=config.amp_enabled):
                preds = model(images)

                if config.multiview_supervise > 0:
                    adjusted_extrinsics, adjusted_supervise_extrinsics = adjust_transl(preds["pose_enc_pre"],
                                                                                       batch["extrinsics"],
                                                                                       batch["supervise_extrinsics"])
                else:
                    adjusted_extrinsics, _ = adjust_transl(preds["pose_enc_pre"], batch["extrinsics"], None)

                targets["extrinsics"] = adjusted_extrinsics.to(device, non_blocking=True, dtype=torch.float32)

                if config.multiview_supervise > 0:
                    combined_images = torch.cat([targets["images"], targets["supervise_images"]], dim=1)
                    del targets["images"]
                    targets["images"] = combined_images

                    combined_sr_images = torch.cat([targets["sr_images"], targets["sr_supervise_images"]], dim=1)
                    del targets["sr_images"]
                    targets["sr_images"] = combined_sr_images

                    combine_intrinsics = torch.cat([batch["intrinsics"], batch["supervise_intrinsics"]], dim=1)
                    combine_extrinsics = torch.cat([adjusted_extrinsics, adjusted_supervise_extrinsics], dim=1)

                    _, _, C, H, W = preds["images"].shape
                    preds["pose_enc"] = encode_poses(combine_intrinsics, combine_extrinsics, H, W, device)

                    preds["render_masks"] = torch.cat(
                        [preds["masks"], targets["supervise_masks"].permute(0, 1, 3, 4, 2).contiguous()], dim=1)

                    targets["masks_sp"] = torch.cat(
                        [targets["masks"], targets["supervise_masks"].contiguous()], dim=1)

                    del combine_intrinsics, combine_extrinsics
                else:
                    _, _, C, H, W = preds["images"].shape
                    preds["pose_enc"] = encode_poses(batch["intrinsics"], adjusted_extrinsics, H, W, device)
                    targets["masks_sp"] = targets["masks"]
                    preds["render_masks"] = preds["masks"]

                rendered_images, render_depths = batch_render_images(preds, wo_bg=True,
                                                                     sr_image_size=config.sr_img_size,
                                                                     render_depth=config.depth_consist_loss_activate,
                                                                     bg_color=targets["bg_color"], render_mode=config.render_mode)

                batch_loss, loss_components = criterion(preds, rendered_images, render_depths, targets, config)

            # 只在主进程保存渲染图像
            if rank == 0 and batch_idx % 20 == 0:
                if config.depth_consist_loss_activate:
                    vis_depth = vis_depth_map(render_depths, preds["render_masks"])
                    combined_output = torch.cat([rendered_images, vis_depth], dim=-1)
                    save_rendered_images(combined_output.detach().cpu().clone(), f"{config.render_images_path}/val",
                                         epoch, batch_idx)
                else:
                    save_rendered_images(rendered_images.detach().cpu().clone(), f"{config.render_images_path}/val",
                                         epoch, batch_idx)

            del images, preds, rendered_images, targets, batch, render_depths

            value_loss = batch_loss.item()

            # 累加损失和指标
            total_loss += value_loss
            for k in metric_logger:
                metric_logger[k] += loss_components.get(k, 0.0)  # 安全获取指标

            # 验证阶段日志（只在主进程）
            if rank == 0 and batch_idx % 20 == 0:
                log_msg = f"Val Batch {batch_idx}/{len(loader)}"
                log_msg += f" | Loss: {value_loss:.4f}"
                log_msg += " | " + " | ".join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
                print(log_msg)

    # 计算平均指标（防止零除）
    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metric_logger.items()}

    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """保存训练状态（修复优化器状态问题）"""
    # 获取当前可训练参数的ID集合
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}

    # 复制优化器状态
    optimizer_state = optimizer.state_dict()

    # 创建新的优化器状态
    new_optimizer_state = {
        "state": {},
        "param_groups": optimizer_state["param_groups"]
    }

    # 过滤状态：只保留当前可训练参数的状态
    for param_id, state in optimizer_state["state"].items():
        if param_id in trainable_ids:
            new_optimizer_state["state"][param_id] = state

    # 保存检查点
    state = {
        "epoch": epoch,
        "model_state": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": new_optimizer_state,
        "scheduler": scheduler.state_dict()
    }

    torch.save(state, save_path)
    print(f"检查点已保存至 {save_path}")


def main_worker(rank, world_size, config):
    """分布式训练工作函数"""
    # 初始化分布式环境
    setup(rank, world_size)

    # 只在主进程初始化TensorBoard
    if rank == 0:
        writer = SummaryWriter(log_dir='runs/multi_frame')

    # 构建数据加载器
    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders_distributed(config, rank, world_size)

    # 初始化模型
    model, optimizer, scheduler, scaler = initialize_model(config, rank)

    # 使用DDP包装模型
    model = DDP(model,
                device_ids=[rank],
                output_device=rank,
                gradient_as_bucket_view=True,
                bucket_cap_mb=50,
                find_unused_parameters=True)

    # 初始化损失函数
    criterion = MultiTaskLoss(device=rank)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    # 训练循环
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()

        # 学习率预热
        if epoch <= config.warmup_epochs:
            lr_scale = min(1.0, epoch / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr * lr_scale

        # 训练阶段
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, criterion, config, epoch, train_sampler, rank
        )

        # 验证阶段（只在主进程）
        if rank == 0 and epoch % config.val_interval == 0:
            val_loss, val_metrics = validate(model, val_loader, criterion, config, epoch, rank)
        else:
            val_loss = best_val_loss

        # 更新学习率
        if epoch > config.warmup_epochs:
            scheduler.step()

        # 保存最佳模型（只在主进程）
        if rank == 0 and epoch % config.val_interval == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    f"checkpoints/best_model_epoch_{epoch}_loss_{val_loss:.4f}.pt"
                )

        # 保存插值渲染结果（只在主进程）
        # if rank == 0 and epoch % config.render_interval == 0:
        #     images = render_images_infer(model, val_loader, config, rank, inter_view=0, type="All")
        #     save_rendered_images(images, save_path=f"{config.render_images_path}/inter_view", epoch=epoch)

        # 定期保存（只在主进程）
        if rank == 0 and epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                f"checkpoints/checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
            )

        # 打印日志（只在主进程）
        if rank == 0:
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch} Summary:")
            print(f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("Train Metrics:", " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            # 记录所有训练指标
            for metric_name, metric_value in train_metrics.items():
                writer.add_scalar(f'Train Metrics/{metric_name}', metric_value, epoch)

            if epoch % config.val_interval == 0:
                print("Val Metrics:", " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
                writer.add_scalars('Metrics', train_metrics | val_metrics, epoch)  # Python 3.9+
            print("-" * 50)

    # 清理分布式环境
    cleanup()

    # 关闭TensorBoard写入器
    if rank == 0:
        writer.close()


def main():
    config = TrainingConfig()
    print("Initializing...")

    # 获取GPU数量
    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 个GPU进行分布式训练")

    # 启动多进程训练
    mp.spawn(
        main_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
