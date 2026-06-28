import os
import sys
import time
import numpy
import math
from math import sqrt
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

from einops import rearrange

from vggt.models.vggt import VGGT
from vggt.training.train_config import TrainingConfig
from vggt.training.data.datasets.dna_rendering_npz import DnaRenderingDatasetNpz, dna_collate_fn
from vggt.training.data.datasets.zju_mocap_npz import ZjuMocapDatasetNpz
from vggt.training.data.datasets.mvhuman_npz import MvHumanDatasetNpz
from vggt.training.loss import camera_loss, mask_loss, distill_geometry_loss, distill_depth_loss, \
    foreground_region_loss, distill_transformer_loss, depth_consist_loss
from vggt.training.render_loss import RenderLoss
from vggt.rendering.render_image import encode_poses, \
    save_rendered_images, adjust_transl, batch_render_images_my
from vggt.utils.visualization import vis_depth_map

torch.backends.cudnn.benchmark = True


def setup(rank, world_size, master_port=20008):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def build_dataloaders_distributed(config, rank, world_size):
    """构建分布式训练和验证数据加载器"""
    def rebalance_datasets(datasets, dataset_names, mode: str, seed: int):
        if len(datasets) == 0:
            return datasets

        lengths = [len(ds) for ds in datasets]
        if any(length == 0 for length in lengths):
            raise ValueError(f"Cannot rebalance datasets with empty subset lengths: {lengths}")

        if mode == "downsample":
            target_len = min(lengths)
            target_counts = [target_len] * len(datasets)
        elif mode == "upsample":
            target_len = max(lengths)
            target_counts = [target_len] * len(datasets)
        elif mode == "weighted":
            if len(dataset_names) != len(datasets):
                raise ValueError("dataset_names and datasets length mismatch in weighted rebalance.")

            dataset_weights = getattr(config, "mix_dataset_weights", None)
            if not isinstance(dataset_weights, dict):
                raise ValueError("mix_dataset_weights must be a dict when mix_balance_mode='weighted'.")

            weights = [float(dataset_weights.get(name, 1.0)) for name in dataset_names]
            if any(weight <= 0 for weight in weights):
                raise ValueError(f"All mix_dataset_weights must be > 0, got: {dataset_weights}")

            weight_sum = sum(weights)
            target_total = getattr(config, "mix_weighted_target_total", None)
            if target_total is None:
                target_total = sum(lengths)
            target_total = int(target_total)
            if target_total <= 0:
                raise ValueError(f"mix_weighted_target_total must be > 0, got: {target_total}")

            raw_targets = [target_total * weight / weight_sum for weight in weights]
            target_counts = [max(1, int(round(raw_target))) for raw_target in raw_targets]

            diff = target_total - sum(target_counts)
            target_counts[-1] = max(1, target_counts[-1] + diff)

            if sum(target_counts) != target_total:
                adjust = target_total - sum(target_counts)
                for idx in range(len(target_counts)):
                    if adjust == 0:
                        break
                    if adjust > 0:
                        target_counts[idx] += 1
                        adjust -= 1
                    elif target_counts[idx] > 1:
                        target_counts[idx] -= 1
                        adjust += 1
        else:
            raise ValueError(f"mix_balance_mode must be 'downsample' or 'upsample' or 'weighted', got: {mode}")

        balanced = []
        for dataset_idx, ds in enumerate(datasets):
            ds_len = len(ds)
            target_len = target_counts[dataset_idx]
            if ds_len == target_len:
                balanced.append(ds)
                continue

            rng = numpy.random.default_rng(seed + dataset_idx)
            if ds_len > target_len:
                indices = rng.choice(ds_len, size=target_len, replace=False).tolist()
            else:
                indices = rng.choice(ds_len, size=target_len, replace=True).tolist()

            balanced.append(Subset(ds, indices))

        return balanced

    def build_dataset(dataset_name, split):
        if dataset_name == "dna":
            data_root = config.dna_data_root
            dataset_cls = DnaRenderingDatasetNpz
        elif dataset_name == "zju":
            data_root = config.zju_data_root
            dataset_cls = ZjuMocapDatasetNpz
        elif dataset_name == "mvhuman":
            data_root = config.mvhuman_data_root
            dataset_cls = MvHumanDatasetNpz
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Expected 'dna' or 'zju' or 'mvhuman'.")

        if getattr(config, "dataset_mode", "single") == "single" and hasattr(config, "data_root"):
            data_root = config.data_root

        return dataset_cls(
            data_root=data_root,
            min_frames=1,
            target_size=config.img_size,
            sr_target_size=config.sr_img_size,
            frame_numbers=config.frame_numbers,
            split=split,
            n_views_supervise=config.multiview_supervise,
        )

    if config.dataset_mode == "single":
        train_set = build_dataset(config.single_dataset, "train")
        val_set = build_dataset(config.single_dataset, "val")
    elif config.dataset_mode == "mix":
        if not hasattr(config, "mix_datasets") or len(config.mix_datasets) == 0:
            raise ValueError("mix_datasets must be a non-empty list when dataset_mode='mix'.")

        train_datasets = [build_dataset(name, "train") for name in config.mix_datasets]
        val_datasets = [build_dataset(name, "val") for name in config.mix_datasets]

        if getattr(config, "mix_balance", False):
            train_datasets = rebalance_datasets(
                train_datasets,
                config.mix_datasets,
                mode=config.mix_balance_mode,
                seed=config.mix_balance_seed,
            )
            if getattr(config, "mix_balance_val", False):
                val_datasets = rebalance_datasets(
                    val_datasets,
                    config.mix_datasets,
                    mode=config.mix_balance_mode,
                    seed=config.mix_balance_seed + 10000,
                )

        if rank == 0:
            train_sizes = [len(ds) for ds in train_datasets]
            val_sizes = [len(ds) for ds in val_datasets]
            print(f"[Data] mix train subset sizes: {train_sizes}")
            print(f"[Data] mix val subset sizes: {val_sizes}")

        train_set = ConcatDataset(train_datasets)
        val_set = ConcatDataset(val_datasets)
    else:
        raise ValueError(f"dataset_mode must be 'single' or 'mix', got: {config.dataset_mode}")

    # 均匀抽取1/n数据
    if config.data_sample_rate is not None:
        total_size = len(train_set)
        subset_size = total_size // config.data_sample_rate
        indices = numpy.linspace(0, total_size - 1, subset_size, dtype=numpy.int64).tolist()

        # 创建子集
        train_set = Subset(train_set, indices)

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
        persistent_workers=True,
        collate_fn=dna_collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size // world_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=dna_collate_fn,
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
    model = VGGT.from_pretrained(config.model_name, local_files_only=True) if config.load_VGGT else VGGT.from_checkpoint(config.checkpoint)
    model = model.to(rank)

    checkpoint = None
    if config.checkpoint and os.path.isfile(config.checkpoint):
        checkpoint = torch.load(config.checkpoint, map_location='cpu')

    if checkpoint is not None and config.only_load_mask_head:
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
            if hasattr(model, 'depth_head') and model.depth_head is not None and config.load_VGGT == True:
                model.activate_depth_head.load_state_dict(model.depth_head.state_dict())

            # 解冻activate_depth_head的所有参数
            if config.depth_head_activate:
                for param in model.activate_depth_head.parameters():
                    param.requires_grad = True

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
        'aggregator': 1e-1,
        'camera': 1e-2,
        # 'activate_point_head': 1.0,
        'activate_depth_head': 1.0,
        # 'point_offset': 1.0,
        'gs_para': 1.0,
        'mask': 1.0,
    }

    optimizer = AdamW(
        [
            {"params": model.aggregator.parameters(), "lr": config.lr * lr_weights["aggregator"], "name": "aggregator"},
            {"params": model.camera_head.parameters(), "lr": config.lr * lr_weights["camera"], "name": "camera"},
            {"params": model.activate_depth_head.parameters(), "lr": config.lr * lr_weights["activate_depth_head"], "name": "activate_depth_head"},
            {"params": model.gs_para_head.parameters(), "lr": config.lr * lr_weights["gs_para"], "name": "gs_para"},
            {"params": model.mask_head.parameters(), "lr": config.lr * lr_weights["mask"], "name": "mask"},
        ],
        weight_decay=config.weight_decay
    )

    def apply_group_lrs_from_config():
        for group in optimizer.param_groups:
            group_name = group.get("name", "default")
            lr_scale = lr_weights.get(group_name, lr_weights["default"])
            target_lr = config.lr * lr_scale
            group["lr"] = target_lr
            group["initial_lr"] = target_lr

    apply_group_lrs_from_config()

    T_total = config.epochs - config.warmup_epochs  # 总退火步数

    def lr_lambda(epoch):
        current_step = epoch - config.warmup_epochs
        # warmup阶段（你当前设为0，可直接跳过）
        if current_step < 0:
            return max(0.0, current_step / config.warmup_epochs) if config.warmup_epochs > 0 else 1.0
        # 超过总步数，保持最小LR
        if current_step >= T_total:
            return 0.5
        # 标准余弦退火：从1.0平滑降到0.5
        return 0.5 * (1.0 + math.cos(math.pi * current_step / T_total))

    # 5. 初始化Scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        last_epoch=-1  # 从头训练设为-1，断点续训设为对应epoch数
    )

    scaler = GradScaler(enabled=config.amp_enabled)

    resume_epoch = 0
    # 仅当 load_VGGT=False（即从checkpoint构建模型）时恢复训练状态
    should_resume_training_state = not bool(getattr(config, "load_VGGT", True))

    if isinstance(checkpoint, dict) and should_resume_training_state:
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if rank == 0:
                    print(f"Loaded optimizer state from checkpoint: {config.checkpoint}")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: failed to load optimizer state from {config.checkpoint}: {e}")
        elif rank == 0:
            print(f"Optimizer state not found in checkpoint: {config.checkpoint}")

        if "scheduler" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                if rank == 0:
                    print(f"Loaded scheduler state from checkpoint: {config.checkpoint}")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: failed to load scheduler state from {config.checkpoint}: {e}")
        elif rank == 0:
            print(f"Scheduler state not found in checkpoint: {config.checkpoint}")

        if getattr(config, "reset_optimizer_lrs_on_resume", True):
            apply_group_lrs_from_config()
            scheduler.base_lrs = [group["lr"] for group in optimizer.param_groups]
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
            if rank == 0:
                print("Reset optimizer/scheduler group lrs from current config after resume.")

        if "epoch" in checkpoint:
            resume_epoch = int(checkpoint["epoch"])
    elif isinstance(checkpoint, dict) and rank == 0 and config.checkpoint:
        print(
            f"Skip training-state resume from checkpoint: {config.checkpoint} "
            f"(load_VGGT={getattr(config, 'load_VGGT', True)})"
        )

    if rank == 0:
        for idx, group in enumerate(optimizer.param_groups):
            group_name = group.get("name", f"group_{idx}")
            print(f"[LR] {group_name}: {group['lr']:.3e}")

    return model, optimizer, scheduler, scaler, resume_epoch


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.camera_loss = camera_loss
        self.render_loss = RenderLoss(lambda_perceptual=0.2, lambda_l1=1.0, mask_weight_factor=3.0, edge_weight_factor=1.0)
        self.mask_loss = mask_loss
        self.distill_depth_loss = distill_depth_loss
        self.distill_geo_loss = distill_geometry_loss
        self.foreground_region_loss = foreground_region_loss
        self.distill_transformer_loss = distill_transformer_loss
        self.depth_consist_loss = depth_consist_loss

    def forward(self, preds, images, gs_depth, target, config: TrainingConfig):
        # 姿态编码损失
        if config.camera_loss_activate:
            camera_loss, camera_loss_dict = self.camera_loss(
                preds["pose_enc_pre"], target, with_auc=False, loss_type="huber"
            )
        else:
            camera_loss = torch.tensor(0.0, device=preds["masks"].device)

        camera_loss_max = getattr(config, "camera_loss_max_for_backward", None)
        if camera_loss.item() > camera_loss_max:  # 避免姿态编码损失过大影响训练稳定性
            return camera_loss * 0.0, {"camera": 0.0}

        # 渲染损失
        if config.render_loss_activate:
            if target["sr_images"] is not None:
                target_images = target["sr_images"]
            else:
                target_images = target["images"]

            combined_masks = torch.cat([target["sr_masks"], target["sr_supervise_masks"]], dim=1)
            render_loss, _ = self.render_loss(images, combined_masks, target_images)
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

        # 蒸馏深度损失
        if config.distill_depth_loss_activate:
            distill_depth_loss, distill_depth_loss_dict = self.distill_depth_loss(
                preds["depth"], preds["masks"], preds["pseudo_label_depth"]
            )
        else:
            distill_depth_loss = torch.tensor(0.0, device=preds["masks"].device)

        if config.depth_consist_loss_activate:
            depth_consist_loss, _ = self.depth_consist_loss(
                gs_depth, target["masks_sp"], preds["depth"], loss_type="MSE"
            )
        else:
            depth_consist_loss = torch.tensor(0.0, device=preds["masks"].device)

        # 蒸馏几何损失
        if config.distill_geo_loss_activate:
            distill_geo_loss, distill_geo_loss_dict = self.distill_geo_loss(
                preds["world_points"], preds["masks"], preds["pseudo_label_points"], loss_type="chamfer+uniform"
            )
        else:
            distill_geo_loss = torch.tensor(0.0, device=preds["masks"].device)
        
        if config.color_dist_loss_activate:
            color_dist = 0.0
            for pc in preds["flat_gs"]:
                color_dist += torch.nn.functional.l1_loss(pc["colors"].squeeze(-1), pc["gt_colors"])
        else:
            color_dist = torch.tensor(0.0, device=preds["masks"].device)

        # 总损失加权求和
            
        render_w = 5.0
        camera_w = 1e2
        mask_w = 1e-1
        foreground_region_w = 1e-2
        distill_depth_w = 1e1
        distill_geometry_w = 1e1
        depth_consist_w = 5e0
        color_dist_w = 5e-1

        total_loss = camera_loss * camera_w + render_loss * render_w + mask_loss * mask_w + \
                     foreground_region_loss * foreground_region_w + \
                     + distill_depth_loss * distill_depth_w + distill_geo_loss * distill_geometry_w + \
                     depth_consist_loss * depth_consist_w + color_dist * color_dist_w

        # 记录各项损失（不带梯度）
        loss_dict = {
            "camera": camera_loss.item() * camera_w,
            "render": render_loss.item() * render_w,
            "mask": mask_loss.item() * mask_w,
            "fore_reg": foreground_region_loss.item() * foreground_region_w,
            "dis_depth": distill_depth_loss.item() * distill_depth_w,
            "depth_consist": depth_consist_loss.item() * depth_consist_w, 
            "color_dist": color_dist.item() * color_dist_w
        }

        return total_loss, loss_dict


def preprocess_data(data, lr_size, rank):
    # 数据加载到设备
    device = rank
    
    n_batch = len(data["image_bytes"])
    n_view = len(data["image_bytes"][0])
    
    bg_color = torch.rand(n_batch, n_view, 3, 1, 1, dtype=torch.float32, device=rank)

    sr_image_list = []
    for image_bytes in data["image_bytes"]:
        for image_byte in image_bytes:
            decoded = torchvision.io.decode_jpeg(
                image_byte.detach().to(device="cpu", dtype=torch.uint8).contiguous().flatten(),
                mode=torchvision.io.ImageReadMode.RGB,
                device=torch.device(f"cuda:{device}"),
            ).float() / 255.0
            sr_image_list.append(decoded)
    sr_images = torch.stack(sr_image_list, dim=0)  # (batch_size, n_views, 3, H, W)

    sr_masks = rearrange(data["masks"].to(device), "b v c h w -> (b v) c h w")
    sr_images_wo_bg = sr_images * sr_masks  # (n_views, 3, H, W)

    masks = torch.nn.functional.interpolate(
        sr_masks, size=(lr_size, lr_size), mode='bilinear'
    )
    masks = rearrange(masks, "(b v) c h w -> b v c h w", b=n_batch)
    
    images = torch.nn.functional.interpolate(
        sr_images, size=(lr_size, lr_size), mode='area'
    )
    images = rearrange(images, "(b v) c h w -> b v c h w", b=n_batch)
    
    images_wo_bg = images * masks + (1.0 - masks) * bg_color
    
    sr_images = rearrange(sr_images, "(b v) c h w -> b v c h w", b=n_batch)
    sr_masks = rearrange(sr_masks, "(b v) c h w -> b v c h w", b=n_batch)
    sr_images_wo_bg = sr_images * sr_masks + (1.0 - sr_masks) * bg_color
    
    targets = ({
        "extrinsics": data["extrinsics"].to(device),
        "intrinsics": data["intrinsics"].to(device),
        "images": images_wo_bg,
        "sr_images": sr_images_wo_bg,
        "sr_masks": sr_masks, 
        "images_origin": images,
        "masks": masks,
    })
            
    if "supervise_image_bytes" in data:
        n_sv_view = len(data["supervise_image_bytes"][0])
        sr_supervise_masks = data["supervise_masks"].to(device)

        sr_supervise_image_list = []
        for supervise_image_bytes in data["supervise_image_bytes"]:
            for image_bytes in supervise_image_bytes:
                decoded = torchvision.io.decode_jpeg(
                    image_bytes.detach().to(device="cpu", dtype=torch.uint8).contiguous().flatten(),
                    mode=torchvision.io.ImageReadMode.RGB,
                    device=torch.device(f"cuda:{device}"),
                ).float() / 255.0

                sr_supervise_image_list.append(decoded)

        sr_supervise_images = rearrange(torch.stack(sr_supervise_image_list, dim=0), "(b v) c h w -> b v c h w", b=n_batch)  # (batch_size, n_views, 3, H, W)
        
        sv_bg_color = torch.rand(n_batch, n_sv_view, 3, 1, 1, dtype=torch.float32, device=rank)
        
        sr_supervise_images_wo_bg = sr_supervise_images * sr_supervise_masks + sv_bg_color * (1.0 - sr_supervise_masks)
        supervise_images_wo_bg = torch.nn.functional.interpolate(
            rearrange(sr_supervise_images_wo_bg, "b v c h w -> (b v) c h w"), size=(lr_size, lr_size), mode='area'
        )  # (n_views, 3, 518, 518)
        
        supervise_masks = torch.nn.functional.interpolate(
            rearrange(sr_supervise_masks, "b v c h w -> (b v) c h w"), size=(lr_size, lr_size), mode='bilinear'
        )  # (n_views, 3, 518, 518)

        targets.update({
            "supervise_intrinsics": data["supervise_intrinsics"].to(device),
            "supervise_extrinsics": data["supervise_extrinsics"].to(device),
            "supervise_images": rearrange(supervise_images_wo_bg, "(b v) c h w -> b v c h w", b=n_batch),
            "supervise_masks": rearrange(supervise_masks, "(b v) c h w -> b v c h w", b=n_batch),
            "sr_supervise_images": sr_supervise_images_wo_bg, 
            "sr_supervise_masks": sr_supervise_masks, 
        })
        
        bg_color = torch.cat([bg_color, sv_bg_color], dim=1).squeeze(-1).squeeze(-1)
        
    targets["bg_color"] = bg_color
    
    return targets


def forward_render_and_loss(model, targets, criterion, config, epoch, device, with_model_grad=True):
    """统一 train/val 的前向、渲染与损失计算流程，避免逻辑漂移"""
    use_gt_mask = epoch <= config.use_gt_mask_until_epoch

    forward_context = torch.enable_grad() if with_model_grad else torch.no_grad()

    with forward_context:
        with autocast(device_type="cuda", enabled=config.amp_enabled):
            if targets["sr_images"].shape[-2:] != (config.sr_img_size, config.sr_img_size):
                bsz, n_views = targets["sr_images"].shape[:2]
                images_hr = torch.nn.functional.interpolate(
                    targets["sr_images"].flatten(0, 1),
                    size=(config.sr_img_size, config.sr_img_size),
                    mode='area'
                )
                images_hr = rearrange(images_hr, "(b v) c h w -> b v c h w", b=bsz, v=n_views)
                
                
            else:
                images_hr = targets["sr_images"]

            preds = model(
                targets["images_origin"],
                images_hr,
                mask_gaussian=True,
                gt_masks=targets["masks"],
                use_gt_mask=use_gt_mask
            )

            if config.multiview_supervise > 0:
                adjusted_extrinsics, adjusted_supervise_extrinsics = adjust_transl(
                    preds["pose_enc_pre"], targets["extrinsics"], targets["supervise_extrinsics"]
                )
            else:
                adjusted_extrinsics, _ = adjust_transl(preds["pose_enc_pre"], targets["extrinsics"], None)

            targets["extrinsics"] = adjusted_extrinsics.to(device, dtype=torch.float32)

            if config.multiview_supervise > 0:
                targets["images"] = torch.cat([targets["images"], targets["supervise_images"]], dim=1)
                targets["sr_images"] = torch.cat([targets["sr_images"], targets["sr_supervise_images"]], dim=1)

                combine_intrinsics = torch.cat([targets["intrinsics"], targets["supervise_intrinsics"]], dim=1)
                combine_extrinsics = torch.cat([adjusted_extrinsics, adjusted_supervise_extrinsics], dim=1)

                _, _, _, H, W = preds["images"].shape
                preds["pose_enc"] = encode_poses(combine_intrinsics, combine_extrinsics, H, W, device)
                preds["render_masks"] = torch.cat(
                    [preds["masks"], targets["supervise_masks"].permute(0, 1, 3, 4, 2).contiguous()], dim=1
                )
                targets["masks_sp"] = torch.cat([targets["masks"], targets["supervise_masks"].contiguous()], dim=1)
            else:
                _, _, _, H, W = preds["images"].shape
                preds["pose_enc"] = encode_poses(targets["intrinsics"], adjusted_extrinsics, H, W, device)
                targets["masks_sp"] = targets["masks"]
                preds["render_masks"] = preds["masks"]

            rendered_images, render_depths = batch_render_images_my(
                preds,
                wo_bg=True,
                sr_image_size=config.sr_img_size,
                render_depth=config.depth_consist_loss_activate,
                bg_color=targets["bg_color"]
            )

    loss_images = rendered_images
    loss_context = torch.enable_grad() if with_model_grad or (config.render_loss_activate and not with_model_grad) else torch.no_grad()
    if config.render_loss_activate and not with_model_grad:
        loss_images = rendered_images.detach().requires_grad_(True)

    with loss_context:
        with autocast(device_type="cuda", enabled=config.amp_enabled):
            batch_loss, loss_components = criterion(preds, loss_images, render_depths, targets, config)

    return batch_loss, loss_components, rendered_images, render_depths, preds, targets


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, config, epoch, sampler, rank, writer=None):
    """单个训练epoch"""
    model.train()
    sampler.set_epoch(epoch)  # 设置采样器epoch
    total_loss = 0.0
    metric_logger = {"camera": 0.0, "render": 0.0, "mask": 0.0, "fore_reg": 0.0, "dis_depth": 0.0,
                     "dis_geo": 0.0, "depth_consist": 0.0,  "color_dist": 0.0}  # 完整指标初始化

    for batch_idx, batch in tqdm(enumerate(loader)):

        # 数据加载到设备
        device = rank
        targets = preprocess_data(batch, config.img_size, rank)
        batch_loss, loss_components, rendered_images, render_depths, preds, targets = forward_render_and_loss(
            model, targets, criterion, config, epoch, device
        )

        if batch_idx > config.random_patch_epoch:
            config.random_patch = True

        if rank == 0 and batch_idx % 300 == 0:
            if config.depth_consist_loss_activate:
                vis_depth = vis_depth_map(render_depths, preds["render_masks"])
                combined_output = torch.cat([rendered_images, vis_depth], dim=-1)
                save_rendered_images(combined_output.detach().cpu().clone(), f"{config.render_images_path}/train", epoch,
                                     batch_idx)
            else:
                target_sr_images = targets["sr_images"][0]
                if target_sr_images.shape[-2:] != rendered_images.shape[-2:]:
                    target_sr_images = torch.nn.functional.interpolate(
                        target_sr_images,
                        size=rendered_images.shape[-2:],
                        mode='area'
                    )
                combined_output = torch.cat([rendered_images, target_sr_images], dim=-1)
                save_rendered_images(
                    combined_output.detach().cpu().clone(),
                    f"{config.render_images_path}/train",
                    epoch,
                    batch_idx,
                )

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

        # 每个iter写入TensorBoard（仅主进程）
        if rank == 0 and writer is not None:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar('Iter/Loss/total', value_loss, global_step)
            writer.add_scalar('Iter/Learning Rate', optimizer.param_groups[0]['lr'], global_step)
            for metric_name, metric_value in loss_components.items():
                writer.add_scalar(f'Iter/Loss/{metric_name}', metric_value, global_step)

        # 按iter保存checkpoint（仅主进程）
        if (
            rank == 0
            and getattr(config, "iter_save_interval", 0) > 0
            and (batch_idx + 1) % config.iter_save_interval == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                f"checkpoints/checkpoint_epoch_{epoch}_iter_{batch_idx + 1}_loss_{value_loss:.4f}.pt",
            )

        # 优化日志输出（只在主进程）
        if rank == 0 and batch_idx % 10 == 0:
            log_msg = f"Epoch {epoch} | Batch {batch_idx}/{len(loader)}"
            log_msg += f" | Loss: {value_loss:.4f}"
            if isinstance(loss_components, dict):
                log_msg += " | " + " | ".join([f"{k}:{v:.4f}" for k, v in loss_components.items()])
            print(log_msg)

        # del batch_loss

    # 计算平均损失
    num_batches = max(len(loader), 1)  # 处理空loader情况
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in metric_logger.items()}

    return avg_loss, avg_metrics


def validate(model, loader, criterion, config, epoch, rank, sampler=None):
    """验证循环"""
    model.eval()
    total_loss = 0.0
    metric_logger = {"camera": 0.0, "render": 0.0, "mask": 0.0, "fore_reg": 0.0, "dis_depth": 0.0,
                     "dis_geo": 0.0, "depth_consist": 0.0}  # 完整指标初始化

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        # 数据加载到设备（强制设备转换）
        device = rank

        targets = preprocess_data(batch, config.img_size, rank)

        batch_loss, loss_components, rendered_images, render_depths, preds, targets = forward_render_and_loss(
            model, targets, criterion, config, epoch, device, with_model_grad=False
        )

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

        del preds, rendered_images, targets, batch, render_depths

        value_loss = batch_loss.item()
        num_batches += 1

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

    reduce_tensor = torch.tensor([total_loss, float(num_batches)], device=rank, dtype=torch.float64)
    dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
    global_total_loss, global_num_batches = reduce_tensor.tolist()
    global_num_batches = max(global_num_batches, 1.0)

    metric_keys = list(metric_logger.keys())
    metric_values = torch.tensor([metric_logger[k] for k in metric_keys], device=rank, dtype=torch.float64)
    dist.all_reduce(metric_values, op=dist.ReduceOp.SUM)

    avg_loss = global_total_loss / global_num_batches
    avg_metrics = {k: (metric_values[idx].item() / global_num_batches) for idx, k in enumerate(metric_keys)}

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


def main_worker(rank, world_size, config, master_port=20008):
    """Distributed training worker function."""
    setup(rank, world_size, master_port=master_port)
    
    torch.random.manual_seed(100019 + rank)
    numpy.random.seed(100019 + rank)

    # 只在主进程初始化TensorBoard
    writer = None
    if rank == 0:
        project_root = os.path.dirname(os.path.abspath(__file__))
        tb_log_dir = os.path.join(project_root, 'runs', 'multi_frame')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard log dir: {tb_log_dir}")

    # 构建数据加载器
    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders_distributed(config, rank, world_size)

    # 初始化模型
    model, optimizer, scheduler, scaler, resume_epoch = initialize_model(config, rank)

    # 初始化损失函数
    criterion = MultiTaskLoss(config).to(rank)
    
    # 使用DDP包装模型
    model = DDP(model,
                device_ids=[rank],
                output_device=rank,
                gradient_as_bucket_view=True,
                bucket_cap_mb=50,
                find_unused_parameters=True)

    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    start_epoch = max(1, resume_epoch + 1)
    if rank == 0 and start_epoch > 1:
        print(f"Resume training from epoch {start_epoch} (checkpoint epoch: {resume_epoch})")

    # 训练循环
    for epoch in range(start_epoch, config.epochs + 1):
        config.current_epoch = epoch
        start_time = time.time()

        # 学习率预热
        if epoch <= config.warmup_epochs:
            lr_scale = min(1.0, epoch / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr * lr_scale

        # 训练阶段
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, config, epoch, train_sampler, rank, writer
        )

        # 验证阶段（所有进程参与，主进程记录）
        did_validate = (epoch % config.val_interval == 0)
        if did_validate:
            val_loss, val_metrics = validate(model, val_loader, criterion, config, epoch, rank, val_sampler)
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

        # 定期保存（只在主进程）
        if rank == 0 and epoch % config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                f"checkpoints/checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
            )

        # 打印日志（只在主进程）
        if rank == 0:
            epoch_time = time.time() - start_time
            mask_mode = "GT" if epoch <= config.use_gt_mask_until_epoch else "Pred"
            print(f"\nEpoch {epoch} Summary:")
            print(f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Mask Mode: {mask_mode} (switch @ epoch {config.use_gt_mask_until_epoch})")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("Train Metrics:", " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()]))
            writer.add_scalar('Loss/train', train_loss, epoch)
            if did_validate:
                writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            # 记录所有训练指标
            for metric_name, metric_value in train_metrics.items():
                writer.add_scalar(f'Train Metrics/{metric_name}', metric_value, epoch)

            if did_validate:
                print("Val Metrics:", " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f'Val Metrics/{metric_name}', metric_value, epoch)
            print("-" * 50)

    # 清理分布式环境
    cleanup()

    # 关闭TensorBoard写入器
    if rank == 0:
        writer.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HiReFF distributed training")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Root directory of training NPZ data (e.g. /path/to/data)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from (default: load VGGT-1B from HuggingFace)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto)")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size per GPU (0=auto)")
    parser.add_argument("--master-port", type=int, default=20008, help="DDP master port")
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--sr-img-size", type=int, default=2072)
    parser.add_argument("--render-mode", type=str, default="gsplat", choices=["gsplat", "mipsplat"])
    parser.add_argument("--dataset-mode", type=str, default="mix", choices=["single", "mix"])
    parser.add_argument("--single-dataset", type=str, default="mvhuman", choices=["dna", "zju", "mvhuman"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=0, help="Epochs between checkpoints (0=auto)")
    parser.add_argument("--val-interval", type=int, default=0, help="Epochs between validation (0=auto)")
    parser.add_argument("--iter-save-interval", type=int, default=2000, help="Iterations between checkpoints (0=off)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    args = parser.parse_args()

    config = TrainingConfig(
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        img_size=args.img_size,
        sr_img_size=args.sr_img_size,
        render_mode=args.render_mode,
        dataset_mode=args.dataset_mode,
        single_dataset=args.single_dataset,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        iter_save_interval=args.iter_save_interval,
        grad_clip=args.grad_clip,
        master_port=args.master_port,
        amp_enabled=not args.no_amp,
    )
    if args.lr is not None:
        config.lr = args.lr
    if args.batch_size > 0:
        config.batch_size = args.batch_size
    if args.save_interval > 0:
        config.save_interval = args.save_interval
    if args.val_interval > 0:
        config.val_interval = args.val_interval

    # Second-phase init for derived fields
    from vggt.training.train_config import TrainingConfig as TC
    # Re-trigger derived field computation
    if config.batch_size == 0:
        config.batch_size = 1 * config.gpus_num
    if config.weight_decay is None:
        config.weight_decay = config.lr / 10
    if config.lr == 1e-5 and config.gpus_num > 0:
        config.lr = 1e-5 * sqrt(config.gpus_num)
    config.save_interval = max(1, config.epochs // 10) if args.save_interval == 0 else config.save_interval
    config.val_interval = max(1, config.epochs // 10) if args.val_interval == 0 else config.val_interval

    print("Initializing...")
    world_size = torch.cuda.device_count()
    print(f"Training on {world_size} GPUs (DDP)")
    print(f"Data root: {config.data_root}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Epochs: {config.epochs}, LR: {config.lr}, Batch size: {config.batch_size}")

    mp.spawn(
        main_worker,
        args=(world_size, config, args.master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
