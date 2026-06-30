"""Training configuration for HiReFF.

All defaults can be overridden by command-line arguments in ``train.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import torch


@dataclass
class TrainingConfig:
    """Configuration for the HiReFF training pipeline."""

    # ---- Data ----
    data_root: str = ""  # REQUIRED: path to training NPZ data
    dataset_mode: str = "mix"  # "single" or "mix"
    single_dataset: str = "mvhuman"  # dna, zju, or mvhuman (only for dataset_mode="single")
    mix_datasets: list[str] = field(default_factory=lambda: ["dna", "zju", "mvhuman"])
    mix_balance: bool = True
    mix_balance_mode: str = "weighted"  # downsample, upsample, or weighted
    mix_balance_seed: int = 42
    mix_balance_val: bool = True
    mix_dataset_weights: dict[str, float] = field(default_factory=lambda: {"dna": 0.45, "zju": 0.1, "mvhuman": 0.45})
    mix_weighted_target_total: int | None = None
    train_type: str = "frames"  # "single_frame" or "frames"
    frame_numbers: list[str] | None = None
    data_sample_rate: int | None = None  # 1/N subsampling

    # ---- Derived data roots ----
    dna_data_root: str = ""
    zju_data_root: str = ""
    mvhuman_data_root: str = ""

    def __post_init__(self):
        if self.data_root:
            self.dna_data_root = f"{self.data_root}/dna-rendering"
            self.zju_data_root = f"{self.data_root}/zju-mocap"
            self.mvhuman_data_root = f"{self.data_root}/mvhuman"
        if self.train_type == "single_frame":
            self.frame_numbers = ["000000"]
        elif self.train_type == "frames":
            self.frame_numbers = None
        if self.frame_numbers and len(self.frame_numbers) <= 10:
            self.frame_numbers = self.frame_numbers * int(100 // len(self.frame_numbers))
        if self.dataset_mode == "single":
            if self.single_dataset == "dna":
                self.data_root = self.dna_data_root
            elif self.single_dataset == "zju":
                self.data_root = self.zju_data_root
            elif self.single_dataset == "mvhuman":
                self.data_root = self.mvhuman_data_root

    # ---- Image sizes ----
    img_size: int = 518  # Aggregator input size
    sr_img_size: int = 2072  # Super-res / render image size
    lpip_patch_size: int = 518
    random_patch: bool = False
    random_patch_epoch: int = 1000
    multiview_supervise: int = 2

    # ---- Model ----
    load_VGGT: bool = True
    model_name: str = "facebook/VGGT-1B"
    checkpoint: str | None = None
    only_load_mask_head: bool = False
    render_mode: str = "gsplat"  # gsplat or mipsplat

    # ---- Training ----
    lr: float = 1e-5
    epochs: int = 10
    weight_decay: float | None = None
    warmup_epochs: int = 0
    save_interval: int = 1
    val_interval: int = 1
    iter_save_interval: int = 2000

    # ---- GPU / distributed ----
    gpus_num: int = field(default_factory=torch.cuda.device_count)
    batch_size: int = 0  # 0 = auto (1 * gpus_num)
    num_workers: int = 4
    master_port: int = 20008
    amp_enabled: bool = True
    grad_clip: float = 1.0
    reset_optimizer_lrs_on_resume: bool = True

    # ---- Head / loss activation ----
    camera_head_activate: bool = True
    depth_head_activate: bool = True
    gs_para_head_activate: bool = True
    aggregator_activate: bool = True
    mask_head_activate: bool = True

    camera_loss_activate: bool = True
    distill_depth_loss_activate: bool = True
    render_loss_activate: bool = True
    mask_loss_activate: bool = True
    color_dist_loss_activate: bool = True
    foreground_region_loss_activate: bool = False
    depth_consist_loss_activate: bool = False
    distill_geo_loss_activate: bool = False

    use_gt_mask_until_epoch: int = 20
    camera_loss_max_for_backward: float | None = 1e-2

    # ---- Derived ----
    render_images_path: str = "render_images/"
    device: str | None = None

    def __post_init_2__(self):
        """Second-phase init called after user overrides (handled in train_npz)."""
        if self.batch_size == 0:
            self.batch_size = 1 * self.gpus_num
        if self.weight_decay is None:
            self.weight_decay = self.lr / 10
        if self.lr == 1e-5 and self.gpus_num > 0:
            self.lr = 1e-5 * sqrt(self.gpus_num)
        self.save_interval = max(1, self.epochs // 10)
        self.val_interval = max(1, self.epochs // 10)
