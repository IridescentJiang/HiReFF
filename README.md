# HiReFF: 4D Gaussian Human Reconstruction via VGGT

**HiReFF** reconstructs animatable 3D Gaussian human avatars from sparse
multi-view images in a single forward pass. It extends [VGGT](https://github.com/facebookresearch/vggt)
with a Gaussian Splatting head that predicts per-pixel 3D Gaussian parameters
(position, opacity, scale, rotation, colour), enabling high-quality novel-view
rendering without per-subject optimisation.

<p align="center">
  <strong><a href="#-environment-setup">Environment</a></strong> ·
  <strong><a href="#-model-inference">Inference</a></strong> ·
  <strong><a href="#-model-training">Training</a></strong> ·
  <strong><a href="#-citation">Citation</a></strong>
</p>

---

##  Environment Setup

### Prerequisites

- **Python** >= 3.10
- **CUDA** >= 11.8 (required for `gsplat` Gaussian rasterizer)
- **GPU** with at least 16 GB VRAM for inference; 8 GPUs recommended for training

### Installation

```bash
git clone https://github.com/IridescentJiang/HiReFF.git
cd HiReFF

# 1. Install PyTorch first (match your CUDA version)
#    This project was developed with torch 2.5 + CUDA 11.8:
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118

# 2. Core install (inference + training)
pip install -e .[gsplat,train]

# Verify
python -c "from vggt import VGGT; print('Install OK')"
```

---

##  Model Inference

### Download Checkpoints

Download pretrained checkpoints and place them under `checkpoints/`:

| Checkpoint | Description |
|---|---|
| `checkpoint_dna_mvh_zju.pt` | Main model trained on DNA-Rendering + ZJU-MoCap + MVHuman |
| `checkpoint_finetune.pt` | Fine-tuned variant |
| `8_view_input.pt` | 8-view input variant |

### Available Scripts

All inference scripts use `argparse` and share utilities in `vggt/utils/inference_utils.py`.

#### 1. Multi-view Rendering (`infer.py`)

The primary entry point. Given sparse input views, predicts Gaussians and renders novel views.

```bash
python infer.py \
    --data-root ./test_data \
    --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --input-views 25,1,13,37 \
    --novel-views 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46 \
    --output-dir output/multiview
```

#### 2. 360° Video Rendering (`infer_360_video.py`)

Generates smooth camera trajectories with Slerp or orbital interpolation between anchor views.

```bash
python infer_360_video.py \
    --data-root ./test_data \
    --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --input-views 25,1,13,37 \
    --inter-views-between 4 \
    --interpolation-mode orbit \
    --output-dir output/multiview
```

#### 3. Video from Sequences (`infer_video.py`)

Processes NPZ sequences or directories of images and outputs MP4 videos with smooth trajectory interpolation.

```bash
python infer_video.py \
    --data-root ./wild_images \
    --checkpoint-path ./checkpoints/8_view_input.pt \
    --input-views 0,3,5,8 \
    --inter-view 30 \
    --fps 18 \
    --output-dir output/videos
```

### Input Data Format

Inference uses **NPZ files** with the following structure:

```
frame_0000.npz
  ├── view_00  (Python dict with keys: image, intrinsic, extrinsic)
  ├── view_01
  └── ...
```

Each view dict contains:
- `image` — JPEG-encoded bytes (RGB)
- `intrinsic` — 3×3 float32 camera intrinsic matrix
- `extrinsic` — 4×4 float32 camera extrinsic matrix (camera-to-world)

See [docs/data_preparation.md](docs/data_preparation.md) for details.

---

##  Model Training

### Data Preparation

Training requires NPZ files with the same structure as inference, plus:
- `mask` — PNG-encoded foreground mask
- Directory layout: `{data_root}/{dna-rendering,zju-mocap,mvhuman}/{subject}/frame_XXXX.npz`

See [docs/data_preparation.md](docs/data_preparation.md) for the full specification and example conversion scripts.

### Running Training

Training uses PyTorch Distributed Data Parallel (DDP) across all available GPUs:

```bash
# Mixed-dataset training (DNA + ZJU + MVHuman)
python train.py \
    --data-root /path/to/training_data \
    --checkpoint ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --epochs 10 \
    --dataset-mode mix

# Single-dataset fine-tuning
python train.py \
    --data-root /path/to/data \
    --checkpoint ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --epochs 5 \
    --dataset-mode single \
    --single-dataset mvhuman
```

To control which GPUs to use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --data-root /path/to/data
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-root` | (required) | Root directory of training NPZ data |
| `--checkpoint` | `checkpoints/checkpoint_dna_mvh_zju.pt` | Pretrained checkpoint path |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | auto | Learning rate (auto-scaled by GPU count) |
| `--batch-size` | 1 per GPU | Batch size per GPU |
| `--dataset-mode` | mix | `single` or `mix` |
| `--render-mode` | gsplat | `gsplat` or `mipsplat` |
| `--master-port` | 20008 | DDP master port |

### Monitoring

```bash
tensorboard --logdir runs/
```

---

##  Project Structure

```
vggt/
  models/        — VGGT model, Aggregator (ViT + alternating attention)
  heads/         — Camera, depth, GS parameter, and mask prediction heads
  layers/        — Transformer blocks, attention, patch embedding, RoPE
  rendering/     — Gaussian splatting rendering (gsplat backend), pose interpolation
  training/      — Loss functions, LPIPS, dataset classes, training config
  utils/         — Pose encoding, geometry, depth unprojection, inference helpers
infer.py         — Primary inference entry point
infer_360_video.py — 360° multi-view rendering with camera interpolation
infer_video.py   — Video rendering from NPZ sequences or image directories
train.py         — DDP training entry point
docs/            — Additional documentation
```

##  License

This project is licensed under CC BY-NC 4.0 — see [LICENSE](LICENSE) for details.

##  Citation

```bibtex
@article{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and others},
  journal={arXiv preprint},
  year={2025}
}
```
