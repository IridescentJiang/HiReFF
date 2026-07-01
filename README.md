# HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video

<p align="center">
  <a href="https://arxiv.org/abs/2606.29333"><img src="https://img.shields.io/badge/arXiv-2606.29333-b31b1b" alt="arXiv"></a>
  <a href="https://iridescentjiang.github.io/HiReFF/"><img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page"></a>
</p>

Official PyTorch implementation for the paper:

> **[HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video](https://arxiv.org/abs/2606.29333) [ECCV 2026]**

[Yiming Jiang](https://scholar.google.com.hk/citations?user=gqaK3igAAAAJ&hl=zh-CN)<sup>&#x2606;</sup>, [Hanzhang Tu](https://scholar.google.com.hk/citations?user=0S0lNhUAAAAJ&hl=zh-CN&oi=ao), [Wenfeng Song](https://scholar.google.com.hk/citations?user=BDfZbbEAAAAJ&hl=zh-CN), [Siyou Lin](https://scholar.google.com.hk/citations?user=XBzr0pkAAAAJ&hl=zh-CN&oi=ao), [Liang An](https://scholar.google.com.hk/citations?user=s0T1w0gAAAAJ&hl=zh-CN&oi=sra), [Shuai Li](https://scholar.google.com.hk/citations?user=hn0KFx8AAAAJ&hl=zh-CN), [Aimin Hao](https://research.buaa.edu.cn/en/persons/aimin-hao/)<sup>&#x2709;</sup>, [Yebin Liu](https://scholar.google.com.hk/citations?user=ogXIdlYAAAAJ&hl=zh-CN)

> <sup>&#x2606;</sup> Work done during an internship at Tsinghua University. &nbsp; <sup>&#x2709;</sup> Corresponding author. Email: jiangyimingjym@buaa.edu.cn ham@buaa.edu.cn liuyebin@tsinghua.edu.cn

![Teaser](static/images/teaser.png)

> **HiReFF** is a feed-forward method for 2K-resolution 360° human video reconstruction from uncalibrated sparse-view videos. Taking only four views separated by 90° as input, it reconstructs temporally consistent 3D Gaussians in a streaming fashion at 3.01 FPS on a single RTX 4090 GPU, and achieves 2K resolution with only 34% additional VRAM during training compared to 0.5K.

## The Pipeline of Our Method

![Pipeline](static/images/pipeline.png)

> **HiReFF** decomposes 4D human reconstruction into two key tasks: foreground 3D Gaussian reconstruction from uncalibrated sparse-view videos and computationally efficient high-resolution synthesis. It employs Scale-synchronized Camera Calibration to resolve metric scale ambiguity, Gaussian-wise Foreground Masking to reconstruct clean foregrounds, and High-resolution Side-tuning for efficient 2K rendering.

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
python -c "from hireff import HiReFF; print('Install OK')"
```

---

##  Data Preparation

### NPZ Format

Both inference and training use **NPZ files** with the following structure:

```
frame_0000.npz
  ├── view_00  (Python dict with keys: image, intrinsic, extrinsic, mask*)
  ├── view_01
  └── ...
```

Each view dict contains:
- `image` — JPEG-encoded bytes (RGB)
- `intrinsic` — 3×3 float32 camera intrinsic matrix
- `extrinsic` — 4×4 float32 camera extrinsic matrix (camera-to-world)
- `mask` — PNG-encoded foreground mask (**required for training only**)

The directory layout for datasets:

```
{data_root}/{dna-rendering,zju-mocap,mvhuman}/{subject}/frame_XXXX.npz
```

See [docs/data_preparation.md](docs/data_preparation.md) for the full NPZ format specification.

### Sample Data

A preprocessed sample dataset is available on [ModelScope](https://www.modelscope.cn/models/IridescentJiang/HiReFF/tree/master/data_example).

### Dataset Preprocessing

Preprocessing scripts for converting raw DNA-Rendering, ZJU-MoCap, and MVHuman
datasets to NPZ format are provided in `preprocessing/`. See each subdirectory's `README.md`
for instructions.

---

##  Model Inference

### Checkpoints

The model is initialised from the **VGGT-1B** pretrained weights (`facebook/VGGT-1B` on HuggingFace),
then fine-tuned on human datasets.

| Checkpoint | Description | Download |
|---|---|---|
| `checkpoint_dna_mvh_zju.pt` | Fine-tuned on DNA-Rendering + ZJU-MoCap + MVHuman | [ModelScope](https://www.modelscope.cn/models/IridescentJiang/HiReFF/tree/master/checkpoint) |

### Available Scripts

All inference scripts use `argparse` and share utilities in `hireff/utils/inference_utils.py`.

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

#### 2. Video from Sequences (`infer_video.py`)

Processes NPZ sequences or directories of images and outputs MP4 videos with smooth trajectory interpolation.

```bash
python infer_video.py \
    --data-root ./wild_images \
    --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --input-views 0,3,5,8 \
    --inter-view 30 \
    --fps 18 \
    --output-dir output/videos
```

---

##  Model Training

### Running Training

Training uses PyTorch Distributed Data Parallel (DDP) across all available GPUs.

```bash
# Multi-dataset training (starts from VGGT-1B pretrained weights)
python train.py \
    --data-root /path/to/training_data \
    --epochs 10 \
    --dataset-mode mix

# Resume from a HiReFF checkpoint
python train.py \
    --data-root /path/to/training_data \
    --checkpoint ./checkpoints/checkpoint_dna_mvh_zju.pt \
    --epochs 10 \
    --dataset-mode mix

# Single-dataset fine-tuning
python train.py \
    --data-root /path/to/data \
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
| `--checkpoint` | (VGGT-1B from HuggingFace) | Checkpoint to resume from |
| `--dataset-mode` | `mix` | `single` or `mix` |
| `--single-dataset` | `mvhuman` | Dataset for single mode: `dna`, `zju`, or `mvhuman` |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | auto | Learning rate (scaled by GPU count) |
| `--batch-size` | auto (1 per GPU) | Batch size per GPU |
| `--img-size` | 518 | Aggregator input size |
| `--sr-img-size` | 2072 | Super-resolution / render size |
| `--render-mode` | `gsplat` | `gsplat` or `mipsplat` |
| `--warmup-epochs` | 0 | Learning rate warmup epochs |
| `--master-port` | 20008 | DDP master port |
| `--no-amp` | off | Disable automatic mixed precision |

### Monitoring

```bash
tensorboard --logdir runs/
```

---

##  Project Structure

```
hireff/
  models/        — HiReFF model, Aggregator (ViT + alternating attention)
  heads/         — Camera, depth, GS parameter, and mask prediction heads
  layers/        — Transformer blocks, attention, patch embedding, RoPE
  rendering/     — Gaussian splatting rendering (gsplat backend), pose interpolation
  training/      — Loss functions, LPIPS, dataset classes, training config
  utils/         — Pose encoding, geometry, depth unprojection, inference helpers
infer.py         — Primary inference entry point
infer_video.py   — Video rendering from NPZ sequences or image directories
train.py         — DDP training entry point
preprocessing/   — Dataset conversion scripts (DNA / ZJU / MVHuman)
docs/            — Additional documentation
```

##  License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

##  Citation

```bibtex
@misc{jiang2026hireff,
      title={HiReFF: High-Resolution Feedforward Human Reconstruction from Uncalibrated Sparse-View Video}, 
      author={Yiming Jiang and Hanzhang Tu and Wenfeng Song and Siyou Lin and Liang An and Shuai Li and Aimin Hao and Yebin Liu},
      year={2026},
      eprint={2606.29333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.29333}, 
}
```

##  Acknowledgement

We gratefully acknowledge the authors of [VGGT](https://github.com/facebookresearch/vggt) and [AnySplat](https://github.com/AnySplat/AnySplat) for making their code publicly available. Any third-party packages are owned by their respective authors and must be used under their respective licenses.
