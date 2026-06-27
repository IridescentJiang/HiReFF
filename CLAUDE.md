# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiReFF — **4D Gaussian Human Reconstruction via VGGT**. This project extends Meta's VGGT (Visual Geometry Grounded Transformer) to predict 3D Gaussian Splatting parameters from sparse multi-view images of humans, enabling novel-view rendering and 4D reconstruction.

The core idea: given S sparse views of a human subject, the model predicts per-pixel Gaussian parameters (position, opacity, scale, rotation, color), camera poses, depth maps, and foreground masks — all in one forward pass.

## Environment & Installation

- **Python**: >= 3.10
- **Package manager**: `pip install -e .` for editable install. Pixi and uv also supported (see `pyproject.toml`).
- **Core deps** (`requirements.txt`): numpy<2, Pillow, huggingface_hub, einops, safetensors, opencv-python
- **Demo deps** (`requirements_demo.txt`): gradio, viser, tqdm, hydra-core, omegaconf, scipy, onnxruntime, trimesh, matplotlib
- **CUDA**: Requires GPU with CUDA support. Uses `gsplat` for Gaussian rasterization. The `build/` directory contains CMake cache for compiled CUDA rasterizer (`diff-gaussian-rasterization`).
- **PyTorch**: Install separately to match CUDA version (see `docs/package.md`).

## Key Commands

```bash
# Training (DDP, multi-GPU)
python train_npz.py

# Inference — multi-view (reads NPZ files)
python infer_multiview.py

# Inference — single frame
python infer_frame.py

# Inference — video
python infer_video.py

# Inference — 360-degree video
python infer_360_video.py

# Inference — test set
python infer_testset.py

# Inference — speed benchmark
python infer_speed.py

# General-purpose inference entry point
python inference.py
```

Training is configured entirely via the `TrainingConfig` class inside `train_npz.py` — there is no external YAML config file for training. The training script uses `torch.distributed` (NCCL backend, 8 GPUs by default at `localhost:20008`).

## Architecture

### Model (`vggt/models/vggt.py` — `VGGT` class)

The model inherits from `nn.Module` and `PyTorchModelHubMixin`. Input: `[B, S, 3, H, W]` images in [0, 1].

Pipeline:
1. **Aggregator** (`vggt/models/aggregator.py`): A ViT-based encoder (DINOv2-L/14 by default) that processes multi-view images with **alternating attention** — frame attention (per-view, tokens shape `[B*S, P, C]`) and global attention (cross-view, tokens shape `[B, S*P, C]`). Uses rotary position embeddings. Outputs a list of intermediate token tensors (one per attention block pair), each concatenated along the channel dim `[B, S, P, 2C]`.

2. **Heads** (all in `vggt/heads/`):
   - **CameraHead**: Predicts pose encoding `[B, S, 9]` (abs translation + quaternion rotation + FoV). Uses iterative refinement with exponentially weighted loss.
   - **DepthHead** (`DPTHead`): DPT-style dense prediction head. Predicts depth `[B, S, H, W, 1]` and confidence. There are two depth heads: one frozen (pseudo-label) and one trainable (activate).
   - **MaskHead** (`DPTHead`): Predicts foreground mask `[B, S, H, W, 1]` with sigmoid activation.
   - **VGGT_DPT_GS_Head** (`vggt/heads/vggt_dpt_gs_head_main.py`): The key custom head. Extends DPTHead to predict per-pixel Gaussian parameters (opacity, scale, rotation quaternion, color). Merges DPT features with a direct image convolution branch (`input_merger`). Output shape: `[B, S, gs_para_ch, H, W]`.

3. **Post-processing**: Depth is unprojected to world points using predicted intrinsics/extrinsics. Gaussian parameters are processed through `process_gs_map()` (`vggt/heads/gs_adaptor.py`) which applies sigmoid/softplus/normalize activations and masks out background pixels.

### Gaussian Rendering (`vggt/rendering/`)

- `render_image.py`: Core rendering pipeline. Converts pose encodings back to extrinsics/intrinsics, then calls `gsplat.rasterization()` for differentiable rendering. Two render backends: `gsplat` (preferred) and `mipsplat` (legacy).
- `gs_ras.py`: Wrapper around the custom `diff-gaussian-rasterization` CUDA module.
- `camera_mapping.py`: Maps between dataset camera spaces and VGGT's normalized prediction space.

### Training (`train_npz.py`)

- **Data**: Reads pre-processed NPZ files from DNA-Rendering, ZJU-MoCap, and MVHuman datasets. Each NPZ contains multi-view images, depths, masks, intrinsics, extrinsics, and world-space point clouds.
- **Dataset classes** in `vggt/training/data/datasets/`: `DnaRenderingDatasetNpz`, `ZjuMocapDatasetNpz`, `MvHumanDatasetNpz`. NPZ-based datasets are the primary training format.
- **Mixed dataset training**: Supports `mix` mode with weighted/upsample/downsample balancing across the three human datasets.
- **DDP**: Uses `DistributedDataParallel` with `DistributedSampler`. The model runs on 8 GPUs with `batch_size=1` per GPU.
- **Automatic mixed precision**: `torch.amp.autocast` with bfloat16 inside the model forward pass. `GradScaler` for the training loop.

### Loss Functions (`vggt/training/loss.py`)

All loss functions are standalone functions (not `nn.Module` subclasses). Key losses:
- **Camera loss**: L1/L2/Huber on translation, rotation, and FoV, with exponentially decaying weights across refinement iterations.
- **Confidence-weighted regression loss** (`conf_loss`): For depth and point maps. Uses `gamma * reg_loss * confidence - alpha * log(confidence)` formulation to jointly learn predictions and their uncertainty.
- **Render loss** (`RenderLoss` class): L1 + perceptual (LPIPS/VGG) loss between rendered and ground-truth images. Supports foreground-weighted loss, random patch sampling, and foreground alignment.
- **Mask loss**: BCE + Dice loss on predicted vs GT masks.
- **Distillation losses**: geometry (Chamfer distance on point clouds), depth (MSE), transformer feature (MSE on intermediate tokens).
- **Depth consistency loss**: Ensures rendered depth matches predicted depth.

### Inference Scripts

All inference scripts follow a common pattern:
1. Load checkpoint via `VGGT.from_checkpoint(path)` (reads `.pt` files saved during training)
2. Read NPZ or image files
3. Run model forward
4. Optionally render novel views using Gaussian Splatting
5. Save output images/videos

Scripts differ by input type: single/multi-view images, video sequences, 360-degree rendering, test set evaluation, or speed benchmarking.

## Key Design Decisions

- **Pose encoding format** (`absT_quaR_FoV`): 9-dim vector `[Tx, Ty, Tz, Qw, Qx, Qy, Qz, FoVx, FoVy]`. Conversion functions in `vggt/utils/pose_enc.py`.
- **Gaussian parameter channels**: opacity(1) + scale(3) + rotation(4) + color(3 × (sh_degree+1)²). With `sh_degree=0`, total = 1+3+4+3 = 11 channels.
- **Background color**: White (1.0) is the default background for both training and rendering, since datasets use white backgrounds.
- **Training image sizes**: Aggregator input = 518×518, super-resolution input = 2072×2072.
- **Checkpoint format**: Standard PyTorch checkpoint with `model_state` or `state_dict` key, with `module.` prefix stripped for DDP compatibility.
- **NPZ data format**: Each view stored as `view_XX` with keys: `image` (JPEG bytes), `mask` (PNG bytes), `intrinsic` (3×3), `extrinsic` (4×4 world-to-camera).
