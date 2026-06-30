# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HiReFF — **4D Gaussian Human Reconstruction via VGGT**. This project extends the Visual Geometry Grounded Transformer (VGGT) architecture to predict 3D Gaussian Splatting parameters from sparse multi-view images of humans, enabling novel-view rendering and 4D reconstruction.

Given S sparse views of a human subject, the model predicts per-pixel Gaussian parameters (position, opacity, scale, rotation, color), camera poses, depth maps, and foreground masks — all in one forward pass.

## Environment & Installation

- **Python**: >= 3.10
- **CUDA**: Required for `gsplat` Gaussian rasterizer.
- **PyTorch**: Install separately to match CUDA version. Listed as a dependency but you may want to pre-install with the right CUDA index.

```bash
pip install -e .[gsplat,train]       # inference + training
pip install -e .[gsplat,train,demo]  # + Gradio / Viser GUI
```

Core dependencies are in `pyproject.toml`. `requirements.txt` is a pointer to it.

## Key Commands

```bash
# Training (DDP, all available GPUs)
python train.py --data-root /path/to/data --epochs 10

# Inference — multi-view rendering (primary entry point)
python infer.py --data-root <dir> --checkpoint-path <path> --input-views 25,1,13,37 --novel-views 1,4,7,10

# Inference — video (NPZ or image directory)
python infer_video.py --data-root <dir> --checkpoint-path <path>
```

All inference scripts use `argparse`. Training is configured via `TrainingConfig` in `hireff/training/train_config.py` with command-line overrides.

## Architecture

### Model (`hireff/models/hireff_model.py` — `HiReFF` class)

Inherits from `nn.Module` and `PyTorchModelHubMixin`. Input: `[B, S, 3, H, W]` images in [0, 1].

Pipeline:
1. **Aggregator** (`hireff/models/aggregator.py`): ViT-based encoder (DINOv2-L/14 default). Processes multi-view images with **alternating attention** — frame attention (per-view, tokens `[B*S, P, C]`) and global attention (cross-view, tokens `[B, S*P, C]`). Uses rotary position embeddings. Outputs a list of intermediate token tensors (one per attention block pair), each concatenated along the channel dim `[B, S, P, 2C]`.

2. **Heads** (all in `hireff/heads/`):
   - **CameraHead**: Predicts pose encoding `[B, S, 9]` (abs translation + quaternion rotation + FoV). Iterative refinement with exponentially weighted loss.
   - **DepthHead** (`DPTHead`): DPT-style dense prediction. Predicts depth `[B, S, H, W, 1]` and confidence. Two instances: one frozen (pseudo-label), one trainable.
   - **MaskHead** (`DPTHead`): Predicts foreground mask `[B, S, H, W, 1]` (sigmoid).
   - **HiReFF_DPT_GS_Head** (`hireff/heads/hireff_dpt_gs_head.py`): The key custom head. Extends DPTHead with an EdgeNeXt encoder for multi-resolution feature extraction and separate sub-heads (rot, scale, color, opacity) for per-pixel Gaussian parameters. Output: `[B, S, gs_para_ch, H, W]`.

3. **Post-processing**: Depth is unprojected to world points via predicted intrinsics/extrinsics. Gaussian parameters go through `process_gs_map()` (`hireff/heads/gs_adaptor.py`) — sigmoid/softplus/normalize activations, background masking.

### Gaussian Rendering (`hireff/rendering/`)

- `render_image.py`: Core pipeline. Converts pose encodings → extrinsics/intrinsics, then calls `gsplat.rasterization()` for differentiable rendering. Active render functions: `batch_render_images_my()` → `vectorized_gaussian_render_gsplat_my()`.
- `camera_mapping.py`: Maps between dataset camera spaces and HiReFF's normalized prediction space.

### Training (`train.py`)

- **Pretrained model**: Training starts from VGGT-1B (`facebook/VGGT-1B` on HuggingFace) by default. Set `load_VGGT=True` in config or pass `--checkpoint` to resume from a HiReFF checkpoint.
- **Config**: `TrainingConfig` dataclass in `hireff/training/train_config.py`. All fields overridable via CLI args.
- **Data**: NPZ files from DNA-Rendering (48 views), ZJU-MoCap (24 views), MVHuman (16 views).
- **Dataset classes**: `DnaRenderingDatasetNpz`, `ZjuMocapDatasetNpz`, `MvHumanDatasetNpz` in `hireff/training/data/datasets/`.
- **Mix mode**: Supports `--dataset-mode mix` with weighted/upsample/downsample balancing across datasets.
- **DDP**: `DistributedDataParallel` with `DistributedSampler`. `batch_size=1` per GPU default.

### Loss Functions (`hireff/training/loss.py` + `render_loss.py`)

- **Camera loss**: L1/L2/Huber on translation, rotation, FoV, with exponential decay across refinement iterations.
- **Confidence-weighted regression** (`conf_loss`): `gamma * reg_loss * confidence - alpha * log(confidence)` formulation.
- **Render loss** (`RenderLoss` class): L1 + perceptual (LPIPS/VGG). Foreground-weighted, random patch sampling.
- **Mask loss**: BCE + Dice.
- **Distillation**: depth (MSE), geometry (Chamfer), transformer feature (MSE).
- **Depth consistency**: Rendered depth vs predicted depth.

### Inference Scripts

All three scripts share utilities in `hireff/utils/inference_utils.py` (`load_model`, `parse_view_ids`, `collect_npz_files`, `read_dna_npz_entry`, `save_render_images`, etc.).

- `infer.py`: Primary entry point. Given sparse input views, predicts Gaussians and renders novel views from NPZ files.
- `infer_video.py`: Video from NPZ sequences or image directories with smooth trajectory interpolation.

## Key Design Decisions

- **Pose encoding** (`absT_quaR_FoV`): 9-dim `[Tx, Ty, Tz, Qw, Qx, Qy, Qz, FoVx, FoVy]`. Conversion in `hireff/utils/pose_enc.py`.
- **Gaussian parameter channels**: opacity(1) + scale(3) + rotation(4) + color(3 × (sh_degree+1)²). With `sh_degree=0` → 11 channels.
- **Background**: White (1.0) default for training and rendering.
- **Image sizes**: Aggregator = 518×518, super-resolution/rendering = 2072×2072.
- **Checkpoint format**: PyTorch `.pt` with `model_state` or `state_dict` key, `module.` prefix stripped for DDP. The VGGT-1B pretrained weights are loaded from HuggingFace (`facebook/VGGT-1B`) when `load_VGGT=True`.
- **NPZ data format**: Each view stored as `view_XX` with keys: `image` (JPEG bytes), `mask` (PNG bytes), `intrinsic` (3×3), `extrinsic` (4×4 camera-to-world). See `docs/data_preparation.md`.
- **Preprocessing**: Conversion scripts for DNA-Rendering, ZJU-MoCap, and MVHuman datasets are in `preprocessing/`. Each subdirectory has its own README.
