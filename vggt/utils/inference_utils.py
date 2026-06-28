"""Shared utilities for VGGT inference scripts.

This module consolidates functions that were duplicated across multiple inference
scripts: model loading, NPZ data reading, image saving, pose alignment helpers, etc.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import numpy as np
import torch
import torchvision
from torchvision.utils import save_image

from vggt.models.vggt import VGGT


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str | None = None) -> tuple[VGGT, str]:
    """Load a VGGT model from a local checkpoint file.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint saved by ``train.py``.
        device: Target device (``"cuda"``, ``"cpu"``, etc.). Auto-detected when None.

    Returns:
        (model, device_str) tuple. Model is in eval mode and on the target device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VGGT.from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device).float()
    return model, device


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def parse_view_ids(view_ids_str: str) -> list[int]:
    """Parse a comma-separated string of view ids into a list of ints.

    Example: ``"25,1,13,37"`` -> ``[25, 1, 13, 37]``.
    """
    view_ids: list[int] = []
    for item in view_ids_str.split(","):
        item = item.strip()
        if not item:
            continue
        view_ids.append(int(item))
    if len(view_ids) == 0:
        raise ValueError("view ids string must not be empty")
    return view_ids


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_npz_files(data_root: str) -> list[str]:
    """Find all ``.npz`` files under *data_root* (direct + recursive)."""
    direct = glob.glob(os.path.join(data_root, "*.npz"))
    recursive = glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True)
    return sorted(set(direct + recursive))


def sort_key_maybe_int(name: str):
    """Sort key: treat digit-only names as integers, others as strings."""
    return int(name) if name.isdigit() else name


# ---------------------------------------------------------------------------
# NPZ reading
# ---------------------------------------------------------------------------

def ensure_homogeneous_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    """Convert a 3×4 extrinsic to 4×4 by appending [0,0,0,1]."""
    if extrinsic.shape == (4, 4):
        return extrinsic
    if extrinsic.shape == (3, 4):
        last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=extrinsic.dtype)
        return torch.cat([extrinsic, last_row.unsqueeze(0)], dim=0)
    raise ValueError(f"Unsupported extrinsic shape: {tuple(extrinsic.shape)}")


def read_dna_npz_entry(
    npz_path: str,
    view_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read images, intrinsics, and extrinsics from a DNA-format NPZ file.

    Each view is stored as ``view_XX`` with keys: ``image`` (JPEG bytes),
    ``intrinsic`` (3×3), ``extrinsic`` (4×4 camera-to-world, inverted to
    world-to-camera).

    Returns:
        (images [V,3,H,W] in [0,1], intrinsics [V,3,3], extrinsics [V,4,4]).
    """
    images: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    extrinsics: list[torch.Tensor] = []

    with np.load(npz_path, allow_pickle=True) as archive:
        for view in view_ids:
            data_key = f"view_{view:02d}"
            if data_key not in archive:
                raise KeyError(f"{data_key} not found in {npz_path}.")

            entry = archive[data_key].item()
            rgb_image = torchvision.io.decode_image(
                torch.from_numpy(entry["image"]),
                mode=torchvision.io.ImageReadMode.RGB,
            ).float() / 255.0

            images.append(rgb_image)
            intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))
            # Invert camera-to-world to world-to-camera (VGGT convention).
            extrinsics.append(
                torch.linalg.inv(torch.from_numpy(entry["extrinsic"])).to(dtype=torch.float32)
            )

    return (
        torch.stack(images, dim=0),
        torch.stack(intrinsics, dim=0),
        torch.stack(extrinsics, dim=0),
    )


def decode_mask_from_entry(
    entry: dict,
    image_hw: tuple[int, int],
) -> torch.Tensor | None:
    """Try to decode a mask from an NPZ entry dict.

    Supports raw arrays (2D/3D), JPEG/PNG bytes, and common key names.
    Returns a float tensor ``[1, H, W]`` in {0,1} or None.
    """
    mask_keys = ["mask", "masks", "fg_mask", "foreground_mask", "alpha", "matte", "seg"]
    for key in mask_keys:
        if key not in entry:
            continue
        raw_mask = entry[key]
        if raw_mask is None:
            continue

        mask_tensor: torch.Tensor | None = None
        if isinstance(raw_mask, np.ndarray):
            if raw_mask.ndim == 1 and raw_mask.dtype == np.uint8:
                mask_tensor = torchvision.io.decode_image(
                    torch.from_numpy(raw_mask),
                    mode=torchvision.io.ImageReadMode.GRAY,
                )
            elif raw_mask.ndim == 2:
                mask_tensor = torch.from_numpy(raw_mask).unsqueeze(0)
            elif raw_mask.ndim == 3:
                tensor_3d = torch.from_numpy(raw_mask)
                if tensor_3d.shape[0] in (1, 3, 4):
                    mask_tensor = tensor_3d[:1]
                elif tensor_3d.shape[-1] in (1, 3, 4):
                    mask_tensor = tensor_3d[..., :1].permute(2, 0, 1)
                else:
                    mask_tensor = tensor_3d[:1]
        elif isinstance(raw_mask, (bytes, bytearray)):
            byte_array = np.frombuffer(raw_mask, dtype=np.uint8)
            mask_tensor = torchvision.io.decode_image(
                torch.from_numpy(byte_array),
                mode=torchvision.io.ImageReadMode.GRAY,
            )

        if mask_tensor is None:
            continue

        # Normalise to [0, 1].
        if mask_tensor.dtype == torch.uint8:
            mask_tensor = mask_tensor.float() / 255.0
        else:
            mask_tensor = mask_tensor.float()
            max_value = float(mask_tensor.max().item()) if mask_tensor.numel() > 0 else 1.0
            if max_value > 1.0:
                mask_tensor = mask_tensor / max_value

        if mask_tensor.shape[-2:] != image_hw:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=image_hw,
                mode="nearest",
            ).squeeze(0)

        return (mask_tensor > 0.5).float()

    return None


def read_dna_npz_views(
    npz_path: str,
    view_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[bool]]:
    """Read images, masks, intrinsics, and extrinsics from a DNA-format NPZ.

    Like :func:`read_dna_npz_entry` but also decodes masks and returns a
    ``mask_from_npz`` flag list.
    """
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    intrinsics: list[torch.Tensor] = []
    extrinsics: list[torch.Tensor] = []
    mask_from_npz: list[bool] = []

    with np.load(npz_path, allow_pickle=True) as archive:
        for view in view_ids:
            data_key = f"view_{view:02d}"
            if data_key not in archive:
                raise KeyError(f"{data_key} not found in {npz_path}")

            entry = archive[data_key].item()
            rgb_image = torchvision.io.decode_image(
                torch.from_numpy(entry["image"]),
                mode=torchvision.io.ImageReadMode.RGB,
            ).float() / 255.0
            images.append(rgb_image)
            intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))

            raw_extrinsic = torch.from_numpy(entry["extrinsic"]).to(dtype=torch.float32)
            raw_extrinsic = ensure_homogeneous_extrinsic(raw_extrinsic)
            extrinsics.append(torch.linalg.inv(raw_extrinsic))

            decoded_mask = decode_mask_from_entry(entry, image_hw=rgb_image.shape[-2:])
            if decoded_mask is None:
                decoded_mask = torch.ones(1, *rgb_image.shape[-2:], dtype=torch.float32)
                mask_from_npz.append(False)
            else:
                mask_from_npz.append(True)
            masks.append(decoded_mask)

    return (
        torch.stack(images, dim=0),
        torch.stack(masks, dim=0),
        torch.stack(intrinsics, dim=0),
        torch.stack(extrinsics, dim=0),
        mask_from_npz,
    )


# ---------------------------------------------------------------------------
# Image directory reading (for infer_video / wild images)
# ---------------------------------------------------------------------------

def read_frame_dir_entry(
    frame_dir: str,
    view_ids: list[int] | None = None,
) -> torch.Tensor:
    """Read images from a directory of individual frame files.

    If *view_ids* is provided and files named by view id exist, those are
    selected; otherwise all images are sorted and returned.
    """
    image_candidates: list[str] = []
    for ext in ("jpg", "jpeg", "png", "bmp", "webp"):
        image_candidates.extend(glob.glob(os.path.join(frame_dir, f"*.{ext}")))
        image_candidates.extend(glob.glob(os.path.join(frame_dir, f"*.{ext.upper()}")))

    if not image_candidates:
        raise FileNotFoundError(f"No images found in frame directory: {frame_dir}")

    image_by_stem: dict[str, str] = {}
    for path in image_candidates:
        stem = os.path.splitext(os.path.basename(path))[0]
        image_by_stem[stem] = path

    selected_paths: list[str] = []
    if view_ids is not None and len(view_ids) > 0:
        all_found = all(str(view_id) in image_by_stem for view_id in view_ids)
        if all_found:
            selected_paths = [image_by_stem[str(view_id)] for view_id in view_ids]

    if not selected_paths:
        def _sort_key(path: str):
            stem = os.path.splitext(os.path.basename(path))[0]
            return (0, int(stem)) if stem.isdigit() else (1, stem)
        selected_paths = sorted(image_candidates, key=_sort_key)

    images: list[torch.Tensor] = []
    for image_path in selected_paths:
        rgb_image = torchvision.io.read_image(
            image_path, mode=torchvision.io.ImageReadMode.RGB,
        ).float() / 255.0
        images.append(rgb_image)

    if len(images) < 1:
        raise RuntimeError(f"Failed to read valid images from frame directory: {frame_dir}")

    return torch.stack(images, dim=0)


# ---------------------------------------------------------------------------
# Camera parameter loading
# ---------------------------------------------------------------------------

def load_cam_para(
    cam_para_path: str,
    cam_view_ids: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Load camera parameters from a JSON file.

    Expected format: ``{"intrinsic": [[...], ...], "extrinsic": [[...], ...]}``.
    """
    with open(cam_para_path, "r", encoding="utf-8") as file:
        cam_para = json.load(file)

    intrinsic_all = torch.tensor(cam_para["intrinsic"], dtype=torch.float32)
    extrinsic_all = torch.tensor(cam_para["extrinsic"], dtype=torch.float32)
    total_views = intrinsic_all.shape[0]

    if cam_view_ids is None or len(cam_view_ids) == 0:
        cam_view_ids = list(range(total_views))

    if any(view_id < 0 or view_id >= total_views for view_id in cam_view_ids):
        raise ValueError(f"cam_view_ids out of range [0, {total_views - 1}]: {cam_view_ids}")

    return intrinsic_all[cam_view_ids], extrinsic_all[cam_view_ids], cam_view_ids


# ---------------------------------------------------------------------------
# Type conversion helpers
# ---------------------------------------------------------------------------

def _to_float32_if_needed(value: Any) -> Any:
    """Recursively convert float tensors to float32."""
    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
        return value.float()
    if isinstance(value, dict):
        return {k: _to_float32_if_needed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_float32_if_needed(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_float32_if_needed(v) for v in value)
    return value


def ensure_render_float32(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Ensure pose_enc and flat_gs tensors are float32 for rendering."""
    predictions = {k: v for k, v in predictions.items()}
    if "pose_enc" in predictions:
        predictions["pose_enc"] = _to_float32_if_needed(predictions["pose_enc"])
    if "flat_gs" in predictions:
        predictions["flat_gs"] = _to_float32_if_needed(predictions["flat_gs"])
    return predictions


def enforce_colorful_flat_gs(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace predicted colors with GT colors from NPZ when available.

    This improves rendering quality when GT color information is embedded in
    the flat_gs. Only active when ``flat_gs`` is a list of dicts containing
    a ``gt_colors`` key.
    """
    if "flat_gs" not in predictions or not isinstance(predictions["flat_gs"], list):
        return predictions

    pcs = []
    for pc in predictions["flat_gs"]:
        if not isinstance(pc, dict):
            pcs.append(pc)
            continue

        updated = {k: v for k, v in pc.items()}
        gt_colors = updated.get("gt_colors", None)
        if isinstance(gt_colors, torch.Tensor) and gt_colors.ndim == 2 and gt_colors.shape[-1] == 3:
            updated["colors"] = gt_colors.unsqueeze(-1).contiguous()

        pcs.append(updated)

    predictions = {k: v for k, v in predictions.items()}
    predictions["flat_gs"] = pcs
    return predictions


# ---------------------------------------------------------------------------
# Pose alignment
# ---------------------------------------------------------------------------

def compute_translation_alignment_scale(
    source_extrinsics: torch.Tensor,
    target_extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute per-batch scale factor aligning source translation norms to target.

    Args:
        source_extrinsics: ``[B, V_s, 4, 4]`` source world-to-camera matrices.
        target_extrinsics: ``[B, V_t, 4, 4]`` target world-to-camera matrices.

    Returns:
        Scale tensor ``[B]``.
    """
    src_t = source_extrinsics[..., :3, 3]
    tgt_t = target_extrinsics[..., :3, 3]

    batch_size = min(src_t.shape[0], tgt_t.shape[0])
    scales: list[torch.Tensor] = []
    for b_idx in range(batch_size):
        view_count = min(src_t.shape[1], tgt_t.shape[1])
        pair_scales: list[torch.Tensor] = []
        for view_idx in range(1, view_count):
            src_norm = torch.norm(src_t[b_idx, view_idx])
            tgt_norm = torch.norm(tgt_t[b_idx, view_idx])
            if src_norm > 1e-6 and tgt_norm > 1e-6:
                pair_scales.append(tgt_norm / src_norm)

        if pair_scales:
            scales.append(torch.stack(pair_scales).mean())
        else:
            scales.append(torch.tensor(1.0, device=source_extrinsics.device, dtype=torch.float32))

    return torch.stack(scales).to(device=source_extrinsics.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_render_images(
    images: torch.Tensor,
    output_root: str,
    seq_id: str,
    frame_name: str,
    view_list: list[int],
) -> None:
    """Save rendered images under ``{output_root}/{seq_id}/{frame_name}_pred_ours/``."""
    save_dir = os.path.join(output_root, seq_id, f"{frame_name}_pred_ours")
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(images.size(0)):
        save_path = os.path.join(save_dir, f"{view_list[idx]:02d}.png")
        save_image(images[idx], save_path, normalize=False)


def save_frame_image(
    image: torch.Tensor,
    save_dir: str,
    frame_name: str,
) -> None:
    """Save a single frame image as ``{save_dir}/{frame_name}.png``."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{frame_name}.png")
    save_image(image, save_path, normalize=False)


def save_video_frames(
    images: torch.Tensor,
    video_path: str,
) -> str:
    """Save video frames to a directory named after the video file."""
    frame_dir = os.path.splitext(video_path)[0]
    os.makedirs(frame_dir, exist_ok=True)

    frames = images.detach().cpu()
    if frames.dim() == 5 and frames.shape[1] == 1:
        frames = frames[:, 0]

    if frames.dim() != 4:
        raise ValueError(f"Expected frames tensor shape [N, C, H, W], got {tuple(frames.shape)}")

    for i in range(frames.shape[0]):
        frame_path = os.path.join(frame_dir, f"{i:06d}.png")
        save_image(frames[i].clamp(0, 1), frame_path, normalize=False)

    return frame_dir
