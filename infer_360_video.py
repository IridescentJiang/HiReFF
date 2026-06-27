"""360-degree multi-view rendering with interpolated camera trajectories.

Given 4 sparse input views, predicts 3D Gaussians and renders a smooth
orbit around the subject by interpolating between views using Slerp or
orbital interpolation.

Example::

    python infer_360_video.py \\
        --data-root ./test_data \\
        --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \\
        --input-views 25,1,13,37 \\
        --inter-views-between 4 \\
        --interpolation-mode orbit \\
        --output-dir output/multiview
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torchvision.utils import save_image

from vggt.utils.inference_utils import (
    load_model,
    parse_view_ids,
    read_dna_npz_entry,
    sort_key_maybe_int,
)
from vggt.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.rendering.render_image import batch_render_images_my


# ---------------------------------------------------------------------------
# Sample preparation
# ---------------------------------------------------------------------------

def prepare_sample(
    npz_path: str,
    input_view_ids: list[int],
    agg_input_size: int,
    input_size: int,
) -> dict[str, torch.Tensor]:
    """Read NPZ and resize images to aggregator and head resolutions."""
    images, intrinsics, extrinsics = read_dna_npz_entry(npz_path, input_view_ids)

    origin_h, origin_w = images.shape[-2:]
    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False,
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False,
    )

    intrinsics_input = adjust_intrinsic_batch(
        intrinsics, orig_w=origin_w, orig_h=origin_h,
        target_w=input_size, target_h=input_size,
    )

    return {
        "images_lr": images_lr,
        "images_hr": images_hr,
        "intrinsics_input": intrinsics_input,
        "extrinsics": extrinsics,
    }


# ---------------------------------------------------------------------------
# Pose interpolation (specialised for 360° rendering)
# ---------------------------------------------------------------------------

def interpolate_pose_open(
    pose_tensor: torch.Tensor,
    inter_views: int,
    close_loop: bool = False,
) -> torch.Tensor:
    """Interpolate poses with Slerp for rotation and linear for translation/fov."""
    if inter_views <= 0:
        return pose_tensor

    batch_size = pose_tensor.shape[0]
    output_batches = []

    for batch_idx in range(batch_size):
        batch_pose = pose_tensor[batch_idx]
        pose_np = batch_pose.detach().cpu().numpy()
        num_views = pose_np.shape[0]

        if num_views <= 1:
            output_batches.append(batch_pose)
            continue

        segments = num_views if close_loop else (num_views - 1)
        interpolated_list = []

        for view_idx in range(segments):
            next_idx = (view_idx + 1) % num_views
            pose_start = pose_np[view_idx]
            pose_end = pose_np[next_idx]

            pos_start, quat_start, fov_start = pose_start[:3], pose_start[3:7], pose_start[7:9]
            pos_end, quat_end, fov_end = pose_end[:3], pose_end[3:7], pose_end[7:9]

            if view_idx == 0:
                interpolated_list.append(pose_start)

            time_steps = np.linspace(0.0, 1.0, inter_views + 2)[1:-1]

            rotations = R.from_quat([quat_start, quat_end])
            slerp = Slerp([0.0, 1.0], rotations)
            interp_quats = slerp(time_steps).as_quat()

            for interp_idx, time_value in enumerate(time_steps):
                interp_pos = (1.0 - time_value) * pos_start + time_value * pos_end
                interp_fov = (1.0 - time_value) * fov_start + time_value * fov_end
                interp_pose = np.concatenate([interp_pos, interp_quats[interp_idx], interp_fov], axis=0)
                interpolated_list.append(interp_pose)

            if close_loop or view_idx < num_views - 1:
                interpolated_list.append(pose_end)

        interpolated_np = np.asarray(interpolated_list, dtype=np.float32)
        output_batches.append(torch.tensor(interpolated_np, dtype=pose_tensor.dtype, device=pose_tensor.device))

    return torch.stack(output_batches, dim=0)


def enforce_orbit_interpolation(
    interp_pose_enc: torch.Tensor,
    num_input_views: int,
    inter_views_between: int,
    keep_height_constant: bool = True,
) -> torch.Tensor:
    """Force interpolated positions onto a circular orbit in the XZ plane."""
    step = inter_views_between + 1

    for batch_idx in range(interp_pose_enc.shape[0]):
        positions = interp_pose_enc[batch_idx, :, :3].detach().cpu().numpy()
        anchor_indices = [anchor_idx * step for anchor_idx in range(num_input_views)]
        anchor_positions = positions[anchor_indices]

        center = anchor_positions.mean(axis=0)
        center_x, center_y, center_z = center.tolist()

        anchor_xz = anchor_positions[:, [0, 2]]
        center_xz = np.array([center_x, center_z], dtype=np.float32)
        anchor_vectors_xz = anchor_xz - center_xz[None, :]
        anchor_radii = np.linalg.norm(anchor_vectors_xz, axis=1)
        valid = anchor_radii > 1e-6
        if not np.any(valid):
            continue

        target_radius = float(np.median(anchor_radii[valid]))
        target_height = float(np.median(anchor_positions[:, 1])) if keep_height_constant else None

        for seg_idx in range(num_input_views - 1):
            start_anchor = anchor_indices[seg_idx]
            end_anchor = anchor_indices[seg_idx + 1]

            start_pos = positions[start_anchor]
            end_pos = positions[end_anchor]

            start_vec_xz = np.array([start_pos[0] - center_x, start_pos[2] - center_z], dtype=np.float32)
            end_vec_xz = np.array([end_pos[0] - center_x, end_pos[2] - center_z], dtype=np.float32)

            start_norm = np.linalg.norm(start_vec_xz)
            end_norm = np.linalg.norm(end_vec_xz)
            if start_norm < 1e-6 or end_norm < 1e-6:
                continue

            start_angle = float(np.arctan2(start_vec_xz[1], start_vec_xz[0]))
            end_angle = float(np.arctan2(end_vec_xz[1], end_vec_xz[0]))

            delta_angle = end_angle - start_angle
            if delta_angle > np.pi:
                delta_angle -= 2.0 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2.0 * np.pi

            for inner_idx in range(1, inter_views_between + 1):
                current_idx = start_anchor + inner_idx
                t = inner_idx / (inter_views_between + 1)
                angle = start_angle + t * delta_angle
                x = center_x + target_radius * np.cos(angle)
                z = center_z + target_radius * np.sin(angle)
                y = target_height if keep_height_constant else ((1.0 - t) * start_pos[1] + t * end_pos[1])
                positions[current_idx] = np.array([x, y, z], dtype=np.float32)

        interp_pose_enc[batch_idx, :, :3] = torch.from_numpy(positions).to(
            device=interp_pose_enc.device, dtype=interp_pose_enc.dtype,
        )

    return interp_pose_enc


def enforce_constant_fov(
    interp_pose_enc: torch.Tensor,
    reference_pose_enc: torch.Tensor,
) -> torch.Tensor:
    """Set all FOV values to match the reference pose's FOV."""
    reference_fov = reference_pose_enc[:, :1, 7:9]
    interp_pose_enc[:, :, 7:9] = reference_fov.expand(-1, interp_pose_enc.shape[1], -1)
    return interp_pose_enc


def normalize_pose_translation_scale_to_prediction(
    gt_pose_enc: torch.Tensor,
    pred_pose_enc: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, float | None]:
    """Rescale ground-truth translations to match the predicted scale."""
    gt_translation = gt_pose_enc[:, 1:, :3]
    pred_translation = pred_pose_enc[:, 1:, :3]

    gt_norm = torch.norm(gt_translation, dim=-1)
    pred_norm = torch.norm(pred_translation, dim=-1)

    valid = gt_norm > 1e-6
    if not torch.any(valid):
        return gt_pose_enc, None

    ratio = pred_norm[valid] / gt_norm[valid]
    scale = torch.median(ratio)
    scale = torch.clamp(scale, min=scale_min, max=scale_max)

    normalized_pose_enc = gt_pose_enc.clone()
    normalized_pose_enc[..., :3] = normalized_pose_enc[..., :3] * scale
    return normalized_pose_enc, float(scale.item())


def normalize_extrinsics_to_c2w(extrinsics: torch.Tensor, convention: str) -> torch.Tensor:
    """Ensure extrinsics are in camera-to-world convention."""
    if convention == "c2w":
        return extrinsics
    if convention == "w2c":
        return torch.linalg.inv(extrinsics)
    raise ValueError("extrinsics_convention must be 'w2c' or 'c2w'")


def get_anchor_indices(num_input_views: int, inter_views_between: int) -> list[int]:
    """Return indices of anchor (original) views in the interpolated sequence."""
    step = inter_views_between + 1
    return [anchor_idx * step for anchor_idx in range(num_input_views)]


def reorder_anchor_first(
    pose_enc: torch.Tensor,
    anchor_indices: list[int],
) -> tuple[torch.Tensor, list[int]]:
    """Move anchor views to the beginning of the sequence."""
    num_views = pose_enc.shape[1]
    anchor_set = set(anchor_indices)
    other_indices = [idx for idx in range(num_views) if idx not in anchor_set]
    reordered_indices = anchor_indices + other_indices
    reordered_pose = pose_enc[:, reordered_indices, :]
    new_anchor_indices = list(range(len(anchor_indices)))
    return reordered_pose, new_anchor_indices


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_multiview_images(
    images: torch.Tensor,
    save_dir: str,
    test_id: str,
    frame: str,
    view_names: list[int] | None = None,
) -> None:
    """Save rendered multi-view images."""
    frame_tag = f"{frame}_baseline" if view_names is not None else f"{frame}_multiview"
    frame_save_dir = os.path.join(save_dir, test_id, frame_tag)
    os.makedirs(frame_save_dir, exist_ok=True)

    for index in range(images.size(0)):
        if view_names is None:
            filename = os.path.join(frame_save_dir, f"{index:03d}.png")
        else:
            filename = os.path.join(frame_save_dir, f"{view_names[index]:02d}.png")
        save_image(images[index], filename, normalize=False)

    print(f"Saved multiview images to {frame_save_dir}")


def save_anchor_images(
    images: torch.Tensor,
    save_dir: str,
    test_id: str,
    frame: str,
    anchor_indices: list[int],
    view_ids: list[int],
) -> None:
    """Save anchor view images separately."""
    anchor_dir = os.path.join(save_dir, test_id, f"{frame}_anchors")
    os.makedirs(anchor_dir, exist_ok=True)

    for anchor_rank, (anchor_index, view_id) in enumerate(zip(anchor_indices, view_ids)):
        if anchor_index >= images.size(0):
            continue
        image = images[anchor_index]
        save_image(image, os.path.join(anchor_dir, f"view_{view_id:02d}.png"), normalize=False)
        save_image(image, os.path.join(anchor_dir, f"anchor_{anchor_rank:02d}_idx_{anchor_index:03d}.png"), normalize=False)

    print(f"Saved anchor images to {anchor_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="360-degree multi-view Gaussian rendering")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing NPZ files")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output/multiview", help="Output directory")
    parser.add_argument("--input-views", type=str, default="25,1,13,37", help="4 comma-separated input view ids")
    parser.add_argument("--inter-views-between", type=int, default=4, help="Number of interpolated views between anchors")
    parser.add_argument("--close-loop", action="store_true", help="Close the interpolation loop")
    parser.add_argument("--render-baseline-only", action="store_true", help="Only render baseline views (no interpolation)")
    parser.add_argument("--interpolation-mode", type=str, default="linear", choices=["linear", "orbit"],
                       help="Interpolation mode: linear or orbit (circular XZ plane)")
    parser.add_argument("--keep-fov-constant", action="store_true", default=True, help="Use constant FOV across all views")
    parser.add_argument("--keep-height-constant", action="store_true", default=True, help="Keep camera height constant (orbit mode)")
    parser.add_argument("--anchor-first-output", action="store_true", default=True, help="Place anchor views first in output")
    parser.add_argument("--extrinsics-convention", type=str, default="w2c", choices=["w2c", "c2w"],
                       help="Extrinsics convention in NPZ files")
    parser.add_argument("--agg-input-size", type=int, default=518, help="Low-resolution input size")
    parser.add_argument("--input-size", type=int, default=2072, help="High-resolution input size")
    parser.add_argument("--render-size", type=int, default=2072, help="Render output resolution")
    parser.add_argument("--normalize-gt-scale", action="store_true", default=True,
                       help="Normalize GT translation scale to predicted scale")
    parser.add_argument("--normalize-scale-min", type=float, default=0.25)
    parser.add_argument("--normalize-scale-max", type=float, default=4.0)
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--bg-color", type=str, default="1.0,1.0,1.0",
                       help="Background colour as comma-separated R,G,B in [0,1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_view_ids = parse_view_ids(args.input_views)
    if len(input_view_ids) != 4:
        raise ValueError("--input-views must contain exactly 4 view ids")

    bg_values = [float(x.strip()) for x in args.bg_color.split(",")]
    bg_color = torch.tensor(bg_values, dtype=torch.float32)

    model, device = load_model(checkpoint_path=args.checkpoint_path, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    anchor_indices = get_anchor_indices(len(input_view_ids), args.inter_views_between)
    print(
        f"[Multiview] GT anchor indices (1-based): {[i + 1 for i in anchor_indices]}, "
        f"view_ids={input_view_ids}, inter_views_between={args.inter_views_between}"
    )
    print(f"[Multiview] extrinsics_convention={args.extrinsics_convention}")
    print(f"[Multiview] render_baseline_only={args.render_baseline_only}")
    print(f"[Multiview] anchor_first_output={args.anchor_first_output}")

    subject_ids = sorted(os.listdir(args.data_root), key=sort_key_maybe_int)
    for subject_id in subject_ids:
        subject_dir = os.path.join(args.data_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        npz_files = sorted(glob.glob(os.path.join(subject_dir, "*.npz")))
        for npz_file in npz_files:
            sample = prepare_sample(
                npz_path=npz_file, input_view_ids=input_view_ids,
                agg_input_size=args.agg_input_size, input_size=args.input_size,
            )

            images_lr = sample["images_lr"].unsqueeze(0).to(device)
            images_hr = sample["images_hr"].unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(images_lr, images_hr, mask_gaussian=True, use_gt_mask=False, if_train=False)

                extrinsics_c2w = normalize_extrinsics_to_c2w(
                    sample["extrinsics"], convention=args.extrinsics_convention,
                )
                relative_extrinsics = convert_extrinsics_to_relative_tensor(extrinsics_c2w)
                gt_pose_enc = extri_intri_to_pose_encoding(
                    relative_extrinsics.unsqueeze(0).to(device),
                    sample["intrinsics_input"].unsqueeze(0).to(device),
                    image_size_hw=(args.input_size, args.input_size),
                )

                if args.normalize_gt_scale:
                    gt_pose_enc, scale_value = normalize_pose_translation_scale_to_prediction(
                        gt_pose_enc, preds["pose_enc_pre"].to(dtype=torch.float32),
                        scale_min=args.normalize_scale_min, scale_max=args.normalize_scale_max,
                    )
                    if scale_value is not None:
                        print(f"[Multiview] normalized GT translation scale={scale_value:.4f}")

                if args.render_baseline_only:
                    preds["pose_enc"] = gt_pose_enc.to(dtype=torch.float32)
                    anchor_indices_render = list(range(len(input_view_ids)))
                else:
                    interpolated_pose_enc = interpolate_pose_open(
                        gt_pose_enc, inter_views=args.inter_views_between, close_loop=args.close_loop,
                    )

                    if args.interpolation_mode == "orbit":
                        interpolated_pose_enc = enforce_orbit_interpolation(
                            interpolated_pose_enc,
                            num_input_views=len(input_view_ids),
                            inter_views_between=args.inter_views_between,
                            keep_height_constant=args.keep_height_constant,
                        )

                    if args.keep_fov_constant:
                        interpolated_pose_enc = enforce_constant_fov(interpolated_pose_enc, gt_pose_enc)

                    anchor_indices_render = anchor_indices
                    if args.anchor_first_output:
                        interpolated_pose_enc, anchor_indices_render = reorder_anchor_first(
                            interpolated_pose_enc, anchor_indices,
                        )

                    preds["pose_enc"] = interpolated_pose_enc.to(dtype=torch.float32)

                view_count = preds["pose_enc"].shape[1]
                bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()
                rendered_images, _ = batch_render_images_my(
                    preds, wo_bg=True, sr_image_size=args.render_size, bg_color=bg,
                )

            frame = os.path.splitext(os.path.basename(npz_file))[0]
            save_multiview_images(
                images=rendered_images.detach().cpu().clone(),
                save_dir=args.output_dir, test_id=subject_id, frame=frame,
                view_names=input_view_ids if args.render_baseline_only else None,
            )

            save_anchor_images(
                images=rendered_images.detach().cpu().clone(),
                save_dir=args.output_dir, test_id=subject_id, frame=frame,
                anchor_indices=anchor_indices_render, view_ids=input_view_ids,
            )


if __name__ == "__main__":
    main()
