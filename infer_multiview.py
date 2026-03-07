import glob
import os

import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.rendering.render_image import batch_render_images_my
from vggt.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


class InferenceConfig:
    test_path = "/home/china/lab/VGGT_human/experiment/test_data/dna_timeseq"
    checkpoint_path = "./checkpoints/checkpoint_epoch_50_loss_0.2110.pt"
    output_dir = "output/dna_timeseq_multiview"

    agg_input_size = 518
    input_size = 2072
    render_size = 2072

    view_ids = [25, 1, 13, 37]
    inter_views_between = 4
    close_loop = False
    render_baseline_only = False
    extrinsics_convention = "w2c"  # "w2c" or "c2w"
    keep_fov_constant = True
    interpolation_mode = "linear"  # "orbit" or "linear"
    anchor_first_output = True
    keep_height_constant = True

    bg_color = [1.0, 1.0, 1.0]
    device = None
    normalize_gt_scale_to_prediction = True
    normalize_scale_min = 0.25
    normalize_scale_max = 4.0


def load_model(checkpoint_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = VGGT.from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, device


def read_dna_npz_entry(npz_path: str, view_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    intrinsics = []
    extrinsics = []

    with np.load(npz_path, allow_pickle=True) as archive:
        for view in view_ids:
            data_key = f"view_{view:02d}"
            if data_key not in archive:
                raise KeyError(f"{data_key} not found in {npz_path}.")
            entry = archive[data_key].item()

            rgb_image = torchvision.io.decode_image(
                torch.from_numpy(entry["image"]), mode=torchvision.io.ImageReadMode.RGB
            ).float() / 255.0

            images.append(rgb_image)
            intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))
            extrinsics.append(torch.from_numpy(entry["extrinsic"]).to(dtype=torch.float32))

    return (
        torch.stack(images, dim=0),
        torch.stack(intrinsics, dim=0),
        torch.stack(extrinsics, dim=0),
    )


def prepare_sample(npz_path: str, input_view_ids: list[int], agg_input_size: int, input_size: int) -> dict[str, torch.Tensor]:
    images, intrinsics, extrinsics = read_dna_npz_entry(npz_path, input_view_ids)

    origin_h, origin_w = images.shape[-2:]
    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False
    )

    intrinsics_input = adjust_intrinsic_batch(
        intrinsics,
        orig_w=origin_w,
        orig_h=origin_h,
        target_w=input_size,
        target_h=input_size,
    )

    return {
        "images_lr": images_lr,
        "images_hr": images_hr,
        "intrinsics_input": intrinsics_input,
        "extrinsics": extrinsics,
    }


def interpolate_pose_open(pose_tensor: torch.Tensor, inter_views: int, close_loop: bool = False) -> torch.Tensor:
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

            pos_start = pose_start[:3]
            pos_end = pose_end[:3]

            quat_start = pose_start[3:7]
            quat_end = pose_end[3:7]

            fov_start = pose_start[7:9]
            fov_end = pose_end[7:9]

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
        if keep_height_constant:
            target_height = float(np.median(anchor_positions[:, 1]))

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

                if keep_height_constant:
                    y = target_height
                else:
                    y = (1.0 - t) * start_pos[1] + t * end_pos[1]

                positions[current_idx] = np.array([x, y, z], dtype=np.float32)

        interp_pose_enc[batch_idx, :, :3] = torch.from_numpy(positions).to(
            device=interp_pose_enc.device,
            dtype=interp_pose_enc.dtype,
        )

    return interp_pose_enc


def enforce_constant_fov(interp_pose_enc: torch.Tensor, reference_pose_enc: torch.Tensor) -> torch.Tensor:
    reference_fov = reference_pose_enc[:, :1, 7:9]
    interp_pose_enc[:, :, 7:9] = reference_fov.expand(-1, interp_pose_enc.shape[1], -1)
    return interp_pose_enc


def normalize_pose_translation_scale_to_prediction(
    gt_pose_enc: torch.Tensor,
    pred_pose_enc: torch.Tensor,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, float | None]:
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
    if convention == "c2w":
        return extrinsics
    if convention == "w2c":
        return torch.linalg.inv(extrinsics)
    raise ValueError("InferenceConfig.extrinsics_convention 只能是 'w2c' 或 'c2w'")


def get_anchor_indices(num_input_views: int, inter_views_between: int) -> list[int]:
    step = inter_views_between + 1
    return [anchor_idx * step for anchor_idx in range(num_input_views)]


def reorder_anchor_first(
    pose_enc: torch.Tensor,
    anchor_indices: list[int],
) -> tuple[torch.Tensor, list[int]]:
    num_views = pose_enc.shape[1]
    anchor_set = set(anchor_indices)
    other_indices = [idx for idx in range(num_views) if idx not in anchor_set]

    reordered_indices = anchor_indices + other_indices
    reordered_pose = pose_enc[:, reordered_indices, :]
    new_anchor_indices = list(range(len(anchor_indices)))
    return reordered_pose, new_anchor_indices


def save_render_images(images: torch.Tensor, save_dir: str, test_id: str, frame: str, view_names: list[int] | None = None):
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
):
    anchor_dir = os.path.join(save_dir, test_id, f"{frame}_anchors")
    os.makedirs(anchor_dir, exist_ok=True)

    for anchor_rank, (anchor_index, view_id) in enumerate(zip(anchor_indices, view_ids)):
        if anchor_index >= images.size(0):
            continue
        image = images[anchor_index]
        save_image(image, os.path.join(anchor_dir, f"view_{view_id:02d}.png"), normalize=False)
        save_image(image, os.path.join(anchor_dir, f"anchor_{anchor_rank:02d}_idx_{anchor_index:03d}.png"), normalize=False)

    print(f"Saved anchor images to {anchor_dir}")


def sort_key_maybe_int(name: str):
    return int(name) if name.isdigit() else name


def main():
    config = InferenceConfig()

    input_view_ids = list(config.view_ids)
    if len(input_view_ids) != 4:
        raise ValueError("InferenceConfig.view_ids 必须是 4 个视角。")

    bg_color = torch.tensor(config.bg_color, dtype=torch.float32)
    if bg_color.numel() != 3:
        raise ValueError("InferenceConfig.bg_color 必须是三个值，例如 [1.0, 1.0, 1.0]")

    model, device = load_model(checkpoint_path=config.checkpoint_path, device=config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    anchor_indices = get_anchor_indices(len(input_view_ids), config.inter_views_between)
    anchor_indices_1based = [index + 1 for index in anchor_indices]
    print(
        "[Multiview] GT anchor indices (1-based): "
        f"{anchor_indices_1based}, view_ids={input_view_ids}, inter_views_between={config.inter_views_between}"
    )
    print(f"[Multiview] extrinsics_convention={config.extrinsics_convention}")
    print(f"[Multiview] render_baseline_only={config.render_baseline_only}")
    print(f"[Multiview] anchor_first_output={config.anchor_first_output}")

    subject_ids = sorted(os.listdir(config.test_path), key=sort_key_maybe_int)
    for subject_id in subject_ids:
        subject_dir = os.path.join(config.test_path, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        npz_files = sorted(glob.glob(os.path.join(subject_dir, "*.npz")))
        for npz_file in npz_files:
            sample = prepare_sample(
                npz_path=npz_file,
                input_view_ids=input_view_ids,
                agg_input_size=config.agg_input_size,
                input_size=config.input_size,
            )

            images_lr = sample["images_lr"].unsqueeze(0).to(device)
            images_hr = sample["images_hr"].unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(
                    images_lr,
                    images_hr,
                    mask_gaussian=True,
                    use_gt_mask=False,
                    if_train=False,
                )

                extrinsics_c2w = normalize_extrinsics_to_c2w(
                    sample["extrinsics"],
                    convention=config.extrinsics_convention,
                )
                relative_extrinsics = convert_extrinsics_to_relative_tensor(extrinsics_c2w)
                gt_pose_enc = extri_intri_to_pose_encoding(
                    relative_extrinsics.unsqueeze(0).to(device),
                    sample["intrinsics_input"].unsqueeze(0).to(device),
                    image_size_hw=(config.input_size, config.input_size),
                )

                if config.normalize_gt_scale_to_prediction:
                    gt_pose_enc, scale_value = normalize_pose_translation_scale_to_prediction(
                        gt_pose_enc,
                        preds["pose_enc_pre"].to(dtype=torch.float32),
                        scale_min=config.normalize_scale_min,
                        scale_max=config.normalize_scale_max,
                    )
                    if scale_value is not None:
                        print(f"[Multiview] normalized GT translation scale={scale_value:.4f}")

                if config.render_baseline_only:
                    preds["pose_enc"] = gt_pose_enc.to(dtype=torch.float32)
                    anchor_indices_render = list(range(len(input_view_ids)))
                else:
                    interpolated_pose_enc = interpolate_pose_open(
                        gt_pose_enc,
                        inter_views=config.inter_views_between,
                        close_loop=config.close_loop,
                    )

                    if config.interpolation_mode == "orbit":
                        interpolated_pose_enc = enforce_orbit_interpolation(
                            interpolated_pose_enc,
                            num_input_views=len(input_view_ids),
                            inter_views_between=config.inter_views_between,
                            keep_height_constant=config.keep_height_constant,
                        )
                    elif config.interpolation_mode != "linear":
                        raise ValueError("InferenceConfig.interpolation_mode 只能是 'orbit' 或 'linear'")

                    if config.keep_fov_constant:
                        interpolated_pose_enc = enforce_constant_fov(interpolated_pose_enc, gt_pose_enc)

                    anchor_indices_render = anchor_indices
                    if config.anchor_first_output:
                        interpolated_pose_enc, anchor_indices_render = reorder_anchor_first(
                            interpolated_pose_enc,
                            anchor_indices,
                        )

                    preds["pose_enc"] = interpolated_pose_enc.to(dtype=torch.float32)

                view_count = preds["pose_enc"].shape[1]
                bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()
                rendered_images, _ = batch_render_images_my(
                    preds,
                    wo_bg=True,
                    sr_image_size=config.render_size,
                    bg_color=bg,
                )

            frame = os.path.splitext(os.path.basename(npz_file))[0]
            save_render_images(
                images=rendered_images.detach().cpu().clone(),
                save_dir=config.output_dir,
                test_id=subject_id,
                frame=frame,
                view_names=input_view_ids if config.render_baseline_only else None,
            )

            save_anchor_images(
                images=rendered_images.detach().cpu().clone(),
                save_dir=config.output_dir,
                test_id=subject_id,
                frame=frame,
                anchor_indices=anchor_indices_render,
                view_ids=input_view_ids,
            )


if __name__ == "__main__":
    main()
