import glob
import json
import os
import time

import numpy as np
import torch
import torchvision
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.rendering.render_image import batch_render_images_my
from vggt.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


class InferenceConfig:
    test_path = "/home/china/lab/VGGT_human/experiment/test_data/dna_timeseq"
    checkpoint_path = "./checkpoints/checkpoint_epoch_50_loss_0.2110.pt"
    output_dir = "output/dna_timeseq"
    cam_para_path = "vggt/rendering/dna_rendering_cam_para_render_video.json"

    agg_input_size = 518
    input_size = 2072
    render_size = 2072

    view_ids = [25, 1, 13, 37]
    cam_view_ids = []

    render_mode = "gsplat"
    bg_color = [1.0, 1.0, 1.0]
    device = None
    warmup_steps = 3


def load_model(checkpoint_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = VGGT.from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model, device


def read_dna_npz_entry(npz_path: str, view_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    masks = []
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


def load_cam_para(cam_para_path: str, cam_view_ids: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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


def prepare_sample(npz_path: str, input_view_ids: list[int], input_size: int, agg_input_size: int) -> dict[str, torch.Tensor]:
    images, intrinsics, extrinsics = read_dna_npz_entry(npz_path, input_view_ids)

    origin_h, origin_w = images.shape[-2:]
    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False
    )

    intrinsics_lr = adjust_intrinsic_batch(
        intrinsics,
        orig_w=origin_w,
        orig_h=origin_h,
        target_w=input_size,
        target_h=input_size,
    )

    return {
        "images_lr": images_lr,
        "images_hr": images_hr,
        "intrinsics_lr": intrinsics_lr,
        "extrinsics": extrinsics,
    }


def build_render_pose_enc(
    pred_pose_enc_pre: torch.Tensor,
    source_extrinsics: torch.Tensor,
    render_extrinsics: torch.Tensor,
    render_intrinsics: torch.Tensor,
    input_size: int,
    device: str,
) -> torch.Tensor:
    all_extrinsics = torch.cat([source_extrinsics, render_extrinsics], dim=0)
    all_relative_extrinsics = convert_extrinsics_to_relative_tensor(all_extrinsics)

    source_view_count = source_extrinsics.shape[0]
    source_relative_extrinsics = all_relative_extrinsics[:source_view_count].unsqueeze(0).to(device)
    render_relative_extrinsics = all_relative_extrinsics[source_view_count:].unsqueeze(0).to(device)

    _, adjusted_render_extrinsics = adjust_transl(
        pred_pose_enc_pre,
        source_relative_extrinsics,
        render_relative_extrinsics,
    )

    render_intrinsics_for_input = render_intrinsics.clone()
    render_intrinsics_for_input[:, 0, 2] = input_size / 2.0
    render_intrinsics_for_input[:, 1, 2] = input_size / 2.0

    pose_enc = extri_intri_to_pose_encoding(
        extrinsics=adjusted_render_extrinsics,
        intrinsics=render_intrinsics_for_input.unsqueeze(0).to(device),
        image_size_hw=(input_size, input_size),
    )
    return pose_enc


def save_render_images(images: torch.Tensor, save_dir: str, test_id: str, frame: str, render_view_ids: list[int]):
    frame_save_dir = os.path.join(save_dir, test_id, f"{frame}_pred")
    os.makedirs(frame_save_dir, exist_ok=True)

    for index in range(images.size(0)):
        filename = os.path.join(frame_save_dir, f"{render_view_ids[index]:02d}.png")
        save_image(images[index], filename, normalize=False)

    print(f"Saved images to {frame_save_dir}")


def sort_key_maybe_int(name: str):
    return int(name) if name.isdigit() else name


def main():
    config = InferenceConfig()

    input_view_ids = list(config.view_ids)
    if len(input_view_ids) == 0:
        raise ValueError("InferenceConfig.view_ids 不能为空。")

    bg_color = torch.tensor(config.bg_color, dtype=torch.float32)
    if bg_color.numel() != 3:
        raise ValueError("InferenceConfig.bg_color 必须是三个值，例如 [1.0, 1.0, 1.0]")
    if config.render_mode not in ["gsplat", "mipsplat"]:
        raise ValueError("InferenceConfig.render_mode 只能是 'gsplat' 或 'mipsplat'")

    model, device = load_model(checkpoint_path=config.checkpoint_path, device=config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    sample_times = []
    render_times = []
    sample_fps_list = []
    render_image_fps_list = []
    profiled_samples = 0
    warmup_counter = 0

    def sync_if_cuda():
        if "cuda" in str(device):
            torch.cuda.synchronize(device=device)

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
                input_size=config.input_size,
                agg_input_size=config.agg_input_size,
            )

            images_lr = sample["images_lr"].unsqueeze(0).to(device)
            images_hr = sample["images_hr"].unsqueeze(0).to(device)

            measure_this_step = warmup_counter >= config.warmup_steps
            if measure_this_step:
                sync_if_cuda()
                start_time = time.perf_counter()

            with torch.no_grad():
                preds = model(
                    images_lr,
                    images_hr,
                    mask_gaussian=True,
                    use_gt_mask=False,
                    if_train=False,
                )

                preds["pose_enc"] = preds["pose_enc_pre"][:, :1].to(dtype=torch.float32)

                view_count = preds["pose_enc"].shape[1]
                bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()

                if measure_this_step:
                    sync_if_cuda()
                    render_start_time = time.perf_counter()

                rendered_images, _ = batch_render_images_my(
                    preds,
                    wo_bg=True,
                    sr_image_size=config.render_size,
                    bg_color=bg,
                )

                if measure_this_step:
                    sync_if_cuda()
                    render_elapsed = time.perf_counter() - render_start_time

            if measure_this_step:
                sync_if_cuda()
                elapsed = time.perf_counter() - start_time
                profiled_samples += 1
                sample_times.append(elapsed)
                render_times.append(render_elapsed)

                sample_fps = 1.0 / elapsed if elapsed > 0 else float("inf")
                sample_fps_list.append(sample_fps)

                n_rendered = rendered_images.shape[0]
                image_fps = n_rendered / render_elapsed if render_elapsed > 0 else float("inf")
                render_image_fps_list.append(image_fps)

                print(
                    f"[FPS] {subject_id}/{os.path.basename(npz_file)} | "
                    f"total_time={elapsed * 1000:.2f} ms | "
                    f"render_time={render_elapsed * 1000:.2f} ms | "
                    f"sample_fps={sample_fps:.2f} | "
                    f"render_image_fps={image_fps:.2f}"
                )
            else:
                warmup_counter += 1

            frame = os.path.splitext(os.path.basename(npz_file))[0]
            save_render_images(
                images=rendered_images.detach().cpu().clone(),
                save_dir=config.output_dir,
                test_id=subject_id,
                frame=frame,
                render_view_ids=[input_view_ids[0]],
            )

    if profiled_samples > 0:
        avg_time = sum(sample_times) / len(sample_times)
        avg_render_time = sum(render_times) / len(render_times)
        avg_sample_fps = sum(sample_fps_list) / len(sample_fps_list)
        avg_render_image_fps = sum(render_image_fps_list) / len(render_image_fps_list)
        print("=" * 80)
        print(f"[FPS Summary] profiled_samples={profiled_samples} (warmup={config.warmup_steps})")
        print(f"[FPS Summary] avg_total_time_per_sample={avg_time * 1000:.2f} ms")
        print(f"[FPS Summary] avg_render_time_per_sample={avg_render_time * 1000:.2f} ms")
        print(f"[FPS Summary] avg_sample_fps={avg_sample_fps:.2f}")
        print(f"[FPS Summary] avg_render_image_fps={avg_render_image_fps:.2f}")
        print("=" * 80)
    else:
        print("[FPS Summary] No profiled samples. Consider reducing warmup_steps.")


if __name__ == "__main__":
    main()
