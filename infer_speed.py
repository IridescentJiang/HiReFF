"""Speed benchmark for VGGT inference.

Measures model forward-pass time and rendering time, reporting per-sample and
average FPS after a configurable number of warmup steps (to exclude CUDA
kernel compilation).

Example::

    python infer_speed.py \\
        --data-root ./test_data \\
        --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \\
        --input-views 25,1,13,37 \\
        --warmup-steps 3
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import torch
from torchvision.utils import save_image

from vggt.rendering.render_image import batch_render_images_my
from vggt.utils.inference_utils import (
    load_model,
    parse_view_ids,
    read_dna_npz_entry,
    sort_key_maybe_int,
)
from vggt.utils.load_fn import adjust_intrinsic_batch


# ---------------------------------------------------------------------------
# Sample preparation (script-specific: timing needs explicit control)
# ---------------------------------------------------------------------------

def prepare_sample(
    npz_path: str,
    input_view_ids: list[int],
    input_size: int,
    agg_input_size: int,
) -> dict[str, torch.Tensor]:
    """Read NPZ and resize images. Returns dict with images_lr, images_hr, intrinsics_lr, extrinsics."""
    images, intrinsics, extrinsics = read_dna_npz_entry(npz_path, input_view_ids)

    origin_h, origin_w = images.shape[-2:]
    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False,
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False,
    )

    intrinsics_lr = adjust_intrinsic_batch(
        intrinsics, orig_w=origin_w, orig_h=origin_h,
        target_w=input_size, target_h=input_size,
    )

    return {
        "images_lr": images_lr,
        "images_hr": images_hr,
        "intrinsics_lr": intrinsics_lr,
        "extrinsics": extrinsics,
    }


def save_single_render(images: torch.Tensor, save_dir: str, test_id: str, frame: str, view_id: int):
    """Save a single rendered image to ``{save_dir}/{test_id}/{frame}_pred/{view_id:02d}.png``."""
    frame_save_dir = os.path.join(save_dir, test_id, f"{frame}_pred")
    os.makedirs(frame_save_dir, exist_ok=True)
    for index in range(images.size(0)):
        filename = os.path.join(frame_save_dir, f"{view_id:02d}.png")
        save_image(images[index], filename, normalize=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed benchmark for VGGT inference")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing NPZ files")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output/benchmark", help="Output directory for rendered images")
    parser.add_argument("--input-views", type=str, default="25,1,13,37", help="Comma-separated input view ids")
    parser.add_argument("--agg-input-size", type=int, default=518, help="Low-resolution input size")
    parser.add_argument("--input-size", type=int, default=2072, help="High-resolution input size")
    parser.add_argument("--render-size", type=int, default=2072, help="Render output resolution")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Number of warmup forward passes")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--bg-color", type=str, default="1.0,1.0,1.0",
                       help="Background colour as comma-separated R,G,B in [0,1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_view_ids = parse_view_ids(args.input_views)
    if len(input_view_ids) == 0:
        raise ValueError("--input-views must not be empty")

    bg_values = [float(x.strip()) for x in args.bg_color.split(",")]
    bg_color = torch.tensor(bg_values, dtype=torch.float32)

    model, device = load_model(checkpoint_path=args.checkpoint_path, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    sample_times: list[float] = []
    render_times: list[float] = []
    profiled_samples = 0
    warmup_counter = 0

    def sync_if_cuda():
        if "cuda" in str(device):
            torch.cuda.synchronize(device=device)

    subject_ids = sorted(os.listdir(args.data_root), key=sort_key_maybe_int)
    for subject_id in subject_ids:
        subject_dir = os.path.join(args.data_root, subject_id)
        if not os.path.isdir(subject_dir):
            continue

        npz_files = sorted(glob.glob(os.path.join(subject_dir, "*.npz")))
        for npz_file in npz_files:
            sample = prepare_sample(
                npz_path=npz_file, input_view_ids=input_view_ids,
                input_size=args.input_size, agg_input_size=args.agg_input_size,
            )

            images_lr = sample["images_lr"].unsqueeze(0).to(device)
            images_hr = sample["images_hr"].unsqueeze(0).to(device)

            measure_this_step = warmup_counter >= args.warmup_steps
            if measure_this_step:
                sync_if_cuda()
                start_time = time.perf_counter()

            with torch.no_grad():
                preds = model(images_lr, images_hr, mask_gaussian=True, use_gt_mask=False, if_train=False)
                preds["pose_enc"] = preds["pose_enc_pre"][:, :1].to(dtype=torch.float32)

                view_count = preds["pose_enc"].shape[1]
                bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()

                if measure_this_step:
                    sync_if_cuda()
                    render_start_time = time.perf_counter()

                rendered_images, _ = batch_render_images_my(
                    preds, wo_bg=True, sr_image_size=args.render_size, bg_color=bg,
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
                n_rendered = rendered_images.shape[0]
                image_fps = n_rendered / render_elapsed if render_elapsed > 0 else float("inf")

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
            save_single_render(
                images=rendered_images.detach().cpu().clone(),
                save_dir=args.output_dir, test_id=subject_id,
                frame=frame, view_id=input_view_ids[0],
            )

    # Summary
    if profiled_samples > 0:
        avg_time = sum(sample_times) / len(sample_times)
        avg_render_time = sum(render_times) / len(render_times)
        avg_sample_fps = (1.0 / avg_time) if avg_time > 0 else float("inf")
        avg_render_image_fps = (1.0 / avg_render_time * rendered_images.shape[0]) if avg_render_time > 0 else float("inf")
        print("=" * 80)
        print(f"[FPS Summary] profiled_samples={profiled_samples} (warmup={args.warmup_steps})")
        print(f"[FPS Summary] avg_total_time_per_sample={avg_time * 1000:.2f} ms")
        print(f"[FPS Summary] avg_render_time_per_sample={avg_render_time * 1000:.2f} ms")
        print(f"[FPS Summary] avg_sample_fps={avg_sample_fps:.2f}")
        print(f"[FPS Summary] avg_render_image_fps={avg_render_image_fps:.2f}")
        print("=" * 80)
    else:
        print("[FPS Summary] No profiled samples. Consider reducing warmup_steps.")


if __name__ == "__main__":
    main()
