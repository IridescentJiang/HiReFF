"""Video rendering from NPZ sequences or image directories.

Supports two input formats:
  1. ``npz`` — a directory of ``frame_XXXX.npz`` files (one per timestep).
  2. ``frame_dir`` — a directory of individual image files per frame.

Outputs an MP4 video and individual frame images.

Example::

    python infer_video.py \\
        --data-root ./wild_images \\
        --checkpoint-path ./checkpoints/8_view_input.pt \\
        --input-views 0,3,5,8 \\
        --inter-view 30 \\
        --fps 18 \\
        --output-dir output/videos
"""

from __future__ import annotations

import argparse
import glob
import os

import torch
from torchvision.utils import save_image

from vggt.rendering.render_image import batch_render_images_my, image_to_video, interpolate_pose
from vggt.utils.inference_utils import (
    load_model,
    parse_view_ids,
    read_dna_npz_entry,
    read_frame_dir_entry,
    save_video_frames,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video rendering from NPZ sequences or image directories")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Root directory: either NPZ sequences or image frame directories")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output/videos",
                       help="Output directory for videos and frames")
    parser.add_argument("--input-views", type=str, default="0,3,5,8",
                       help="Comma-separated input view ids")
    parser.add_argument("--input-format", type=str, default="auto",
                       choices=["npz", "frame_dir", "auto"],
                       help="Input format: npz, frame_dir, or auto-detect")
    parser.add_argument("--inter-view", type=int, default=30,
                       help="Number of interpolated views between adjacent input views")
    parser.add_argument("--agg-input-size", type=int, default=518,
                       help="Low-resolution input size")
    parser.add_argument("--input-size", type=int, default=2072,
                       help="High-resolution input size")
    parser.add_argument("--render-size", type=int, default=2072,
                       help="Render output resolution")
    parser.add_argument("--fps", type=int, default=18,
                       help="Frames per second for output video")
    parser.add_argument("--max-npz-files", type=int, default=150,
                       help="Max number of NPZ files to process per test_id (for npz format)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--bg-color", type=str, default="1.0,1.0,1.0",
                       help="Background colour as comma-separated R,G,B in [0,1]")
    return parser.parse_args()


def detect_input_format(input_path: str) -> str:
    """Detect whether *input_path* contains NPZ files or frame directories."""
    npz_files = glob.glob(os.path.join(input_path, "*.npz"))
    if npz_files:
        return "npz"
    frame_dirs = [p for p in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(p)]
    if frame_dirs:
        return "frame_dir"
    return "none"


def load_sample_npz(npz_path: str, view_ids: list[int], input_size: int, agg_input_size: int) -> dict:
    """Load and resize images from a single NPZ file."""
    images, _, _ = read_dna_npz_entry(npz_path, view_ids)
    return _resize_images(images, input_size, agg_input_size)


def load_sample_frames(frame_dir: str, view_ids: list[int], input_size: int, agg_input_size: int) -> dict:
    """Load and resize images from a directory of frame images."""
    images = read_frame_dir_entry(frame_dir, view_ids)
    return _resize_images(images, input_size, agg_input_size)


def _resize_images(images: torch.Tensor, input_size: int, agg_input_size: int) -> dict:
    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False,
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False,
    )
    return {"images_lr": images_lr, "images_hr": images_hr}


def run_model(data: dict, model, device: str) -> dict:
    """Run the VGGT model on pre-loaded data."""
    images_lr = data["images_lr"].unsqueeze(0).to(device)
    images_hr = data["images_hr"].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(images_lr, images_hr, mask_gaussian=True, use_gt_mask=False, if_train=False)

    return {k: v for k, v in predictions.items()}


def main() -> None:
    args = parse_args()

    view_ids = parse_view_ids(args.input_views)
    model, device = load_model(checkpoint_path=args.checkpoint_path, device=args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    bg_values = [float(x.strip()) for x in args.bg_color.split(",")]
    bg_color = torch.tensor(bg_values, dtype=torch.float32)

    test_ids = sorted(os.listdir(args.data_root))
    for test_id in test_ids:
        input_path = os.path.join(args.data_root, test_id)
        if not os.path.isdir(input_path):
            continue

        # Determine input format
        if args.input_format == "auto":
            input_format = detect_input_format(input_path)
        else:
            input_format = args.input_format

        # Collect sample entries
        if input_format == "npz":
            npz_files = sorted(glob.glob(os.path.join(input_path, "*.npz")))
            sample_entries = [("npz", p) for p in npz_files]
        elif input_format == "frame_dir":
            frame_dirs = sorted([p for p in glob.glob(os.path.join(input_path, "*")) if os.path.isdir(p)])
            sample_entries = [("frame_dir", p) for p in frame_dirs]
        else:
            print(f"Skipping {test_id}: no recognised input format found")
            continue

        rendered_images_f = []
        pose_enc = None
        inter_view = args.inter_view

        for i, (fmt, sample_path) in enumerate(sample_entries):
            # For NPZ format, only process first frame and up to max_npz_files
            if fmt == "npz" and (i == 0 or i > args.max_npz_files):
                continue

            if fmt == "npz":
                data = load_sample_npz(sample_path, view_ids, args.input_size, args.agg_input_size)
            else:
                data = load_sample_frames(sample_path, view_ids, args.input_size, args.agg_input_size)

            preds = run_model(data, model, device)

            # Initialise the smooth trajectory from the first frame's predicted poses
            if pose_enc is None:
                pose_enc = preds["pose_enc_pre"].float()
                pose_enc = interpolate_pose(pose_enc, inter_view=inter_view).to(device)

            _, lens, _ = pose_enc.shape
            preds["pose_enc"] = pose_enc[:, i % lens, :].unsqueeze(1)

            view_count = preds["pose_enc"].shape[1]
            bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()

            rendered_image_f, _ = batch_render_images_my(
                preds, wo_bg=True, sr_image_size=args.render_size, bg_color=bg,
            )
            rendered_images_f.append(rendered_image_f.squeeze(0).cpu().detach())

            frame_name = os.path.splitext(os.path.basename(sample_path))[0] if fmt == "npz" else os.path.basename(sample_path)
            print(f"Processing: {test_id} {frame_name}")

        if not rendered_images_f:
            print(f"No rendered frames for {test_id}, skip output.")
            continue

        rendered_images = torch.stack(rendered_images_f)
        video_path = os.path.join(args.output_dir, f"rendered_video_{test_id}.mp4")
        image_to_video(rendered_images, video_path, fps=args.fps)
        frame_dir = save_video_frames(rendered_images, video_path)
        print(f"Saved video: {video_path}")
        print(f"Saved frames: {frame_dir}")


if __name__ == "__main__":
    main()
