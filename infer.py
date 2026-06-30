"""Multi-view inference from NPZ files.

Given sparse input views, predicts 3D Gaussian parameters and renders novel
views. This is the primary inference entry point.

Example::

    python infer_multiview.py \\
        --data-root ./test_data \\
        --checkpoint-path ./checkpoints/checkpoint_dna_mvh_zju.pt \\
        --input-views 25,1,13,37 \\
        --novel-views 1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46 \\
        --output-dir output/multiview
"""

from __future__ import annotations

import argparse
import os

import torch

from hireff.rendering.render_image import adjust_transl, batch_render_images_my
from hireff.utils.inference_utils import (
    collect_npz_files,
    enforce_colorful_flat_gs,
    ensure_render_float32,
    load_model,
    parse_view_ids,
    read_dna_npz_entry,
    save_render_images,
)
from hireff.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor
from hireff.utils.pose_enc import extri_intri_to_pose_encoding


def load_sample(
    npz_path: str,
    input_view_ids: list[int],
    novel_view_ids: list[int],
    agg_input_size: int,
    input_size: int,
) -> dict:
    """Read NPZ, resize images, adjust intrinsics, and compute relative extrinsics."""
    input_images, input_intrinsics, input_extrinsics = read_dna_npz_entry(npz_path, input_view_ids)
    _, novel_intrinsics, novel_extrinsics = read_dna_npz_entry(npz_path, novel_view_ids)

    intrinsics_all = torch.cat([input_intrinsics, novel_intrinsics], dim=0)
    extrinsics_all = torch.cat([input_extrinsics, novel_extrinsics], dim=0)

    origin_h, origin_w = input_images.shape[-2:]
    images_lr = torch.nn.functional.interpolate(
        input_images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False,
    )
    images_hr = torch.nn.functional.interpolate(
        input_images, size=(input_size, input_size), mode="bilinear", align_corners=False,
    )

    adjusted_intrinsics_all = adjust_intrinsic_batch(
        intrinsics_all,
        orig_w=origin_w, orig_h=origin_h,
        target_w=input_size, target_h=input_size,
    )

    relative_input_extrinsics = convert_extrinsics_to_relative_tensor(input_extrinsics)
    relative_all_extrinsics = convert_extrinsics_to_relative_tensor(extrinsics_all)
    relative_novel_extrinsics = relative_all_extrinsics[-len(novel_extrinsics):]

    return {
        "images_lr": images_lr,
        "images_hr": images_hr,
        "intrinsics": adjusted_intrinsics_all,
        "extrinsics": relative_input_extrinsics,
        "sp_extrinsics": relative_novel_extrinsics,
    }


def run_model_on_npz(
    npz_path: str,
    model,
    device: str,
    input_view_ids: list[int],
    novel_view_ids: list[int],
    agg_input_size: int,
    input_size: int,
) -> tuple[dict, dict]:
    """Run the HiReFF model on a single NPZ file and return predictions + input data."""
    data = load_sample(
        npz_path=npz_path,
        input_view_ids=input_view_ids,
        novel_view_ids=novel_view_ids,
        agg_input_size=agg_input_size,
        input_size=input_size,
    )

    images_lr = data["images_lr"].unsqueeze(0).to(device=device, dtype=torch.float32)
    images_hr = data["images_hr"].unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(images_lr, images_hr, mask_gaussian=True, use_gt_mask=False, if_train=False)

    predictions = {k: v for k, v in predictions.items()}
    return predictions, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-view Gaussian rendering from NPZ files")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing NPZ files (recursive)")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output/multiview", help="Output directory")
    parser.add_argument("--agg-input-size", type=int, default=518, help="Low-resolution input size for the aggregator")
    parser.add_argument("--input-size", type=int, default=2072, help="High-resolution input size")
    parser.add_argument("--render-size", type=int, default=2072, help="Render output resolution")
    parser.add_argument("--input-views", type=str, default="25,1,13,37", help="Comma-separated input view ids")
    parser.add_argument("--novel-views", type=str, default="1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46",
                       help="Comma-separated novel view ids to render")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu), auto-detected if not set")
    parser.add_argument("--max-files", type=int, default=0, help="Max NPZ files to process (0 = all)")
    parser.add_argument("--bg-color", type=str, default="1.0,1.0,1.0",
                       help="Background colour as comma-separated R,G,B in [0,1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_view_ids = parse_view_ids(args.input_views)
    novel_view_ids = parse_view_ids(args.novel_views)
    render_view_ids = input_view_ids + novel_view_ids

    npz_files = collect_npz_files(args.data_root)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found in {args.data_root}")

    if args.max_files > 0:
        npz_files = npz_files[: args.max_files]

    model, device = load_model(args.checkpoint_path, args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    bg_values = [float(x.strip()) for x in args.bg_color.split(",")]
    bg_color = torch.tensor(bg_values, dtype=torch.float32)

    print(f"Found {len(npz_files)} npz files to process")
    for npz_path in npz_files:
        preds, data = run_model_on_npz(
            npz_path=npz_path, model=model, device=device,
            input_view_ids=input_view_ids, novel_view_ids=novel_view_ids,
            agg_input_size=args.agg_input_size, input_size=args.input_size,
        )

        adjusted_extrinsics, adjusted_sp_extrinsics = adjust_transl(
            preds["pose_enc_pre"],
            data["extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32),
            data["sp_extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32),
        )
        all_extrinsics = torch.cat([adjusted_extrinsics, adjusted_sp_extrinsics], dim=1)

        pose_enc = extri_intri_to_pose_encoding(
            extrinsics=all_extrinsics,
            intrinsics=data["intrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32),
            image_size_hw=(data["images_hr"].shape[2], data["images_hr"].shape[3]),
        )
        preds["pose_enc"] = pose_enc.float().to(device)
        preds = ensure_render_float32(preds)
        preds = enforce_colorful_flat_gs(preds)

        view_count = preds["pose_enc"].shape[1]
        bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()
        rendered_images, _ = batch_render_images_my(
            preds, wo_bg=True, sr_image_size=args.render_size, bg_color=bg,
        )
        rendered_images = rendered_images.detach().cpu().clone()

        frame_name = os.path.splitext(os.path.basename(npz_path))[0]
        seq_id = os.path.basename(os.path.dirname(npz_path))
        save_render_images(
            images=rendered_images, output_root=args.output_dir,
            seq_id=seq_id, frame_name=frame_name, view_list=render_view_ids,
        )

        print(f"Processed {seq_id}/{frame_name}")

    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
