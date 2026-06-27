"""Single-frame prediction-vs-ground-truth comparison.

Renders a specified target view and saves the predicted image alongside the
ground-truth image (with background compositing).

Example::

    python infer_frame.py \\
        --data-root ./test_data \\
        --checkpoint-path ./checkpoints/checkpoint_finetune.pt \\
        --input-views 25,37,1,13 \\
        --target-view 25 \\
        --output-dir output/frame
"""

from __future__ import annotations

import argparse
import os

import torch

from vggt.rendering.render_image import batch_render_images_my
from vggt.utils.inference_utils import (
    collect_npz_files,
    compute_translation_alignment_scale,
    ensure_render_float32,
    load_model,
    parse_view_ids,
    read_dna_npz_views,
    save_frame_image,
)
from vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


def prepare_sample(
    npz_path: str,
    input_view_ids: list[int],
    agg_input_size: int,
    input_size: int,
) -> dict:
    """Read NPZ with masks and resize images for the aggregator and GS head."""
    images, masks, intrinsics, extrinsics, mask_from_npz = read_dna_npz_views(npz_path, input_view_ids)

    images_lr = torch.nn.functional.interpolate(
        images, size=(agg_input_size, agg_input_size), mode="bilinear", align_corners=False,
    )
    images_hr = torch.nn.functional.interpolate(
        images, size=(input_size, input_size), mode="bilinear", align_corners=False,
    )

    return {
        "images": images,
        "masks": masks,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "mask_from_npz": mask_from_npz,
        "images_lr": images_lr,
        "images_hr": images_hr,
    }


def run_model_on_npz(
    npz_path: str,
    model,
    device: str,
    input_view_ids: list[int],
    agg_input_size: int,
    input_size: int,
) -> tuple[dict, dict]:
    """Run the VGGT model on a single NPZ file."""
    data = prepare_sample(
        npz_path=npz_path, input_view_ids=input_view_ids,
        agg_input_size=agg_input_size, input_size=input_size,
    )

    images_lr = data["images_lr"].unsqueeze(0).to(device=device, dtype=torch.float32)
    images_hr = data["images_hr"].unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(images_lr, images_hr, mask_gaussian=True, use_gt_mask=False, if_train=False)

    predictions = {key: value for key, value in predictions.items()}
    return predictions, data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-frame pred vs GT comparison")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing NPZ files")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="output/frame", help="Output directory")
    parser.add_argument("--input-views", type=str, default="25,37,1,13", help="Comma-separated input view ids")
    parser.add_argument("--target-view", type=int, required=True,
                       help="Target view id (must be in --input-views)")
    parser.add_argument("--agg-input-size", type=int, default=518, help="Low-resolution input size")
    parser.add_argument("--input-size", type=int, default=2072, help="High-resolution input size")
    parser.add_argument("--render-size", type=int, default=2072, help="Render output resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--bg-color", type=str, default="1.0,1.0,1.0",
                       help="Background colour as comma-separated R,G,B in [0,1]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_view_ids = parse_view_ids(args.input_views)
    if args.target_view not in input_view_ids:
        raise ValueError(
            f"target_view={args.target_view} is not in input_views={input_view_ids}. "
            f"Please add the target view to --input-views."
        )

    target_index = input_view_ids.index(args.target_view)

    pred_dir = os.path.join(args.output_dir, f"pred_{args.target_view}")
    gt_dir = os.path.join(args.output_dir, f"gt_{args.target_view}")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    model, device = load_model(checkpoint_path=args.checkpoint_path, device=args.device)
    npz_files = collect_npz_files(args.data_root)

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found in {args.data_root}")

    bg_values = [float(x.strip()) for x in args.bg_color.split(",")]
    bg_color = torch.tensor(bg_values, dtype=torch.float32)

    print(f"Found {len(npz_files)} npz files.")
    missing_npz_mask_warned = False

    for npz_path in npz_files:
        preds, data = run_model_on_npz(
            npz_path=npz_path, model=model, device=device,
            input_view_ids=input_view_ids, agg_input_size=args.agg_input_size, input_size=args.input_size,
        )

        pred_input_extrinsics, pred_input_intrinsics = pose_encoding_to_extri_intri(
            preds["pose_enc_pre"].float(),
            image_size_hw=(args.render_size, args.render_size),
        )

        gt_input_extrinsics = data["extrinsics"].unsqueeze(0).to(device=device, dtype=torch.float32)
        base_c2w = torch.linalg.inv(gt_input_extrinsics[:, 0:1, :, :])
        gt_rel_input_extrinsics = torch.matmul(gt_input_extrinsics, base_c2w)

        gt_target_rel_extrinsic = gt_rel_input_extrinsics[:, target_index:target_index + 1, :, :].clone()

        translation_scale = compute_translation_alignment_scale(
            source_extrinsics=gt_rel_input_extrinsics,
            target_extrinsics=pred_input_extrinsics,
        )
        gt_target_rel_extrinsic[:, :, :3, 3] = (
            gt_target_rel_extrinsic[:, :, :3, 3] * translation_scale.view(-1, 1, 1)
        )

        target_intrinsic = pred_input_intrinsics[:, target_index:target_index + 1, :, :]
        preds["pose_enc"] = extri_intri_to_pose_encoding(
            extrinsics=gt_target_rel_extrinsic[:, :, :3, :4],
            intrinsics=target_intrinsic,
            image_size_hw=(args.render_size, args.render_size),
        ).float()
        preds = ensure_render_float32(preds)

        view_count = preds["pose_enc"].shape[1]
        bg = bg_color.view(1, 1, 3).to(device).expand(1, view_count, 3).contiguous()
        rendered_images, _ = batch_render_images_my(
            preds, wo_bg=True, sr_image_size=args.render_size, bg_color=bg,
        )

        pred_image = rendered_images.squeeze(0).detach().cpu()
        gt_image = data["images"][target_index].detach().cpu()

        if data["mask_from_npz"][target_index]:
            gt_mask = data["masks"][target_index].detach().cpu()
        elif "masks" in preds:
            pred_mask = preds["masks"][:, target_index:target_index + 1, ...]
            pred_mask = pred_mask.squeeze(0)
            if pred_mask.ndim == 4 and pred_mask.shape[-1] == 1:
                pred_mask = pred_mask.permute(0, 3, 1, 2)
            pred_mask = torch.nn.functional.interpolate(
                pred_mask.float(), size=gt_image.shape[-2:], mode="bilinear", align_corners=False,
            )
            gt_mask = (pred_mask[0] > 0.5).float().detach().cpu()
        else:
            gt_mask = data["masks"][target_index].detach().cpu()

        gt_image = gt_image * gt_mask + (1 - gt_mask) * bg_color.view(3, 1, 1)

        if (not data["mask_from_npz"][target_index]) and (not missing_npz_mask_warned):
            print("Warning: no mask field found in npz; using predicted mask for gt foreground.")
            missing_npz_mask_warned = True

        frame_name = os.path.splitext(os.path.basename(npz_path))[0]
        save_frame_image(pred_image, pred_dir, frame_name)
        save_frame_image(gt_image, gt_dir, frame_name)

        print(f"Processed {frame_name}")

    print(f"Done. Predictions saved to: {pred_dir}")
    print(f"Done. Inputs saved to: {gt_dir}")


if __name__ == "__main__":
    main()
