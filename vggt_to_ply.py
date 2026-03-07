import os
import argparse
import sys
import torch
import glob


os.environ["NO_PROXY"] = "*"
os.environ["all_proxy"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
sys.path.append("vggt/")

from visual_util import predictions_to_ply
from vggt.models.vggt_origin import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.depth2points import unproject_depth_map_to_point_map
import time
from einops import rearrange


def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pretrained_model_path = "facebook/VGGT-1B"
    model = VGGT.from_pretrained(pretrained_model_path)

    # checkpoint_path = "./checkpoints/best_model_epoch_950_loss_0.1292.pt"
    # model = VGGT.from_checkpoint(checkpoint_path)

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)
    return model, device


def run_model(target_dir, model, use_mask=False) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    if "pose_enc_pre" in predictions.keys():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc_pre"], images.shape[-2:])
    else:
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    with torch.cuda.amp.autocast(enabled=True):
        if "pseudo_label_points" in predictions.keys():
            predictions["world_points_from_depth"] = predictions["world_points"]
        else:
            # Generate world points from depth map
            print("Computing world points from depth map...")
            if "depth" in predictions.keys():
                depth_map = predictions["depth"]  # (S, H, W, 1)
                world_points = unproject_depth_map_to_point_map(depth_map, extrinsic.squeeze(0),
                                                                intrinsic.squeeze(0))
                predictions["world_points_from_depth"] = world_points.unsqueeze(0)

    if use_mask and "masks" in predictions.keys():
        masks = predictions["masks"] > 0.5

        predictions["world_points_from_depth"] = predictions["world_points_from_depth"] * masks
        predictions["world_points"] = predictions["world_points"] * masks

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Clean up
    torch.cuda.empty_cache()
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT")
    parser.add_argument("--image_dir", type=str, default="./examples/dna_rendering_test/1",
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save COLMAP files")
    parser.add_argument("--conf_threshold", type=float, default=20.0,
                        help="Confidence threshold (0-100%) for including points")
    parser.add_argument("--mask_sky", action="store_true",
                        help="Filter out points likely to be sky")
    parser.add_argument("--mask_black_bg", action="store_true",
                        help="Filter out points with very dark/black color")
    parser.add_argument("--mask_white_bg", action="store_true",
                        help="Filter out points with very bright/white color")
    parser.add_argument("--use_mask", action="store_true", default=True,
                        help="Filter out points with predicted mask")
    parser.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                        choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                        help="Which prediction branch to use")
    parser.add_argument("--run_batchs", action="store_true",
                        help="Run in batchs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, device = load_model()

    if args.run_batchs:
        # 创建子目录下的result目录
        output_dir = os.path.join(args.image_dir, "result")
        os.makedirs(output_dir, exist_ok=True)

        for sub_dir in os.listdir(args.image_dir):
            # 构造完整子目录路径
            sub_dir_path = os.path.join(args.image_dir, sub_dir)

            # 跳过非目录文件
            if not os.path.isdir(sub_dir_path):
                continue

            print(f"\nProcessing subdirectory: {sub_dir_path}")

            # 运行模型处理当前子目录
            try:
                predictions = run_model(sub_dir_path, model, args.use_mask)
            except Exception as e:
                print(f"Error processing {sub_dir}: {str(e)}")
                continue

            # 生成PLY文件名（保持与子目录同名）
            ply_file = os.path.join(output_dir, f"{sub_dir}.ply")

            # 转换并保存PLY文件
            ply_scene = predictions_to_ply(
                predictions,
                conf_thres=args.conf_threshold,
                mask_black_bg=args.mask_black_bg,
                mask_white_bg=args.mask_white_bg,
                mask_sky=args.mask_sky,
                prediction_mode=args.prediction_mode,
            )
            ply_scene.export(file_obj=ply_file)

            print(f"Successfully saved PLY to: {ply_file}")

        print("\nAll subdirectories processed!")
    else:
        predictions = run_model(args.image_dir, model, args.use_mask)

        timestamp = time.time_ns()
        plyfile = os.path.join(args.output_dir, f"output_{timestamp}.ply")

        # Convert predictions to GLB
        plyscene = predictions_to_ply(
            predictions,
            conf_thres=args.conf_threshold,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            mask_sky=args.mask_sky,
            prediction_mode=args.prediction_mode,
        )
        plyscene.export(file_obj=plyfile)

        print(f"PLY files successfully written to {args.output_dir}")


if __name__ == "__main__":
    main()
