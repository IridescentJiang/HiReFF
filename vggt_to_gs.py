import os
import argparse
import queue
import sys
import threading

import torch
import glob
import time

from matplotlib import pyplot as plt
from torch import Stream

os.environ["NO_PROXY"] = "*"
os.environ["all_proxy"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.rendering.render_image import batch_render_images, save_rendered_images, get_all_poses, image_to_video, \
    interpolate_pose, adjust_transl
from torch.cuda.amp import autocast
from vggt.utils.interpolate import interpolate_images


def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # checkpoint_path = "./checkpoints/checkpoint_epoch_880_loss_0.0297.pt"
    checkpoint_path = "./checkpoints/highRes_model.pt"
    model = VGGT.from_checkpoint(checkpoint_path)

    model.eval()
    model = model.to(device)
    return model, device


def run_model(target_dir, model, device, input_size=518, render_size=518) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "*"))
    image_names = sorted(image_names)

    start_time = time.time_ns() // 1_000_000
    images = load_and_preprocess_images(image_names, target_size=input_size).to(device)
    end_time = time.time_ns() // 1_000_000
    print(f"Processed preprocess images in {end_time - start_time:.2f} ms")

    with torch.no_grad():
        predictions = model(images, if_train=False)

    # predictions["pose_enc"] = predictions["pose_enc_pre"]
    predictions = {k: v for k, v in predictions.items()}

    return predictions


def add_perturbation(pose_enc, perturbation_level=0.1, perturbed_dims=7):
    """
    为 pose_enc 的前7个维度添加随机扰动，最后2个维度保持不变

    参数:
        preds: 包含 pose_enc 的字典
        perturbation_level: 扰动级别 (0.1 = 10%)
        perturbed_dims: 需要扰动的维度数 (默认为7)

    返回:
        添加扰动后的 preds
    """

    # 确保是张量且在正确设备上
    if not isinstance(pose_enc, torch.Tensor):
        pose_enc = torch.tensor(pose_enc, device=pose_enc.device)

    # 分离需要扰动和不需要扰动的部分
    to_perturb = pose_enc[..., :perturbed_dims]  # 前7个维度
    to_keep = pose_enc[..., perturbed_dims:]  # 最后2个维度

    # 计算扰动范围
    abs_values = torch.abs(to_perturb)
    perturbation_range = perturbation_level * abs_values

    # 生成随机扰动 (均匀分布)
    perturbation = torch.empty_like(to_perturb).uniform_(-1, 1) * perturbation_range

    # 添加扰动
    perturbed_part = to_perturb + perturbation

    # 重新组合张量
    perturbed_pose_enc = torch.cat([perturbed_part, to_keep], dim=-1)

    return perturbed_pose_enc


def main():
    parser = argparse.ArgumentParser(description="Convert images to COLMAP format using VGGT")
    parser.add_argument("--image_dir", type=str, default="./examples/dna_rendering_dynamic_video",
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save COLMAP files")
    parser.add_argument("--dynamic_video", action="store_true", default=True,
                        help="Run in batchs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, device = load_model()

    video_ids = ["0018_01"]

    if args.dynamic_video:
        for video_id in video_ids:
            # 创建子目录下的result目录
            input_path = f"./examples/dna_rendering_dynamic_video/1K/{video_id}"

            rendered_images = []
            render_mode = 'gsplat'
            input_size = 1036
            render_size = 2072

            inter_view = 26  # 每两个相邻视角之间插入N个新视角
            pose_enc = None

            stream_start_time = time.time()

            for i, sub_dir in enumerate(sorted(os.listdir(input_path), key=lambda x: int(x))):

                start_time = time.time_ns() // 1_000_000  # 开始计时

                sub_dir_path = os.path.join(input_path, sub_dir)
                if not os.path.isdir(sub_dir_path):
                    continue

                preds = run_model(sub_dir_path, model, device, input_size=input_size, render_size=render_size)

                if pose_enc is None:
                    # pose_enc = get_all_poses(preds["images"], if_mapping=False, if_render_video=True)
                    pose_enc = preds["pose_enc_pre"].float()
                    pose_enc = interpolate_pose(pose_enc, inter_view=inter_view).to(device)
                    _, lens, _ = pose_enc.shape
                # preds["pose_enc"] = pose_enc

                preds["pose_enc"] = pose_enc[:, i % lens, :].unsqueeze(1)
                # preds["pose_enc"] = pose_enc[:, 62, :].unsqueeze(1)
                # preds["pose_enc"] = preds["pose_enc_pre"][:, 2, :].unsqueeze(1).float()

                # render_start_time = time.time_ns() // 1_000_000
                rendered_image, rendered_depth = batch_render_images(preds, wo_bg=True, render_mode=render_mode, sr_image_size=render_size)
                # render_end_time = time.time_ns() // 1_000_000
                # print(f"Processed rendering in {render_end_time - render_start_time:.2f} ms")

                rendered_images.append(rendered_image.squeeze(0).cpu().detach())

                end_time = time.time_ns() // 1_000_000
                print(f"Processed single frame {i} in {end_time - start_time:.2f} ms")

            stream_end_time = time.time()
            print(f"Processed subdirectory {sub_dir} in {stream_end_time - stream_start_time:.2f} seconds")

            video_path = os.path.join(args.output_dir, f"rendered_video_{video_id}.mp4")
            image_to_video(torch.stack(rendered_images), video_path)

            print("\nAll subdirectories processed!")

    else:
        for video_id in video_ids:
            predictions = run_model(f"./examples/dna_rendering_dynamic_video/1K/{video_id}", model)

            video_path = os.path.join(args.output_dir, f"rendered_static_video.mp4")
            image_to_video(predictions, video_path)

            # predictions["pose_enc"] = get_all_poses(predictions["images"], if_mapping=False)
            #
            # rendered_images = render_images(predictions, "All", True)
            # save_rendered_images(rendered_images.detach().cpu().clone(), "render_images/pre", 0, 0)

            print(f"Guaasian rendering video successfully written to {args.output_dir}")


if __name__ == "__main__":
    main()
