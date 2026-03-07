import os
import sys
import numpy as np
import torch
import glob

import torchvision
from matplotlib import pyplot as plt
from torch import Stream, Tensor
from torchvision.utils import save_image

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor
from vggt.rendering.render_image import batch_render_images, get_all_poses, image_to_video, \
    interpolate_pose, adjust_transl
from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding


def load_model(device=None, checkpoint_path=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = VGGT.from_checkpoint(checkpoint_path)

    model.eval()
    model = model.to(device)
    return model, device


def read_dna_npz_entry(npz_path: str, view_ids: list[int]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    images = []
    intrinsics = []
    extrinsics = []

    with np.load(npz_path, allow_pickle=True) as archive:
        for view in view_ids:
            data_key = f"view_{view:02d}"
            if data_key not in archive:
                raise KeyError(f"{data_key} not found in {npz_path}.")

            entry = archive[data_key].item()

            rgb_image = torchvision.io.decode_image(torch.from_numpy(entry["image"]),
                                                    mode=torchvision.io.ImageReadMode.RGB).float() / 255.0

            images.append(rgb_image)

    images = torch.stack(images, dim=0)  # (n_views, 3, H, W)

    return images, intrinsics, extrinsics


def load_sample(seq_dir, target_size=1036):
    """加载单个样本数据"""

    images, intrinsics, extrinsics = (
        read_dna_npz_entry(seq_dir, [int(v) for v in view_ids]))

    origin_size = images.shape[2]
    images = torch.nn.functional.interpolate(
        images, size=(target_size, target_size), mode='bilinear', align_corners=False)

    # 转换为numpy数组
    data = {
        "images": images
    }

    return data


def run_model(target_dir, model, device, input_size=518) -> tuple[dict, dict[str, Tensor]]:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    # Load and preprocess images
    data = load_sample(target_dir, input_size)
    images = data["images"].to(device)

    with torch.no_grad():
        predictions = model(images, if_train=False)

    predictions = {k: v for k, v in predictions.items()}

    return predictions, data


def save_render_images(images: np.array, save_path: str, test_model="ours", test_id="0000_00", frame="00", view=None):
    if view is None:
        view = [0]

    save_dir = os.path.join(save_path, test_id, f"{frame}_pred_{test_model}")
    os.makedirs(save_dir, exist_ok=True)

    # 批量保存（每个图像单独保存）
    for b in range(images.size(0)):
        filename = os.path.join(save_dir, f"{view[b]:02d}.png")
        save_image(images[b], filename, normalize=False)

    # print(f"Save images to {save_dir}")


view_ids = [25, 37, 1, 13]
novel_ids = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]


def main():
    test_path = "/home/china/lab/VGGT_human/experiment/test_data/dna_timeseq/video"
    checkpoint_path = "./checkpoints/stage_3_plus.pt"
    output_dir = "output/"
    input_save_size = 518
    input_size = 1036
    render_size = 2072
    bg_color = torch.tensor([1.0, 1.0, 1.0])
    render_mode = 'gsplat'

    model, device = load_model(device=None, checkpoint_path=checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    for test_id in os.listdir(test_path):
        rendered_images_f = []
        rendered_images_b = []
        input_views_list = []
        input_masks_list = []
        test_model = "ours"
        pose_enc = None
        inter_view = 60  # 每两个相邻视角之间插入N个新视角
        input_path = os.path.join(test_path, test_id)
        if not os.path.isdir(input_path):
            continue

        npz_files = glob.glob(os.path.join(input_path, "*.npz"))

        for i, npz_file in enumerate(sorted(npz_files)):
            if i == 0 or i > 150:
                continue
            preds, data = run_model(npz_file, model, device, input_size=input_size)

            if pose_enc is None:
                pose_enc = preds["pose_enc_pre"].float()
                # 交换pose_enc的第二个维度为 3 0 1 2
                pose_enc = pose_enc[:, [3, 0, 1, 2], :]
                pose_enc = interpolate_pose(pose_enc, inter_view=inter_view).to(device)
            _, lens, _ = pose_enc.shape

            preds["pose_enc"] = pose_enc[:, i % lens, :].unsqueeze(1)
            rendered_image_f, _ = batch_render_images(preds, wo_bg=True, render_mode=render_mode,
                                                     bg_color=bg_color, sr_image_size=render_size)

            # preds["pose_enc"] = pose_enc[:, (i + int(lens / 2)) % lens, :].unsqueeze(1)
            # rendered_image_b, _ = batch_render_images(preds, wo_bg=True, render_mode=render_mode,
            #                                          bg_color=bg_color, sr_image_size=render_size)

            # preds["pose_enc"] = pose_enc[:, i % lens, :].unsqueeze(1)
            # rendered_image_b, _ = batch_render_images(preds, wo_bg=True, render_mode=render_mode,
            #                                          bg_color=bg_color, sr_image_size=render_size)

            # 输出视频原图
            # ds_images = []
            # for image in data["images"]:
            #     ds_image = torch.nn.functional.interpolate(image.unsqueeze(0),
            #                                                size=(input_save_size, input_save_size),
            #                                                mode='bilinear',
            #                                                align_corners=False)
            #     ds_images.append(ds_image)
            # all_images = torch.cat(ds_images, dim=2)

            # all_masks = preds["masks"].squeeze(0).permute(0, 3, 1, 2).reshape(1, 1, -1, 1036)

            # input_masks_list.append(all_masks.squeeze(0).cpu().detach())
            # input_views_list.append(all_images.squeeze(0).cpu().detach())
            rendered_images_f.append(rendered_image_f.squeeze(0).cpu().detach())
            # rendered_images_b.append(rendered_image_b.squeeze(0).cpu().detach())

            frame = os.path.splitext(os.path.basename(npz_file))[0]
            print("Processing: ", test_id, frame)

            # save_render_images(images=all_images.detach().cpu().clone(),
            #                    save_path=output_dir,
            #                    test_model=test_model,
            #                    frame=frame,
            #                    test_id=test_id,
            #                    view=view_ids + novel_ids)

        # video_path = os.path.join(output_dir, f"rendered_video_{test_id}_masks.mp4")
        # image_to_video(torch.stack(input_masks_list), video_path, fps=18)

        # video_path = os.path.join(output_dir, f"rendered_video_{test_id}_input.mp4")
        # image_to_video(torch.stack(input_views_list), video_path, fps=18)

        video_path = os.path.join(output_dir, f"rendered_video_{test_id}_front.mp4")
        image_to_video(torch.stack(rendered_images_f), video_path, fps=18)

        # video_path = os.path.join(output_dir, f"rendered_video_{test_id}_back.mp4")
        # image_to_video(torch.stack(rendered_images_b), video_path, fps=18)


if __name__ == "__main__":
    main()
