import os
import sys
import numpy as np
import torch
import glob

import torchvision
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
            intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))
            extrinsics.append(torch.linalg.inv(torch.from_numpy(entry["extrinsic"])).to(dtype=torch.float32))

    images = torch.stack(images, dim=0)  # (n_views, 3, H, W)
    intrinsics = torch.stack(intrinsics, dim=0)  # (n_views, 3, 3)
    extrinsics = torch.stack(extrinsics, dim=0)  # (n_views, 4, 4)

    return images, intrinsics, extrinsics


def load_sample(seq_dir, target_size=1036):
    """加载单个样本数据"""

    images, intrinsics, extrinsics = (
        read_dna_npz_entry(seq_dir, [int(v) for v in view_ids]))

    _, novel_intrinsics, novel_extrinsics = (
        read_dna_npz_entry(seq_dir, [int(v) for v in novel_ids]))

    intrinsics = torch.cat([intrinsics, novel_intrinsics], dim=0)
    all_extrinsics = torch.cat([extrinsics, novel_extrinsics], dim=0)

    origin_size = images.shape[2]
    images = torch.nn.functional.interpolate(
        images, size=(target_size, target_size), mode='bilinear', align_corners=False)

    adjusted_intrinsic = adjust_intrinsic_batch(intrinsics,
                                                origin_size, origin_size,
                                                target_size, target_size)
    relatived_extrinsics = convert_extrinsics_to_relative_tensor(extrinsics)
    relatived_all_extrinsics = convert_extrinsics_to_relative_tensor(all_extrinsics)
    relatived_sp_extrinsics = relatived_all_extrinsics[-len(novel_extrinsics):]

    # 转换为numpy数组
    data = {
        "images": images,
        "intrinsics": adjusted_intrinsic,
        "extrinsics": relatived_extrinsics,
        "sp_extrinsics": relatived_sp_extrinsics
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

    print(f"Save images to {save_dir}")


view_ids = [25, 1, 13, 37]
novel_ids = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]


def main():
    test_path = "/home/china/lab/VGGT_human/experiment/test_data/dna_timeseq"
    checkpoint_path = "./checkpoints/stage_3_plus.pt"
    output_dir = "output/dna_timeseq"
    test_model = "ours"
    input_size = 1036
    render_size = 2072
    bg_color = torch.tensor([1.0, 1.0, 1.0])
    render_mode = 'gsplat'

    model, device = load_model(device=None, checkpoint_path=checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    for i, test_id in enumerate(sorted(os.listdir(test_path), key=lambda x: int(x))):
        input_path = os.path.join(test_path, test_id)
        if not os.path.isdir(input_path):
            continue

        npz_files = glob.glob(os.path.join(input_path, "*.npz"))

        for idx, npz_file in enumerate(sorted(npz_files)):
            if idx % 5 != 0:  # 跳过非间隔5的文件
                continue

            preds, data = run_model(npz_file, model, device, input_size=input_size)

            adj_extrinsics, adj_sp_extrinsics = adjust_transl(preds["pose_enc_pre"],
                                                              data["extrinsics"].unsqueeze(0),
                                                              data["sp_extrinsics"].unsqueeze(0))
            extrinsics = torch.cat([adj_extrinsics, adj_sp_extrinsics], dim=1)

            pose_enc = extri_intri_to_pose_encoding(extrinsics=extrinsics,
                                                    intrinsics=data["intrinsics"].unsqueeze(0),
                                                    image_size_hw=(data["images"].shape[2],
                                                                   data["images"].shape[3]))
            preds["pose_enc"] = pose_enc.float().to(device)
            rendered_images, _ = batch_render_images(preds, wo_bg=True, render_mode=render_mode,
                                                     bg_color=bg_color, sr_image_size=render_size)

            frame = os.path.splitext(os.path.basename(npz_file))[0]

            save_render_images(images=rendered_images.detach().cpu().clone(),
                               save_path=output_dir,
                               test_model=test_model,
                               frame=frame,
                               test_id=test_id,
                               view=view_ids + novel_ids)


if __name__ == "__main__":
    main()
