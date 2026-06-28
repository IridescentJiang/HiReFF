import os
import numpy as np
import torch
import cv2
import torchvision
from tqdm import tqdm
from kornia.geometry.transform import remap

from SMCReader import SMCReader



def process_images(image, mask, map_x, map_y, target_size=518):
    image_dist = remap(image, map_x, map_y, mode='bilinear', padding_mode='zeros')
    mask_dist = remap(mask, map_x, map_y, mode='bilinear', padding_mode='zeros')
    
    height, width = image_dist.shape[2], image_dist.shape[3]
    
    # first crop to square
    if height > width:
        start_y = (height - width) // 2
        image_dist = image_dist[:, :, start_y: start_y + width, :]
        mask_dist = mask_dist[:, :, start_y: start_y + width, :]
    elif width > height:
        start_x = (width - height) // 2
        image_dist = image_dist[:, :, :, start_x: start_x + height]
        mask_dist = mask_dist[:, :, :, start_x: start_x + height]
    
    # then resize to target size
    image_dist = torch.nn.functional.interpolate(image_dist, size=(target_size, target_size), mode='bilinear', align_corners=False)
    mask_dist = torch.nn.functional.interpolate(mask_dist, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    return image_dist, mask_dist


def get_new_intrinsic(ori_intr, dist_coeff, ori_size, target_size):
    # first do undistortion
    new_K_0, _ = cv2.getOptimalNewCameraMatrix(
        ori_intr, dist_coeff, ori_size, alpha=0.0, newImgSize=ori_size, centerPrincipalPoint=True
    )
    
    new_K = new_K_0.copy()
    
    # crop to square
    height, width = ori_size[1], ori_size[0]
    if height > width:
        start_y = (height - width) // 2
        new_K[1, 2] -= start_y
    elif width > height:
        start_x = (width - height) // 2
        new_K[0, 2] -= start_x
    
    # resize to target size
    scale = target_size / min(height, width)
    new_K[0, 0] *= scale
    new_K[1, 1] *= scale
    new_K[0, 2] *= scale
    new_K[1, 2] *= scale
    
    return new_K, new_K_0


def get_undist_map(ori_intr, dist_coeff, ori_size, target_size, device):
    dist_coeff = np.zeros_like(dist_coeff)
    new_K, new_K_0 = get_new_intrinsic(ori_intr, dist_coeff, ori_size, target_size)
    map_x, map_y = cv2.initUndistortRectifyMap(
        ori_intr, dist_coeff, R=np.eye(3), newCameraMatrix=new_K_0,
        size=ori_size, m1type=cv2.CV_32FC1
    )
    
    map_x_tensor = torch.from_numpy(map_x).to(device)
    map_y_tensor = torch.from_numpy(map_y).to(device)
    
    return map_x_tensor, map_y_tensor, new_K


def get_all_frames_data(rd, rd_annots, save_path=None):
    print("===========================================================", flush=True)
    print(f"=== get data from all frames, it takes several minutes!!!", flush=True)
    print("===========================================================", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # first process camera intrinsics and undistort maps for all views
    undistort_maps_x = []
    undistort_maps_y = []
    extrinsics = []
    new_intrinsics = []
    ori_size = (2048, 2448)
    
    rgb_cam_dict = rd_annots.get_Calibration_all()
    for rgb_vidx in tqdm(range(48), desc="Processing camera intrinsics"):
        calib = rgb_cam_dict[f"{rgb_vidx:02d}"]
        _kk = calib["K"]
        RT = calib["RT"]
        w2c = np.linalg.inv(RT)
        extrinsics.append(w2c)
        
        map_x_tensor, map_y_tensor, new_K = get_undist_map(_kk, calib["D"], ori_size, target_size=518 * 4, device=device)
        undistort_maps_x.append(map_x_tensor)
        undistort_maps_y.append(map_y_tensor)
        new_intrinsics.append(new_K)
    
    undistort_maps_x = torch.stack(undistort_maps_x, dim=0)  # (n_views, H, W)
    undistort_maps_y = torch.stack(undistort_maps_y, dim=0)  # (n_views, H, W)
    extrinsics = np.stack(extrinsics, axis=0)  # (n_views, 4, 4)
    new_intrinsics = np.stack(new_intrinsics, axis=0)  # (n_views, 3, 3)
    
    n_frame = rd.get_Camera_5mp_info()['num_frame']
    _reso = rd.get_Camera_5mp_info()['resolution']
    assert _reso[0] == ori_size[1] and _reso[1] == ori_size[0], "Original resolution does not match!"
    
    for fid in tqdm(range(n_frame), desc="Processing frames"):
        codec_result = {}
        frame_images = []
        frame_masks = []
        
        for rgb_vidx in range(48):
            camera_str = "Camera_5mp"
            view_img = rd.get_img(camera_str, rgb_vidx, 'color', Frame_id=fid, return_bytes=True)
            view_img = torchvision.io.decode_jpeg(torch.from_numpy(view_img), mode=torchvision.io.ImageReadMode.RGB, device=device).float() / 255.0 # (3, H, W)
            
            view_mask = rd_annots.get_mask(rgb_vidx, Frame_id=fid, return_bytes=True)
            # view_mask = torch.from_numpy(view_mask).to(device).float().unsqueeze(0) / 255.0  # (1, H, W)
            view_mask = torchvision.io.decode_image(torch.from_numpy(view_mask), mode=torchvision.io.ImageReadMode.GRAY).to(device).float() / 255.0 # (1, H, W)

            frame_images.append(view_img)
            frame_masks.append(view_mask)

        frame_images = torch.stack(frame_images, dim=0)  # (n_views, 3, H, W)
        frame_masks = torch.stack(frame_masks, dim=0)  # (n_views, 1, H, W)
        
        processed_images, processed_masks = process_images(
            frame_images, frame_masks,
            undistort_maps_x, undistort_maps_y,
            target_size=518 * 4
        )
        
        processed_images = (processed_images * 255.0).to(torch.uint8)
        processed_masks = (processed_masks > 0.5).to(torch.uint8) * 255
        
        for rgb_vidx in range(48):
            key = f"view_{rgb_vidx:02d}"
            codec_result[key] = {
                "image": torchvision.io.encode_jpeg(processed_images[rgb_vidx], quality=90).cpu().numpy(),
                "mask": torchvision.io.encode_png(processed_masks[rgb_vidx].cpu()).numpy(),
                "intrinsic": new_intrinsics[rgb_vidx],
                "extrinsic": extrinsics[rgb_vidx],
            }
            
        os.makedirs(save_path, exist_ok=True)

        np.savez(save_path + f"/frame_{fid:04d}.npz", **codec_result)

ids = ["0008_01", "0012_09", "0018_05", "0019_06", "0022_10"]
for id in ids:
    rd = SMCReader(f"./data_smc/{id}.smc")
    rd_annots = SMCReader(f"./data_smc/{id}_annots.smc")
    get_all_frames_data(rd, rd_annots, save_path=f"./data_npz/{id}/")
