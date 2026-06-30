import os

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image
import imageio
from gsplat import rasterization

from hireff.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding


def image_to_video(images, video_path, fps=24):
    """
    Save a batch of images as a video.
    Args:
        images: Tensor of shape [N, C, H, W], where N is the number of frames.
        video_path: Path to save the video.
        fps: Frames per second for the video.
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Key optimization: frame-by-frame processing (avoids loading all images at once)
    macro_block_size = 16
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        format='mp4',
        codec='libx264',  # Use efficient encoder
        bitrate='25M',  # High bitrate
        quality=10,  # 0-10 highest quality (imageio-specific parameter)
        macro_block_size=None  # Disable internal secondary scaling
    )

    # Initialize sharpening parameters (adjustable)
    SHARPEN_STRENGTH = 0.5  # Sharpening strength (recommended range 1.0-2.0)

    # Frame-by-frame processing, only single frame data kept in memory
    for i in range(images.shape[0]):
        # 1. Extract single frame and convert dimensions [C, H, W] -> [H, W, C]
        frame_tensor = images[i].permute(1, 2, 0)  # Process single frame only

        # 2. Compute alignment size (avoid full-image interpolation)
        height, width = frame_tensor.shape[:2]
        new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
        new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size

        # 3. Single-frame interpolation (memory footprint reduced to 1/N)
        frame_resized = torch.nn.functional.interpolate(
            frame_tensor.permute(2, 0, 1).unsqueeze(0),  # Restore to [1, C, H, W]
            size=(new_height, new_width),
            mode="bilinear",
            antialias=True,
            align_corners=False
        ).squeeze(0)  # Keep [C, H, W] format for subsequent convolution processing

        # Method 1: Use Laplacian sharpening (lightweight)
        if SHARPEN_STRENGTH > 0:
            # Create sharpening convolution kernel
            identity = torch.tensor([[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]], dtype=frame_resized.dtype, device=frame_resized.device)
            laplacian = torch.tensor([[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]], dtype=frame_resized.dtype, device=frame_resized.device)

            # Combine sharpening kernel (original image + Laplacian edge enhancement)
            kernel = identity + SHARPEN_STRENGTH * laplacian
            kernel = kernel.expand(frame_resized.size(0), 1, 3, 3)  # Match channel count

            # Apply sharpening convolution (edge reflection padding to avoid black borders)
            sharpened = torch.nn.functional.conv2d(
                frame_resized.unsqueeze(0),
                kernel,
                padding=1,
                groups=frame_resized.size(0)  # Per-channel grouped processing
            ).squeeze(0)

            # Clamp to valid range
            sharpened = torch.clamp(sharpened, 0, 1)
            frame_resized = sharpened

        # Convert to output format [H, W, C]
        frame_output = frame_resized.permute(1, 2, 0)

        # 4. Convert to NumPy and scale to [0, 255]
        frame_np = (frame_output.detach().cpu().numpy() * 255).astype('uint8')
        writer.append_data(frame_np)

        # 5. Explicitly release intermediate variables
        del frame_tensor, frame_resized, frame_output, frame_np
        if i % 10 == 0:  # Periodic cache cleanup
            torch.cuda.empty_cache() if images.is_cuda else None


def encode_poses(intrinsics, extrinsics, H, W, device):
    """
    Convert input camera parameters to HiReFF pose encoding format.
    """

    B, V, _, _ = extrinsics.shape

    new_poses = []
    for b_id in range(B):
        for v_id in range(V):
            true_intrinsic = intrinsics[b_id][v_id].unsqueeze(0).unsqueeze(0)
            true_extrinsic = extrinsics[b_id][v_id].unsqueeze(0).unsqueeze(0)

            # Convert intrinsics/extrinsics to HiReFF pose encoding format
            encoded_pose = extri_intri_to_pose_encoding(true_extrinsic, true_intrinsic, (H, W)).squeeze(0)

            new_poses.append(encoded_pose)

    new_poses_torch = torch.stack(new_poses, dim=1).to(device)

    return new_poses_torch


def adjust_transl(target_pose_enc, extrinsics, supervise_extrinsics):
    """
    Adjust translation in the input extrinsics to match the scale of the pose encoding.
    Uses the average scale factor across all views in each batch (excluding the first view).
    """
    B, V, _, _ = extrinsics.shape

    # Ensure there are multiple views for computing average scale
    if V <= 1:
        return extrinsics, supervise_extrinsics

    new_extrinsics = extrinsics.clone()
    new_supervise_extrinsics = supervise_extrinsics.clone() if supervise_extrinsics is not None else None

    for b_id in range(B):
        # Compute average scale factor across all views (excluding the first view)
        scale_norm = []

        for v_id in range(1, V):  # Start from 1, skip first view
            transl = extrinsics[b_id, v_id, :3, 3]
            pose_transl = target_pose_enc[b_id, v_id, :3]

            transl_norm = torch.norm(transl)
            pose_norm = torch.norm(pose_transl)

            if transl_norm > 1e-6:  # Avoid division by zero
                scale = pose_norm / transl_norm
                scale_norm.append(scale)

        # Compute average scale factor
        scale = torch.mean(torch.stack(scale_norm).float()) if scale_norm else 1.0
        scale = scale.to(device=transl.device)

        # Apply uniformly to all views
        for v_id in range(V):  # Not excluding the first view
            # Update extrinsic matrix
            transl = extrinsics[b_id, v_id, :3, 3]
            new_extrinsics[b_id, v_id, :3, 3] = transl * scale

        if supervise_extrinsics is not None:
            # Update supervision extrinsic matrices
            _, s_V, _, _ = supervise_extrinsics.shape
            for v_id in range(s_V):
                supervise_transl = supervise_extrinsics[b_id, v_id, :3, 3]
                if new_supervise_extrinsics is not None:
                    new_supervise_extrinsics[b_id, v_id, :3, 3] = supervise_transl * scale

    return new_extrinsics, new_supervise_extrinsics


def interpolate_pose(pose_tensor, inter_view):
    """
    Perform precise center-based interpolation between adjacent views, handling different distance issues.

    Args:
        preds: Dictionary containing pose encoding with key "pose_enc" of shape [b, V, 9]
        inter_view: Number of new views to insert between each pair of adjacent views

    Returns:
        preds: Updated dictionary with "pose_enc" expanded to shape [b, V*(inter_view+1), 9]
    """
    # Get original pose tensor
    b_size = pose_tensor.shape[0]

    # Handle edge cases
    if inter_view < 1:
        return pose_tensor

    # Store interpolation results for each batch
    interpolated_batches = []

    for b_idx in range(b_size):
        batch_poses = pose_tensor[b_idx]  # [V, 9]
        num_views = batch_poses.shape[0]

        # If only 1 view, no interpolation needed
        if num_views <= 1:
            interpolated_batches.append(batch_poses)
            continue

        # Convert to NumPy array for interpolation computation
        poses_np = batch_poses.detach().cpu().numpy()
        interpolated_poses = []

        # 1. Compute average center and height
        positions = poses_np[:, :3]
        centers = np.mean(positions, axis=0)
        avg_height = np.mean(positions[:, 1])  # Y coordinate (height)

        # Convert positions to relative coordinates with center as origin
        centered_positions = positions - centers

        # 2. Convert positions to cylindrical coordinates (rho, phi, y)
        rho = np.linalg.norm(centered_positions[:, [0, 2]], axis=1)  # XZ plane distance
        phi = np.arctan2(centered_positions[:, 2], centered_positions[:, 0])  # Angle
        y = centered_positions[:, 1]  # Height

        # 3. Iterate over all view pairs for angular interpolation
        for i in range(num_views):
            # Get current view pair (forming a closed loop)
            idx_i = i
            idx_j = (i + 1) % num_views

            # Original poses
            pose_i = poses_np[idx_i]
            pose_j = poses_np[idx_j]

            # Quaternion representation of camera orientation
            q_i = pose_i[3:7]
            q_j = pose_j[3:7]

            # Cylindrical coordinates of current views
            rho_i = rho[idx_i]
            phi_i = phi[idx_i]
            y_i = y[idx_i]

            rho_j = rho[idx_j]
            phi_j = phi[idx_j]
            y_j = y[idx_j]

            # Handle angle difference (account for circular wrap-around)
            angle_diff = phi_j - phi_i
            # Ensure shortest path is chosen
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Generate interpolation parameters
            t_vals = np.linspace(0, 1, inter_view + 2)[1:-1]  # Intermediate points

            # Angular interpolation (linear in angle space)
            inter_phi = phi_i + t_vals * angle_diff

            # Radius interpolation (linear)
            inter_rho = (1 - t_vals) * rho_i + t_vals * rho_j

            # Height interpolation (linear)
            inter_y = (1 - t_vals) * y_i + t_vals * y_j

            # Convert cylindrical coordinates back to Cartesian
            inter_x = inter_rho * np.cos(inter_phi)
            inter_z = inter_rho * np.sin(inter_phi)

            # Convert to absolute positions (add center back)
            inter_positions = np.column_stack([
                inter_x + centers[0],
                inter_y + centers[1],
                inter_z + centers[2]
            ])

            # Rotation quaternion spherical linear interpolation
            rotations = R.from_quat([q_i, q_j])
            slerp = Slerp([0, 1], rotations)
            inter_rots = slerp(t_vals)
            inter_quat = inter_rots.as_quat()

            # Field of view keeps consistent with the current view
            fov_i = pose_i[7:9]
            inter_fov = np.tile(fov_i, (inter_view, 1))

            # Add original view
            interpolated_poses.append(pose_i)

            # Add interpolated points
            for k in range(len(t_vals)):
                new_pose = np.concatenate([
                    inter_positions[k],
                    inter_quat[k],
                    inter_fov[k]
                ])
                interpolated_poses.append(new_pose)

        # Convert to PyTorch tensor and add back to batch
        inter_tensor = torch.tensor(
            np.array(interpolated_poses),
            dtype=pose_tensor.dtype,
            device=pose_tensor.device
        )
        interpolated_batches.append(inter_tensor)

    return torch.stack(interpolated_batches, dim=0)


def batch_render_images_my(pred, wo_bg=True, sr_image_size=None, render_depth=False, bg_color=None):

    if sr_image_size:
        H, W = (sr_image_size, sr_image_size)
    else:
        _, _, _, H, W = pred["images"].shape

    B, V = pred["pose_enc"].shape[:2]  # Get batch and view dimensions
    rendered_depth = None
    
    if bg_color is None:
        bg_color = torch.ones(B, V, 3, dtype=torch.float32, device=pred["masks"].device)

    rendered_images, rendered_depth = vectorized_gaussian_render_gsplat_my(
        pose_enc=pred["pose_enc"],
        pcs=pred["flat_gs"], 
        image_size=(H, W),
        view_size=V,
        render_depth=render_depth,
        bg_color = bg_color
    )

    return rendered_images, rendered_depth


def vectorized_gaussian_render_gsplat_my(pose_enc, pcs, image_size, view_size, render_depth, bg_color):
    """
    Memory-optimized vectorized Gaussian rendering.
    """
    H, W = image_size
    device = pose_enc.device
    batch_size = len(pcs)

    # Precompute camera parameters
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))

    # bg = bg_color.reshape(1, 1, 3).expand(batch_size, view_size, 3).to(device)  # shape [B, V, 3]
    bg = bg_color.flatten(0, 1)[None]

    last_row = torch.tensor([0, 0, 0, 1], device=device)
    last_row = last_row.reshape(1, 1, 4)  # shape [1, 1, 4]

    render_images_list = []
    render_depth_list = []

    for b_idx in range(batch_size):
        current_extrinsic = extrinsics[b_idx]  # shape [B * V, 3, 4]
        current_extrinsic_extend = torch.cat([
            current_extrinsic,
            last_row.expand(batch_size * view_size, -1, -1)  # Expand to [B * V, 1, 4]
        ], dim=1)  # shape [B * V, 4, 4]
        pc = pcs[b_idx]

        rendering, *_ = rasterization(
            pc["xyz"][None], 
            pc["rotations"][None], 
            pc["scales"][None], 
            pc["opacities"][None].squeeze(-1),  # [1, 1, N] -> [1, N]
            pc["colors"][None].squeeze(-1),
            current_extrinsic_extend.unsqueeze(0),  # [1, B*V, 4, 4]
            intrinsics[b_idx:b_idx + 1],
            W, H,
            sh_degree=None,
            render_mode="RGB+D" if render_depth else "RGB",
            packed=False,
            near_plane=1e-10,
            backgrounds=bg,
            radius_clip=0.1,
            rasterize_mode='classic'
        )

        if render_depth:
            render_image, render_depth = torch.split(rendering, [3, 1], dim=-1)
            render_depth_list.append(render_depth)
        else:
            render_image = rendering

        render_image = render_image.clamp(0.0, 1.0)
        render_images_list.append(render_image)

    if render_depth:
        # Merge results and adjust dimensions
        render_images = torch.cat(render_images_list, dim=0)  # [batch_size, 8, H, W, 3]
        render_depths = torch.cat(render_depth_list, dim=0)  # [batch_size, 8, H, W, 3]
        return rearrange(render_images, "b v h w c-> (b v) c h w").contiguous(), rearrange(render_depths,
                                                                                           "b v h w c-> (b v) c h w").contiguous()  # [batch_size, 8, 3, H, W]
    else:
        # Merge results and adjust dimensions
        render_images = torch.cat(render_images_list, dim=0)  # [batch_size, 8, H, W, 3]
        return rearrange(render_images, "b v h w c-> (b v) c h w").contiguous(), None


def save_rendered_images(images: np.array, save_path: str, epoch=0, start_id=None):
    """
    Batch save rendered images to a specified path.
    Args:
        images: Image tensor to save [N, C, H, W]
        save_path: Save directory path
        epoch: Epoch number
    """
    # Create save directory
    save_dir = os.path.join(save_path, str(f"{epoch:02d}"))
    os.makedirs(save_dir, exist_ok=True)

    # Batch save (each image saved separately)
    for b in range(images.size(0)):
        if start_id:
            filename = os.path.join(save_dir, f"render_ep_{epoch:02d}_{start_id * images.size(0) + b:02d}.png")
        else:
            filename = os.path.join(save_dir, f"render_ep_{epoch:02d}_{b:02d}.png")
        save_image(images[b], filename, normalize=False)
