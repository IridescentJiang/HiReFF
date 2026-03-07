import torch
import numpy as np
from typing import Tuple, Union


def unproject_depth_map_to_point_map(
        depth_map: torch.Tensor,
        extrinsics_cam: torch.Tensor,
        intrinsics_cam: torch.Tensor
) -> torch.Tensor:
    """
    Unproject a batch of depth maps to 3D world coordinates with gradient support.

    Args:
        depth_map (torch.Tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.Tensor): Batch of camera extrinsic matrices (cam_from_world) of shape (S, 3, 4)
        intrinsics_cam (torch.Tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        torch.Tensor: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    # Remove channel dimension if present (from shape 1,S,H,W,1 or S,H,W,1 to S,H,W)

    if depth_map.ndim == 5 and depth_map.shape[0] == 1:
        depth_map = depth_map.squeeze(0)

    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)

    S, H, W = depth_map.shape
    world_points = []

    # Process each frame in batch
    for i in range(S):
        depth = depth_map[i]
        intrinsic = intrinsics_cam[i]
        extrinsic = extrinsics_cam[i]

        # Get world coordinates for this frame
        world_coords = depth_to_world_coords_points(
            depth, extrinsic, intrinsic
        )
        world_points.append(world_coords.unsqueeze(0))

    # Stack all frames
    return torch.cat(world_points, dim=0)


def depth_to_world_coords_points(
        depth_map: torch.Tensor,
        extrinsic: torch.Tensor,  # cam_from_world extrinsic (3, 4)
        intrinsic: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Convert a depth map to world coordinates with gradient support.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W)
        extrinsic (torch.Tensor): Cam_from_world extrinsic matrix of shape (3, 4)
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
        eps (float): Epsilon value for depth validity check

    Returns:
        torch.Tensor: World coordinates of shape (H, W, 3)
    """
    # Convert depth map to camera coordinates
    cam_coords = depth_to_cam_coords_points(depth_map, intrinsic)

    # Get camera to world transformation by inverting the extrinsic
    # Invert the cam_from_world extrinsic to get world_from_cam
    world_from_cam = invert_extrinsic(extrinsic)

    # Separate rotation and translation components
    R_world_from_cam = world_from_cam[:, :3]  # (3,3)
    t_world_from_cam = world_from_cam[:, 3]  # (3,)

    # Transform camera coordinates to world coordinates
    # P_world = R * P_cam + t
    world_coords = cam_coords @ R_world_from_cam.transpose(0, 1) + t_world_from_cam

    return world_coords


def depth_to_cam_coords_points(
        depth_map: torch.Tensor,
        intrinsic: torch.Tensor
) -> torch.Tensor:
    """
    Convert a depth map to camera coordinates with gradient support.

    Args:
        depth_map (torch.Tensor): Depth map of shape (H, W)
        intrinsic (torch.Tensor): Camera intrinsic matrix of shape (3, 3)

    Returns:
        torch.Tensor: Camera coordinates of shape (H, W, 3)
    """

    H, W = depth_map.shape
    dtype = depth_map.dtype
    device = depth_map.device

    # Generate grid of pixel coordinates using PyTorch
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')

    # Extract intrinsic parameters
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # Calculate camera coordinates with broadcasting
    x_cam = (u - cx) * depth_map / fx
    y_cam = (v - cy) * depth_map / fy
    z_cam = depth_map

    # Stack coordinates
    return torch.stack([x_cam, y_cam, z_cam], dim=-1)


def invert_extrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    """
    Invert a cam_from_world extrinsic matrix to get world_from_cam matrix.

    Args:
        extrinsic (torch.Tensor): Cam_from_world extrinsic matrix of shape (3, 4)

    Returns:
        torch.Tensor: World_from_cam matrix of shape (3, 4)
    """
    # Extract rotation and translation
    R_cam_from_world = extrinsic[:, :3]
    t_cam_from_world = extrinsic[:, 3]

    # Invert rotation: R^T
    R_world_from_cam = R_cam_from_world.T

    # Invert translation: t' = -R^T * t
    t_world_from_cam = -torch.matmul(R_world_from_cam, t_cam_from_world)

    # Construct the inverted matrix
    world_from_cam = torch.cat([
        R_world_from_cam,
        t_world_from_cam.unsqueeze(1)
    ], dim=1)

    return world_from_cam