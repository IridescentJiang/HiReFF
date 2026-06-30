# from torch_scatter import scatter_add
import torch


def voxelizaton_with_fusion(pts3d, colors, rotations, scales, opacities, voxel_size):
    """
    Voxelization and feature fusion (using average weights).

    Args:
        pts3d: 3D point cloud coordinate tensor [B, V, N, 3]
        colors: Color feature tensor [B, V, N, C_color]
        rotations: Rotation feature tensor [B, V, N, C_rot]
        scales: Scale feature tensor [B, V, N, C_scale]
        opacities: Opacity feature tensor [B, V, N, C_opacity]
        voxel_size: Voxel size (cube edge length)
    """
    B, V, N, _ = pts3d.shape

    # Flatten all input tensors
    pts3d_flat = pts3d.flatten(0, 2)  # [B*V*N, 3]
    colors_flat = colors.flatten(0, 2)  # [B*V*N, C_color]
    rotations_flat = rotations.flatten(0, 2)  # [B*V*N, C_rot]
    scales_flat = scales.flatten(0, 2)  # [B*V*N, C_scale]
    opacities_flat = opacities.flatten(0, 2)  # [B*V*N, C_opacity]

    # Compute voxel indices
    voxel_indices = (pts3d_flat / voxel_size).round().int()  # [B*V*N, 3]

    # Find unique voxels
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_indices, dim=0, return_inverse=True, return_counts=True
    )

    # Compute average weights (each point weighted by 1 / voxel point count)
    weights = (1.0 / counts[inverse_indices].float()).unsqueeze(-1)  # [B*V*N, 1]

    # Compute weighted positions and features
    weighted_pts = pts3d_flat * weights
    weighted_colors = colors_flat * weights
    weighted_rotations = rotations_flat * weights
    weighted_scales = scales_flat * weights
    weighted_opacities = opacities_flat * weights

    # Aggregate by voxel
    voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [num_voxels, 3]
    voxel_colors = scatter_add(weighted_colors, inverse_indices, dim=0)  # [num_voxels, C_color]
    voxel_rotations = scatter_add(weighted_rotations, inverse_indices, dim=0)  # [num_voxels, C_rot]
    voxel_scales = scatter_add(weighted_scales, inverse_indices, dim=0)  # [num_voxels, C_scale]
    voxel_opacities = scatter_add(weighted_opacities, inverse_indices, dim=0)  # [num_voxels, C_opacity]

    # Reshape to [B, M, ...] format
    voxel_pts = voxel_pts.reshape(B, -1, 3)
    voxel_colors = voxel_colors.reshape(B, -1, colors_flat.shape[-1])
    voxel_rotations = voxel_rotations.reshape(B, -1, rotations_flat.shape[-1])
    voxel_scales = voxel_scales.reshape(B, -1, scales_flat.shape[-1])
    voxel_opacities = voxel_opacities.reshape(B, -1, opacities_flat.shape[-1])

    return voxel_pts, voxel_colors, voxel_rotations, voxel_scales, voxel_opacities