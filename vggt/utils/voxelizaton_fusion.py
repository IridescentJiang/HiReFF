# from torch_scatter import scatter_add
import torch


def voxelizaton_with_fusion(pts3d, colors, rotations, scales, opacities, voxel_size):
    """
    体素化与特征融合（使用平均权重）

    参数:
        pts3d: 3D点云坐标张量 [B, V, N, 3]
        colors: 颜色特征张量 [B, V, N, C_color]
        rotations: 旋转特征张量 [B, V, N, C_rot]
        scales: 缩放特征张量 [B, V, N, C_scale]
        opacities: 不透明度特征张量 [B, V, N, C_opacity]
        voxel_size: 体素大小（立方体边长）
    """
    B, V, N, _ = pts3d.shape

    # 展平所有输入张量
    pts3d_flat = pts3d.flatten(0, 2)  # [B*V*N, 3]
    colors_flat = colors.flatten(0, 2)  # [B*V*N, C_color]
    rotations_flat = rotations.flatten(0, 2)  # [B*V*N, C_rot]
    scales_flat = scales.flatten(0, 2)  # [B*V*N, C_scale]
    opacities_flat = opacities.flatten(0, 2)  # [B*V*N, C_opacity]

    # 计算体素索引
    voxel_indices = (pts3d_flat / voxel_size).round().int()  # [B*V*N, 3]

    # 找出唯一体素
    unique_voxels, inverse_indices, counts = torch.unique(
        voxel_indices, dim=0, return_inverse=True, return_counts=True
    )

    # 计算平均权重 (每个点权重为 1/该体素点数)
    weights = (1.0 / counts[inverse_indices].float()).unsqueeze(-1)  # [B*V*N, 1]

    # 计算加权位置和特征
    weighted_pts = pts3d_flat * weights
    weighted_colors = colors_flat * weights
    weighted_rotations = rotations_flat * weights
    weighted_scales = scales_flat * weights
    weighted_opacities = opacities_flat * weights

    # 按体素聚合
    voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [体素数, 3]
    voxel_colors = scatter_add(weighted_colors, inverse_indices, dim=0)  # [体素数, C_color]
    voxel_rotations = scatter_add(weighted_rotations, inverse_indices, dim=0)  # [体素数, C_rot]
    voxel_scales = scatter_add(weighted_scales, inverse_indices, dim=0)  # [体素数, C_scale]
    voxel_opacities = scatter_add(weighted_opacities, inverse_indices, dim=0)  # [体素数, C_opacity]

    # 重塑为 [B, M, ...] 格式
    voxel_pts = voxel_pts.reshape(B, -1, 3)
    voxel_colors = voxel_colors.reshape(B, -1, colors_flat.shape[-1])
    voxel_rotations = voxel_rotations.reshape(B, -1, rotations_flat.shape[-1])
    voxel_scales = voxel_scales.reshape(B, -1, scales_flat.shape[-1])
    voxel_opacities = voxel_opacities.reshape(B, -1, opacities_flat.shape[-1])

    return voxel_pts, voxel_colors, voxel_rotations, voxel_scales, voxel_opacities