#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np


def gaussian_render(cam_param, pts_xyz, pts_rgb, rotations, scales, opacity, bg_color, feature=False):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Create zero tensor for screen-space means (used by rasterizer for gradient computation)
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0

    # Set up rasterization configuration
    tanfovx = math.tan(cam_param['FovX'] * 0.5)
    tanfovy = math.tan(cam_param['FovY'] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam_param['height']),
        image_width=int(cam_param['width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=cam_param['world_view_transform'],
        projmatrix=cam_param['full_proj_transform'],
        sh_degree=3,
        campos=cam_param['camera_center'],
        prefiltered=False,
        debug=False,
        subpixel_offset=torch.zeros((int(cam_param['height']), int(cam_param['height']), 2), dtype=torch.float32, device=device),
        kernel_size=0.1
    )

    if feature:
        # rasterizer = GaussianRasterizer16(raster_settings=raster_settings)
        raise NotImplementedError
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=pts_xyz,  # [N, 3]
        means2D=screenspace_points,
        shs=None,
        colors_precomp=pts_rgb,  # [N, 3]
        opacities=opacity,  # 0.5左右
        scales=scales,  # [N, 3] 几乎接近0, 0.002左右
        rotations=rotations,  # [N, 4]
        cov3D_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return rendered_image


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, fovX, fovY, K, h, w):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    if K is None:
        # print("using colmap style")
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right
        sign = 1
    else:
        device = K.device
        dtype = K.dtype
        # print("using guaasian style")
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        near_fx = znear / fx
        near_fy = znear / fy
        left = - (w - cx) * near_fx
        right = cx * near_fx
        bottom = (cy - h) * near_fy
        top = cy * near_fy
        sign = -1

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) * sign
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    # P[0, 0] = 2 * K[0, 0] / w
    # P[1, 1] = 2 * K[1, 1] / h
    # P[0, 2] = 2 * (K[0, 2] / w) - 1
    # P[1, 2] = 2 * (K[1, 2] / h) - 1
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    # P[2, 3] = -2.0 * (zfar * znear) / (zfar - znear)

    return P


def getWorld2View2Tensor(R: torch.Tensor,
                         t: torch.Tensor,
                         translate: torch.Tensor = torch.zeros(3),
                         scale: float = 1.0) -> torch.Tensor:
    """
    可微分版本的 getWorld2View2，使用 PyTorch Tensor 操作

    Args:
        R: 旋转矩阵 [..., 3, 3]
        t: 平移向量 [..., 3]
        translate: 附加平移 [..., 3]
        scale: 缩放系数 [...,] 或标量

    Returns:
        Rt: 组合后的视图矩阵 [..., 4, 4]
    """
    # 确保所有输入在相同设备上
    device = R.device
    dtype = R.dtype
    translate = translate.to(device=device, dtype=dtype)
    scale = torch.as_tensor(scale, device=device, dtype=dtype)

    # 构建基础 Rt 矩阵 [..., 4, 4]
    Rt = torch.zeros(*R.shape[:-2], 4, 4, device=device, dtype=dtype)
    Rt[..., :3, :3] = R
    Rt[..., :3, 3] = t  # 平移向量
    Rt[..., 3, 3] = 1.0

    # 计算相机到世界矩阵 C2W = Rt^-1
    C2W = torch.linalg.inv(Rt)

    # 应用平移和缩放
    cam_center = C2W[..., :3, 3]
    cam_center = (cam_center + translate) * scale.unsqueeze(-1)
    C2W[..., :3, 3] = cam_center

    # 重新计算视图矩阵
    Rt_new = torch.linalg.inv(C2W)
    return Rt_new

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_render_cam(intr, R, T, h, w):
    device = R.device
    cam_param = {}
    cam_param['height'] = h
    cam_param['width'] = w
    cam_param['FovX'] = focal2fov(intr[0, 0], w)
    cam_param['FovY'] = focal2fov(intr[1, 1], h)

    projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=cam_param['FovX'], fovY=cam_param['FovY'],
                                            K=intr, h=h, w=w).transpose(0, 1).contiguous().to(device=device)
    # world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).contiguous()
    world_view_transform = getWorld2View2Tensor(R, T, torch.tensor((0.0, 0.0, 0.0)), 1.0).transpose(0, 1).contiguous()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    cam_param['world_view_transform'] = world_view_transform.float().cuda()
    cam_param['full_proj_transform'] = full_proj_transform.float().cuda()
    cam_param['camera_center'] = camera_center.float().cuda()
    cam_param['prcp_point'] = torch.tensor([intr[0, 2], intr[1, 2]], dtype=torch.float32, device=device)
    cam_param['patch_bbox'] = torch.tensor([0, 0, h, w], dtype=torch.float32, device=device)

    return cam_param

