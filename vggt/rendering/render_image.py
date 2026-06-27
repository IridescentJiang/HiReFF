import os

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from torchvision.utils import save_image
import imageio
from gsplat import rasterization

from vggt.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding


def image_to_video(images, video_path, fps=24):
    """
    Save a batch of images as a video.
    Args:
        images: Tensor of shape [N, C, H, W], where N is the number of frames.
        video_path: Path to save the video.
        fps: Frames per second for the video.
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # 关键优化：逐帧处理（避免一次性加载所有图像）
    macro_block_size = 16
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        format='mp4',
        codec='libx264',  # 使用高效编码器
        bitrate='25M',  # 提升比特率
        quality=10,  # 0-10最高质量（imageio特有参数）
        macro_block_size=None  # 禁用内部二次缩放
    )

    # 初始化锐化参数（可调整）
    SHARPEN_STRENGTH = 0.5  # 锐化强度 (推荐范围 1.0-2.0)

    # 逐帧处理，内存中仅保留单帧数据
    for i in range(images.shape[0]):
        # 1. 提取单帧并转换维度 [C, H, W] -> [H, W, C]
        frame_tensor = images[i].permute(1, 2, 0)  # 仅处理单帧

        # 2. 计算对齐尺寸（避免全量插值）
        height, width = frame_tensor.shape[:2]
        new_height = (height + macro_block_size - 1) // macro_block_size * macro_block_size
        new_width = (width + macro_block_size - 1) // macro_block_size * macro_block_size

        # 3. 单帧插值（内存占用降至1/N）
        frame_resized = torch.nn.functional.interpolate(
            frame_tensor.permute(2, 0, 1).unsqueeze(0),  # 恢复为 [1, C, H, W]
            size=(new_height, new_width),
            mode="bilinear",
            antialias=True,
            align_corners=False
        ).squeeze(0)  # 保持 [C, H, W] 格式方便后续卷积处理

        # 方案1：使用拉普拉斯锐化（轻量级）
        if SHARPEN_STRENGTH > 0:
            # 创建锐化卷积核
            identity = torch.tensor([[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]], dtype=frame_resized.dtype, device=frame_resized.device)
            laplacian = torch.tensor([[0, -1, 0],
                                      [-1, 4, -1],
                                      [0, -1, 0]], dtype=frame_resized.dtype, device=frame_resized.device)

            # 组合锐化核（原始图像 + 拉普拉斯边缘增强）
            kernel = identity + SHARPEN_STRENGTH * laplacian
            kernel = kernel.expand(frame_resized.size(0), 1, 3, 3)  # 适配通道数

            # 应用锐化卷积（边缘用反射填充避免黑边）
            sharpened = torch.nn.functional.conv2d(
                frame_resized.unsqueeze(0),
                kernel,
                padding=1,
                groups=frame_resized.size(0)  # 通道分组处理
            ).squeeze(0)

            # 约束到合理范围
            sharpened = torch.clamp(sharpened, 0, 1)
            frame_resized = sharpened

        # 转换到输出格式 [H, W, C]
        frame_output = frame_resized.permute(1, 2, 0)

        # 4. 转NumPy并缩放至[0, 255]
        frame_np = (frame_output.detach().cpu().numpy() * 255).astype('uint8')
        writer.append_data(frame_np)

        # 5. 显式释放中间变量
        del frame_tensor, frame_resized, frame_output, frame_np
        if i % 10 == 0:  # 定期清理缓存
            torch.cuda.empty_cache() if images.is_cuda else None


def encode_poses(intrinsics, extrinsics, H, W, device):
    """
    将输入的相机参数转换为VGGT的pose encoding格式
    """

    B, V, _, _ = extrinsics.shape

    new_poses = []
    for b_id in range(B):
        for v_id in range(V):
            true_intrinsic = intrinsics[b_id][v_id].unsqueeze(0).unsqueeze(0)
            true_extrinsic = extrinsics[b_id][v_id].unsqueeze(0).unsqueeze(0)

            # 将内外参转换为VGGT的pose encoding格式
            encoded_pose = extri_intri_to_pose_encoding(true_extrinsic, true_intrinsic, (H, W)).squeeze(0)

            new_poses.append(encoded_pose)

    new_poses_torch = torch.stack(new_poses, dim=1).to(device)

    return new_poses_torch


def adjust_transl(target_pose_enc, extrinsics, supervise_extrinsics):
    """
    调整输入的外参中的平移部分，使其与位姿编码的尺度一致
    使用每个批次中所有视角的平均缩放因子（排除第一个视角）
    """
    B, V, _, _ = extrinsics.shape

    # 确保有多个视角用于计算平均缩放
    if V <= 1:
        return extrinsics, supervise_extrinsics

    new_extrinsics = extrinsics.clone()
    new_supervise_extrinsics = supervise_extrinsics.clone() if supervise_extrinsics is not None else None

    for b_id in range(B):
        # 计算所有视角的平均缩放因子（排除第一个视角）
        scale_norm = []

        for v_id in range(1, V):  # 从1开始，跳过第一个视角
            transl = extrinsics[b_id, v_id, :3, 3]
            pose_transl = target_pose_enc[b_id, v_id, :3]

            transl_norm = torch.norm(transl)
            pose_norm = torch.norm(pose_transl)

            if transl_norm > 1e-6:  # 避免除以零
                scale = pose_norm / transl_norm
                scale_norm.append(scale)

        # 计算平均缩放因子
        scale = torch.mean(torch.stack(scale_norm).float()) if scale_norm else 1.0
        scale = scale.to(device=transl.device)

        # 统一应用到所有视角
        for v_id in range(V):  # 不包括第一个视角
            # 更新外参矩阵
            transl = extrinsics[b_id, v_id, :3, 3]
            new_extrinsics[b_id, v_id, :3, 3] = transl * scale

        if supervise_extrinsics is not None:
            # 更新监督外参矩阵
            _, s_V, _, _ = supervise_extrinsics.shape
            for v_id in range(s_V):
                supervise_transl = supervise_extrinsics[b_id, v_id, :3, 3]
                if new_supervise_extrinsics is not None:
                    new_supervise_extrinsics[b_id, v_id, :3, 3] = supervise_transl * scale

    return new_extrinsics, new_supervise_extrinsics


def interpolate_pose(pose_tensor, inter_view):
    """
    在相邻视角之间进行围绕中心的精确插值，处理不同距离问题

    参数:
        preds: 包含位姿编码的字典，键"pose_enc"的值为形状[b, V, 9]的torch张量
        inter_view: 每两个相邻视角之间要插入的新视角数量

    返回:
        preds: 更新后的字典，其中"pose_enc"变为形状[b, V*(inter_view+1), 9]
    """
    # 获取原始位姿张量
    b_size = pose_tensor.shape[0]

    # 处理边缘情况
    if inter_view < 1:
        return pose_tensor

    # 存储每个批次的插值结果
    interpolated_batches = []

    for b_idx in range(b_size):
        batch_poses = pose_tensor[b_idx]  # [V, 9]
        num_views = batch_poses.shape[0]

        # 如果只有1个视角，不需要插值
        if num_views <= 1:
            interpolated_batches.append(batch_poses)
            continue

        # 转换为NumPy数组用于插值计算
        poses_np = batch_poses.detach().cpu().numpy()
        interpolated_poses = []

        # 1. 计算平均中心和高度
        positions = poses_np[:, :3]
        centers = np.mean(positions, axis=0)
        avg_height = np.mean(positions[:, 1])  # Y坐标（高度）

        # 将位置转换为以中心为原点的相对坐标
        centered_positions = positions - centers

        # 2. 将位置转换为柱坐标 (rho, phi, y)
        rho = np.linalg.norm(centered_positions[:, [0, 2]], axis=1)  # XZ平面距离
        phi = np.arctan2(centered_positions[:, 2], centered_positions[:, 0])  # 角度
        y = centered_positions[:, 1]  # 高度

        # 3. 遍历所有视角对进行角度插值
        for i in range(num_views):
            # 获取当前视角对（形成闭环）
            idx_i = i
            idx_j = (i + 1) % num_views

            # 原始位姿
            pose_i = poses_np[idx_i]
            pose_j = poses_np[idx_j]

            # 四元数表示相机朝向
            q_i = pose_i[3:7]
            q_j = pose_j[3:7]

            # 当前视角的柱坐标
            rho_i = rho[idx_i]
            phi_i = phi[idx_i]
            y_i = y[idx_i]

            rho_j = rho[idx_j]
            phi_j = phi[idx_j]
            y_j = y[idx_j]

            # 处理角度差（考虑圆周环绕）
            angle_diff = phi_j - phi_i
            # 确保选择最短路径
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # 生成插值参数
            t_vals = np.linspace(0, 1, inter_view + 2)[1:-1]  # 中间点

            # 角度插值（线性在角度空间中）
            inter_phi = phi_i + t_vals * angle_diff

            # 半径插值（线性）
            inter_rho = (1 - t_vals) * rho_i + t_vals * rho_j

            # 高度插值（线性）
            inter_y = (1 - t_vals) * y_i + t_vals * y_j

            # 将柱坐标转换回笛卡尔坐标
            inter_x = inter_rho * np.cos(inter_phi)
            inter_z = inter_rho * np.sin(inter_phi)

            # 转换为绝对位置（加回中心点）
            inter_positions = np.column_stack([
                inter_x + centers[0],
                inter_y + centers[1],
                inter_z + centers[2]
            ])

            # 旋转四元数球面线性插值
            rotations = R.from_quat([q_i, q_j])
            slerp = Slerp([0, 1], rotations)
            inter_rots = slerp(t_vals)
            inter_quat = inter_rots.as_quat()

            # 视场角保持与当前视角一致
            fov_i = pose_i[7:9]
            inter_fov = np.tile(fov_i, (inter_view, 1))

            # 添加原始视角
            interpolated_poses.append(pose_i)

            # 添加插值点
            for k in range(len(t_vals)):
                new_pose = np.concatenate([
                    inter_positions[k],
                    inter_quat[k],
                    inter_fov[k]
                ])
                interpolated_poses.append(new_pose)

        # 转换为PyTorch张量并添加回批次
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

    B, V = pred["pose_enc"].shape[:2]  # 获取批次和视图维度
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
    优化显存使用的向量化高斯渲染
    """
    H, W = image_size
    device = pose_enc.device
    batch_size = len(pcs)

    # 提前计算相机参数
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
            last_row.expand(batch_size * view_size, -1, -1)  # 扩展为 [B * V, 1, 4]
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
        # 合并结果并调整维度
        render_images = torch.cat(render_images_list, dim=0)  # [batch_size, 8, H, W, 3]
        render_depths = torch.cat(render_depth_list, dim=0)  # [batch_size, 8, H, W, 3]
        return rearrange(render_images, "b v h w c-> (b v) c h w").contiguous(), rearrange(render_depths,
                                                                                           "b v h w c-> (b v) c h w").contiguous()  # [batch_size, 8, 3, H, W]
    else:
        # 合并结果并调整维度
        render_images = torch.cat(render_images_list, dim=0)  # [batch_size, 8, H, W, 3]
        return rearrange(render_images, "b v h w c-> (b v) c h w").contiguous(), None


def save_rendered_images(images: np.array, save_path: str, epoch=0, start_id=None):
    """
    批量保存渲染图像到指定路径
    Args:
        images: 待保存图像张量 [N, C, H, W]
        save_path: 保存目录路径
        epoch: epoch数
    """
    # 创建保存目录
    save_dir = os.path.join(save_path, str(f"{epoch:02d}"))
    os.makedirs(save_dir, exist_ok=True)

    # 批量保存（每个图像单独保存）
    for b in range(images.size(0)):
        if start_id:
            filename = os.path.join(save_dir, f"render_ep_{epoch:02d}_{start_id * images.size(0) + b:02d}.png")
        else:
            filename = os.path.join(save_dir, f"render_ep_{epoch:02d}_{b:02d}.png")
        save_image(images[b], filename, normalize=False)
