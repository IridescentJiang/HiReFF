import torch
import torch.nn.functional as F


def interpolate_images(images, target_size, mode='bilinear', align_corners=True):
    """
    通用维度感知的图像插值函数

    参数:
        images: 输入图像张量 (b v c h w) 或 (b v h w c)
        target_size: 目标尺寸 (H, W)
        mode: 插值模式 ('bilinear', 'nearest', 'bicubic')
        align_corners: 是否对齐角点

    返回:
        插值后的图像，保持原始维度顺序
    """
    # 验证输入
    if not isinstance(images, torch.Tensor):
        raise TypeError("输入必须是torch.Tensor")

    if images.dim() == 4:
        images = images.unsqueeze(-1)

    if images.dim() != 5:
        raise ValueError("输入必须是5维张量 (b v c h w 或 b v h w c)")

    # 保存原始形状和维度顺序
    original_shape = images.shape
    is_channels_last = images.shape[-1] < 20  # 通道在最后

    # 转换为标准格式 (b v c h w)
    if is_channels_last:
        # 从 (b v h w c) 转换为 (b v c h w)
        images = images.permute(0, 1, 4, 2, 3)

    # 合并批次和视图维度
    b, v, c, h, w = images.shape
    images_flat = images.reshape(b * v, c, h, w)

    # 执行插值
    interpolated_flat = F.interpolate(
        images_flat,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )

    # 恢复原始维度
    H, W = target_size
    interpolated = interpolated_flat.view(b, v, c, H, W)

    # 如果需要，恢复通道在最后的格式
    if is_channels_last:
        # 从 (b v c H W) 转换为 (b v H W c)
        interpolated = interpolated.permute(0, 1, 3, 4, 2)

    return interpolated
