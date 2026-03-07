import torch
from matplotlib import cm
from einops import rearrange


def apply_color_map(
        x,
        color_map: str = "inferno",
):
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
        image,
        color_map: str = "inferno",
):
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def vis_depth_map(result: torch.Tensor, mask, near: torch.Tensor = None, far: torch.Tensor = None,
                  cmap_name: str = "turbo") -> torch.Tensor:
    """
    可视化深度图，背景为纯黑色

    参数:
        result: 深度图张量
        near: 近平面深度值
        far: 远平面深度值
        cmap_name: 颜色映射名称

    返回:
        可视化结果张量 [H, W, 3]
    """
    # 创建有效深度掩码
    mask = rearrange(mask, "b v h w c -> (b v) c h w ").contiguous()
    valid_mask = (result > 1e-5)  # & mask > 0.5

    # 如果没有提供near/far，计算它们
    if near is None or far is None:
        if valid_mask.any():
            valid_result = result[valid_mask]
            far = valid_result[:16_000_000].quantile(0.99)
            near = valid_result[:16_000_000].quantile(0.01)
        else:
            near = torch.tensor(0.1)
            far = torch.tensor(100.0)

    # 对有效深度进行归一化
    normalized = torch.zeros_like(result)
    if valid_mask.any():
        # 对有效深度取对数
        log_result = torch.log(result[valid_mask])
        log_near = torch.log(near)
        log_far = torch.log(far)

        # 归一化到 [0, 1]
        normalized[valid_mask] = 1 - (log_result - log_near) / (log_far - log_near)
        normalized[valid_mask] = normalized[valid_mask].clip(0, 1)

    # 应用颜色映射
    vis = apply_color_map_to_image(normalized, cmap_name)

    # 将背景设置为纯黑色
    background_mask = ~valid_mask
    if background_mask.any():
        # 确保vis是浮点类型
        if not vis.is_floating_point():
            vis = vis.float()

        if background_mask.dim() == 4:
            background_mask = background_mask.unsqueeze(1).expand(-1, -1, 3, -1, -1)
        elif background_mask.dim() == 3:
            background_mask = background_mask.unsqueeze(1).expand(-1, 3, -1, -1)

        # 将背景设置为黑色
        vis[background_mask] = 0.0

    return vis.squeeze(1)
