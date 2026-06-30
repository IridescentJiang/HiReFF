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
    Visualize depth map with pure black background.

    Args:
        result: Depth map tensor
        near: Near plane depth value
        far: Far plane depth value
        cmap_name: Colormap name

    Returns:
        Visualization result tensor [H, W, 3]
    """
    # Create valid depth mask
    mask = rearrange(mask, "b v h w c -> (b v) c h w ").contiguous()
    valid_mask = (result > 1e-5)  # & mask > 0.5

    # If near/far not provided, compute them
    if near is None or far is None:
        if valid_mask.any():
            valid_result = result[valid_mask]
            far = valid_result[:16_000_000].quantile(0.99)
            near = valid_result[:16_000_000].quantile(0.01)
        else:
            near = torch.tensor(0.1)
            far = torch.tensor(100.0)

    # Normalize valid depth values
    normalized = torch.zeros_like(result)
    if valid_mask.any():
        # Take logarithm of valid depth values
        log_result = torch.log(result[valid_mask])
        log_near = torch.log(near)
        log_far = torch.log(far)

        # Normalize to [0, 1]
        normalized[valid_mask] = 1 - (log_result - log_near) / (log_far - log_near)
        normalized[valid_mask] = normalized[valid_mask].clip(0, 1)

    # Apply colormap
    vis = apply_color_map_to_image(normalized, cmap_name)

    # Set background to pure black
    background_mask = ~valid_mask
    if background_mask.any():
        # Ensure vis is a floating-point type
        if not vis.is_floating_point():
            vis = vis.float()

        if background_mask.dim() == 4:
            background_mask = background_mask.unsqueeze(1).expand(-1, -1, 3, -1, -1)
        elif background_mask.dim() == 3:
            background_mask = background_mask.unsqueeze(1).expand(-1, 3, -1, -1)

        # Set background to black
        vis[background_mask] = 0.0

    return vis.squeeze(1)
