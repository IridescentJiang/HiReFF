import torch
import torch.nn.functional as F


def interpolate_images(images, target_size, mode='bilinear', align_corners=True):
    """
    Generic dimension-aware image interpolation function.

    Args:
        images: Input image tensor (b v c h w) or (b v h w c)
        target_size: Target size (H, W)
        mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        align_corners: Whether to align corners

    Returns:
        Interpolated image, preserving original dimension order
    """
    # Validate input
    if not isinstance(images, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    if images.dim() == 4:
        images = images.unsqueeze(-1)

    if images.dim() != 5:
        raise ValueError("Input must be a 5-dim tensor (b v c h w or b v h w c)")

    # Save original shape and dimension order
    original_shape = images.shape
    is_channels_last = images.shape[-1] < 20  # Channels are in the last dimension

    # Convert to standard format (b v c h w)
    if is_channels_last:
        # Convert from (b v h w c) to (b v c h w)
        images = images.permute(0, 1, 4, 2, 3)

    # Merge batch and view dimensions
    b, v, c, h, w = images.shape
    images_flat = images.reshape(b * v, c, h, w)

    # Perform interpolation
    interpolated_flat = F.interpolate(
        images_flat,
        size=target_size,
        mode=mode,
        align_corners=align_corners
    )

    # Restore original dimensions
    H, W = target_size
    interpolated = interpolated_flat.view(b, v, c, H, W)

    # If needed, restore channels-last format
    if is_channels_last:
        # Convert from (b v c H W) to (b v H W c)
        interpolated = interpolated.permute(0, 1, 3, 4, 2)

    return interpolated
