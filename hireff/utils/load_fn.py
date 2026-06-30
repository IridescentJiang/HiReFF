# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
from typing import List
import cv2
import concurrent.futures
import os


def adjust_intrinsic_batch(K: torch.Tensor, orig_w: int, orig_h: int, target_w: int, target_h: int, type: str = "crop") -> torch.Tensor:
    """
    Adjust intrinsic matrix for scaling and cropping operations.

    Args:
        K: Original intrinsic matrix [N, 3, 3]
        orig_w: Original image width
        orig_h: Original image height
        target_w: Target image width
        target_h: Target image height
        type: Processing type, "crop" or "pad"
            - "crop": Set width to target_w and scale height accordingly, preserving aspect ratio
            - "pad": Preserve original aspect ratio, max dimension = target_w, min dimension padded to square

    Returns:
        Adjusted intrinsic matrix, shape [N, 3, 3]
    """
    if K.ndim != 3 or K.shape[1:] != (3, 3):
        raise ValueError("Input K must be a tensor of shape [N, 3, 3]")

    if type == "crop":
        # Compute scale factor (based on width)
        scale_w = target_w / orig_w

        # Actual intermediate size after scaling
        scaled_h = round(orig_h * scale_w / 14) * 14
        scale_h = scaled_h / orig_h

        # Compute crop amount (top-bottom cropping)
        crop_amount = scaled_h - target_h
        crop_top = crop_amount / 2.0
    else:
        # Compute scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        crop_top = 0

    # Adjust intrinsic
    adjusted_K = K.clone()
    adjusted_K[:, 0, 0] *= scale_w
    adjusted_K[:, 0, 2] *= scale_w
    adjusted_K[:, 1, 1] *= scale_h
    adjusted_K[:, 1, 2] = adjusted_K[:, 1, 2] * scale_h - crop_top

    return adjusted_K


def adjust_intrinsic(K, orig_w, orig_h, target_w, target_h, type="crop"):
    """
    Adjust intrinsic matrix for scaling and cropping operations.

    Args:
        K: Original intrinsic matrix [3x3]
        orig_w: Original image width
        orig_h: Original image height
        target_w: Target image width
        target_h: Target image height
        type: Processing type, "crop" or "pad"
            - "crop": Set width to target_w and scale height accordingly, preserving aspect ratio
            - "pad": Preserve original aspect ratio, max dimension = target_w, min dimension padded to square

    Returns:
        Adjusted intrinsic matrix
    """

    if type == "crop":
        # Compute scale factor (based on width)
        scale_w = target_w / orig_w

        # Actual intermediate size after scaling
        scaled_h = round(orig_h * scale_w / 14) * 14  # Note: orig_h * scale_w
        scale_h = scaled_h / orig_h

        # Compute crop amount (top-bottom cropping)
        crop_amount = scaled_h - target_h  # Total vertical crop pixels
        crop_top = crop_amount / 2.0  # Assuming symmetric top-bottom cropping
    else:
        # Compute scale factors
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        crop_top = 0

    # Adjust intrinsic
    adjusted_K = np.array([
        [K[0][0] * scale_w, 0, K[0][2] * scale_w],  # fx and cx only need width scale
        [0, K[1][1] * scale_h, K[1][2] * scale_h - crop_top],  # fy uses width scale, cy needs crop_top subtracted
        [0, 0, 1]
    ])

    return adjusted_K


def normalize_extrinsic(extrinsic, rat=3):
    """Normalize extrinsic matrix to [-1, 1] range."""
    # Separate rotation and translation
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]

    # Rotation matrix normalization (already in [-1, 1])
    # Translation normalization (assuming max translation 4 meters)
    t_normalized = t / rat

    return np.vstack([
        np.hstack([R, t_normalized.reshape(3, 1)]),
        [0, 0, 0, 1]
    ])


def convert_extrinsics_to_relative_tensor(
    extrinsics_list: List[torch.Tensor] | torch.Tensor,
    scale_factor: float = 1  # Explicit scale parameter
) -> torch.Tensor:
    """
    Convert absolute extrinsic matrices (camera-to-world transforms) to relative transforms
    with respect to the first camera.

    Args:
        extrinsics_list: List of camera-to-world extrinsic matrices [4x4]

    Returns:
        List of relative transform matrices: each matrix represents the transform from the
        reference camera to the current camera [4x4]
    """
    if isinstance(extrinsics_list, torch.Tensor):
        extrinsics_tensor = extrinsics_list
        if extrinsics_tensor.ndim == 2:
            extrinsics_tensor = extrinsics_tensor.unsqueeze(0)
        if extrinsics_tensor.ndim != 3 or extrinsics_tensor.shape[-2:] != (4, 4):
            raise ValueError("extrinsics_list tensor must have shape [N, 4, 4] or [4, 4]")
    else:
        if len(extrinsics_list) == 0:
            return torch.empty((0, 3, 4), dtype=torch.float32)
        extrinsics_tensor = torch.stack(extrinsics_list, dim=0)

    if extrinsics_tensor.shape[0] == 0:
        return torch.empty((0, 3, 4), dtype=extrinsics_tensor.dtype, device=extrinsics_tensor.device)

    # Get reference camera (world -> camera)
    base_inv = torch.linalg.inv(extrinsics_tensor[0])
    R_base = base_inv[:3, :3]
    t_base = base_inv[:3, 3]

    relative_list = []

    for ext in extrinsics_tensor:
        # Current camera (world -> camera)
        cur_inv = torch.linalg.inv(ext)
        R_cur = cur_inv[:3, :3]
        t_cur = cur_inv[:3, 3]

        # Compute relative transform: δ = T_current_w2c * T_base_w2c^{-1}
        R_rel = R_cur @ R_base.T
        t_rel = t_cur - R_rel @ t_base

        # Build matrix and scale translation
        rel_mat = torch.eye(4, dtype=ext.dtype, device=ext.device)
        rel_mat[:3, :3] = R_rel
        rel_mat[:3, 3] = t_rel * scale_factor

        relative_list.append(rel_mat[:3, :4])

    return torch.stack(relative_list, dim=0)


def convert_extrinsics_to_relative(
        extrinsics_list: List[np.ndarray],
        scale_factor: float = 1  # Explicit scale parameter
) -> List[np.ndarray]:
    """
    Convert absolute extrinsic matrices (camera-to-world transforms) to relative transforms
    with respect to the first camera.

    Args:
        extrinsics_list: List of camera-to-world extrinsic matrices [4x4]

    Returns:
        List of relative transform matrices: each matrix represents the transform from the
        reference camera to the current camera [4x4]
    """
    if not extrinsics_list:
        return []

    # Get reference camera (world -> camera)
    base_inv = np.linalg.inv(extrinsics_list[0])
    R_base = base_inv[:3, :3]
    t_base = base_inv[:3, 3]

    relative_list = []

    for ext in extrinsics_list:
        # Current camera (world -> camera)
        cur_inv = np.linalg.inv(ext)
        R_cur = cur_inv[:3, :3]
        t_cur = cur_inv[:3, 3]

        # Compute relative transform: δ = T_current_w2c * T_base_w2c^{-1}
        R_rel = R_cur @ R_base.T
        t_rel = t_cur - R_rel @ t_base

        # Build matrix and scale translation
        rel_mat = np.eye(4)
        rel_mat[:3, :3] = R_rel
        rel_mat[:3, 3] = t_rel * scale_factor

        relative_list.append(rel_mat[:3, :4])

    return relative_list


def load_masked_image(image_path, mask_path, threshold=128):
    """
    Load an image with a mask, returning only the masked region.
    Args:
        image_path: Original image path
        mask_path: Mask image path
        threshold: Mask binarization threshold (0-255)
    Returns:
        masked_image: PIL.Image object, keeping only the masked region
    """
    # Load original image and mask
    img = Image.open(image_path).convert('RGBA')  # Must convert to RGBA format
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale

    if img.size != mask.size:
        mask = mask.resize(img.size, Image.BILINEAR)

    # Convert to numpy arrays for processing
    img_array = np.array(img)
    mask_array = np.array(mask)

    # Binarize mask (True indicates regions to keep)
    binary_mask = (mask_array > threshold)

    # Create fully transparent background (alpha=0)
    transparent_bg = np.zeros_like(img_array)
    transparent_bg[..., 3] = 0  # Alpha channel all zeros

    # Merge: keep pixels in mask region, make others transparent
    masked_array = np.where(
        binary_mask[..., None],  # Add dimension to match RGBA
        img_array,  # Keep original pixels when True
        transparent_bg  # Set transparent when False
    )

    return Image.fromarray(masked_array)


def gen_mask_image(images, masks, bg=torch.tensor([1.0, 1.0, 1.0])):
    # Unify mask shape
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    bg = bg.view(1, 3, 1, 1).to(images.device)

    # Apply mask
    return images * masks + bg * (1 - masks)


def load_and_preprocess_images(image_path_list, target_size=518, mask_path_list=None, Ks=None, Ds=None, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for i, image_path in enumerate(image_path_list):

        # Open image
        if mask_path_list:
            mask_path = mask_path_list[i]
            img = load_masked_image(image_path, mask_path)
        else:
            img = Image.open(image_path)

        if Ks is not None and Ds is not None:
            img_np = np.array(img)
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)

            # Get current image dimensions
            height, width = img_np.shape[:2]

            # Compute ideal principal point (image center)
            ideal_cx = width / 2.0
            ideal_cy = height / 2.0

            # Get actual principal point from current intrinsics
            current_cx = Ks[i][0, 2]
            current_cy = Ks[i][1, 2]

            # Compute required translation (to compensate for off-center offset)
            tx = current_cx - ideal_cx
            ty = current_cy - ideal_cy

            # Build new intrinsic matrix (corrected principal point offset)
            new_K = Ks[i].copy()
            new_K[0, 2] = ideal_cx  # Horizontal principal point set to image center
            new_K[1, 2] = ideal_cy  # Vertical principal point set to image center

            # Perform distortion correction with image translation
            img_np = cv2.undistort(img_np, Ks[i], Ds[i], newCameraMatrix=new_K)

            img = Image.fromarray(img_np)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y: start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
