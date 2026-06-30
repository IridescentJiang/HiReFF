# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# DIRTY VERSION, TO BE CLEANED UP

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import threading
import weakref
from math import ceil, floor
from einops import rearrange
from torchvision.models import vgg16
from torchvision.ops import masks_to_boxes
from hireff.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from functools import lru_cache
from hireff.training.lpips.lpips import LPIPS


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


def camera_loss(pred_pose_enc_list, batch, with_auc=False, loss_type="l1", gamma=0.6,
                pose_encoding_type="absT_quaR_FoV", weight_T=1.0, weight_R=1.0, weight_fl=0.5):
    # Extract predicted and ground truth components
    num_predictions = len(pred_pose_enc_list)

    gt_extrinsic = batch['extrinsics']
    gt_intrinsic = batch['intrinsics']
    image_size_hw = batch['images'].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic, gt_intrinsic, image_size_hw,
                                                    pose_encoding_type=pose_encoding_type).to(pred_pose_enc_list.device)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i]
        cur_gt_pose_encoding = gt_pose_encoding[i]

        torch.set_printoptions(
            precision=4,  # Display 4 decimal places
            sci_mode=False  # Disable scientific notation (avoid displaying as 1e-4, etc.)
        )

        loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc.clone(), cur_gt_pose_encoding.clone(),
                                                           loss_type=loss_type)

        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_camera_dict = {
        "loss_camera": loss_camera.item(),
        "loss_T": loss_T.item(),
        "loss_R": loss_R.item(),
        "loss_fl": loss_fl.item()
    }

    if with_auc:
        with torch.no_grad():
            # compute auc
            last_pred_pose_enc = pred_pose_enc_list[-1]

            last_pred_extrinsic, _ = pose_encoding_to_extri_intri(last_pred_pose_enc.detach(), image_size_hw,
                                                                  pose_encoding_type=pose_encoding_type,
                                                                  build_intrinsics=False)

            rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(last_pred_extrinsic.float(), gt_extrinsic.float(),
                                                               gt_extrinsic.device)

            if rel_rangle_deg.numel() == 0 and rel_tangle_deg.numel() == 0:
                rel_rangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)
                rel_tangle_deg = torch.FloatTensor([0]).to(gt_extrinsic.device).to(gt_extrinsic.dtype)

            thresholds = [5, 15]
            for threshold in thresholds:
                loss_dict[f"Rac_{threshold}"] = (rel_rangle_deg < threshold).float().mean()
                loss_dict[f"Tac_{threshold}"] = (rel_tangle_deg < threshold).float().mean()

            _, normalized_histogram = calculate_auc(
                rel_rangle_deg, rel_tangle_deg, max_threshold=30, return_list=True
            )

            auc_thresholds = [30, 10, 5, 3]
            for auc_threshold in auc_thresholds:
                cur_auc = torch.cumsum(
                    normalized_histogram[:auc_threshold], dim=0
                ).mean()
                loss_dict[f"Auc_{auc_threshold}"] = cur_auc

    return loss_camera, loss_camera_dict


def huber_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        delta: float = 1.0,
        reduction: str = 'mean'
) -> torch.Tensor:
    """
    Huber loss function implementation for PyTorch

    Args:
        input (torch.Tensor): predicted values of shape (N, *)
        target (torch.Tensor): ground truth values of shape (N, *)
        delta (float, optional): threshold where loss transitions from L2 to L1. Default: 1.0
        reduction (str): reduction method ['none', 'mean', 'sum']. Default: 'mean'

    Returns:
        torch.Tensor: computed loss value
    """
    # Input dimension check
    assert input.shape == target.shape, "Input and target shapes must match"
    assert reduction in ['none', 'mean', 'sum'], "Invalid reduction method"

    # Device check
    assert input.device == target.device, "Input and target must at the same device."

    # Compute absolute error
    error = input - target
    abs_error = torch.abs(error)

    # Compute losses for both regimes
    quadratic = torch.min(abs_error, torch.tensor(delta, device=error.device))
    linear = abs_error - quadratic

    # Combine Huber loss
    loss = 0.5 * quadratic ** 2 + delta * linear

    # Process output according to reduction parameter
    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(cur_pred_pose_enc[..., :3], gt_pose_encoding[..., :3])
        loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[..., 3:7])
        loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[..., 7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100)  # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1, 2, 3])
    valid_count = valid_mask.sum(dim=[1, 2, 3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    # avg_scale = avg_scale.view(-1, 1, 1, 1, 1)

    pts3d = pts3d / avg_scale.view(-1, 1, 1, 1, 1)
    return pts3d, avg_scale


@lru_cache(maxsize=None)
def get_perceptual_loss_model(device='cuda'):
    """Process-safe model acquisition method (lru_cache guarantees singleton)"""
    vgg = vgg16(pretrained=True).features[:5]
    vgg = vgg.half().to(device)
    vgg.requires_grad_(False)

    return vgg


class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda', patch_size=256):
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target):
        # Merge two forward passes into one computation
        concatenated = torch.cat([input, target], dim=0)
        features = self.vgg(concatenated)

        # Split feature results
        vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

        # Ensure gradients only flow from the input branch
        return self.l1_loss(vgg_input, vgg_target.detach())


class PerceptualLossPatch(nn.Module):
    def __init__(self, device='cuda', patch_size=256):
        """
        Perceptual loss based on valid regions (simplified version)

        Args:
            device: compute device
            patch_size: sampled patch size
        """
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()
        self.patch_size = patch_size

    def find_valid_region(self, image):
        """
        Detect valid pixel regions in the image (non-zero regions)

        Returns:
            bbox: (x_min, y_min, x_max, y_max) bounding box of the valid region
        """
        # Create binary mask (non-zero pixels)
        mask = (image != 0).any(dim=0)  # [H, W]

        # If no valid pixels, return the entire image
        if not mask.any():
            return (0, 0, image.shape[1], image.shape[2]), -1

        # Find coordinates of valid pixels
        coords = torch.nonzero(mask)

        # Compute bounding box
        x_min = coords[:, 0].min().item()
        x_max = coords[:, 0].max().item()
        y_min = coords[:, 1].min().item()
        y_max = coords[:, 1].max().item()

        area_size = (x_max - x_min + 1) * (y_max - y_min + 1)

        return (x_min, y_min, x_max, y_max), area_size

    def sample_patch(self, image, target, bbox):
        """
        Randomly sample a patch from the upper-left part of the valid region

        Args:
            image: input image [C, H, W]
            bbox: bounding box of the valid region (x_min, y_min, x_max, y_max)

        Returns:
            patch: sampled image patch [C, patch_size, patch_size]
        """
        x_min, y_min, x_max, y_max = bbox

        # Compute the size of the valid region
        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

        # Compute the samplable region
        if bbox_width > self.patch_size:
            max_x = x_min + (bbox_width - self.patch_size)
            start_x = torch.randint(x_min, max_x + 1, (1,)).item()
            end_x = start_x + self.patch_size
        else:
            start_x = x_min
            end_x = start_x + bbox_width

        if bbox_height > self.patch_size:
            max_y = y_min + (bbox_height - self.patch_size)
            start_y = torch.randint(y_min, max_y + 1, (1,)).item()
            end_y = start_y + self.patch_size
        else:
            start_y = y_min
            end_y = start_y + bbox_height

        # Extract image patches
        patch_img = image[:, start_x:end_x, start_y:end_y]
        patch_target = target[:, start_x:end_x, start_y:end_y]

        return patch_img, patch_target

    def forward(self, input, target):
        """
        Perceptual loss based on valid regions (simplified version)

        Args:
            input: input image [B, C, H, W]
            target: target image [B, C, H, W]

        Returns:
            perceptual loss value
        """
        # Ensure input and target shapes match
        assert input.shape == target.shape, "Input and target shape mismatch"

        total_loss = 0
        batch_size, C, H, W = input.shape

        # Process each image
        for i in range(batch_size):
            input_img = input[i]  # [C, H, W]
            target_img = target[i]  # [C, H, W]

            # Find valid region
            input_bbox, area_size = self.find_valid_region(input_img)

            if area_size == -1:
                continue
            elif area_size > (H * W) / 2:
                # print("Inaccurate mask, skipping LPIP.")
                # print(input_bbox, area_size)
                continue
            elif area_size < 50:
                continue

            # Sample the upper-left patch
            input_patch, target_patch = self.sample_patch(input_img, target_img, input_bbox)

            if input_patch.shape != target_patch.shape:
                print("Patch shapes do not match, skipping LPIP.")
                continue

            # Concatenate patches
            concatenated = torch.cat([input_patch.unsqueeze(0), target_patch.unsqueeze(0)], dim=0)

            # Compute features
            features = self.vgg(concatenated)

            # Split features
            vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

            # Compute loss for the current patch
            patch_loss = self.l1_loss(vgg_input, vgg_target.detach())

            # Accumulate loss
            total_loss += patch_loss / batch_size

        return total_loss


class PerceptualLossPatch_fix(nn.Module):
    def __init__(self, device='cuda', patch_size=256, min_nonbg=100, max_attempts=20):
        """
        Perceptual loss based on valid regions (improved version)

        Args:
            device: compute device
            patch_size: sampled patch size
            min_nonzero: minimum number of non-zero pixels
            max_attempts: maximum number of attempts
        """
        super().__init__()
        self.vgg = get_perceptual_loss_model(device)
        self.l1_loss = nn.L1Loss()
        self.patch_size = patch_size
        self.min_nonbg = min_nonbg
        self.max_attempts = max_attempts

    def sample_patch(self, image, target, bbox):
        x_min, y_min, x_max, y_max = bbox

        # Extract image patches
        patch_img = image[:, x_min:x_max, y_min:y_max]
        patch_target = target[:, x_min:x_max, y_min:y_max]

        return patch_img, patch_target

    def count_nonbg_pixels(self, patch):
        """
        Count the number of non-background pixels in an image patch

        Args:
            patch: image patch [C, H, W]

        Returns:
            number of non-background pixels
        """
        # Create mask: at least one channel is not 0 (background)
        mask = (patch != 0).any(dim=0)  # [H, W]
        return mask.sum().item()

    def generate_random_patch_coords(self, H, W):
        """
        Generate coordinates for a random sampled patch

        Args:
            H: image height
            W: image width

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        # Compute the number of possible sampling positions
        n_H = max(1, int(H / (self.patch_size * 0.75)))
        n_W = max(1, int(W / (self.patch_size * 0.75)))

        # Randomly select positions
        random_n_H = torch.randint(0, n_H + 1, (1,)).item()
        random_n_W = torch.randint(0, n_W + 1, (1,)).item()

        # Compute coordinates
        x_min = int(random_n_H * self.patch_size * 0.75)
        y_min = int(random_n_W * self.patch_size * 0.75)
        x_max = min(x_min + self.patch_size, H)
        y_max = min(y_min + self.patch_size, W)

        return (x_min, y_min, x_max, y_max)

    def forward(self, input, target):
        """
        Perceptual loss based on valid regions (improved version)

        Args:
            input: input image [B, C, H, W]
            target: target image [B, C, H, W]

        Returns:
            perceptual loss value
        """
        # Ensure input and target shapes match
        assert input.shape == target.shape, "Input and target shape mismatch"

        total_loss = 0
        batch_size, C, H, W = input.shape

        # Process each image
        for i in range(batch_size):
            input_img = input[i]  # [C, H, W]
            target_img = target[i]  # [C, H, W]

            attempts = 0
            valid_patch_found = False

            # Attempt to sample a valid patch
            while attempts < self.max_attempts and not valid_patch_found:
                # Generate random patch coordinates
                patch_coords = self.generate_random_patch_coords(H, W)

                # Sample patch
                input_patch, target_patch = self.sample_patch(input_img, target_img, patch_coords)

                # Check if patch shapes match
                if input_patch.shape != target_patch.shape:
                    attempts += 1
                    continue

                # Count non-background pixels
                nonbg_count = self.count_nonbg_pixels(input_patch)

                # Check if minimum non-background pixel requirement is met
                if nonbg_count >= self.min_nonbg:
                    valid_patch_found = True
                else:
                    attempts += 1

            # If no valid patch found, use the last sampled patch
            if not valid_patch_found:
                print(f"Warning: No matching blocks found for image {i}, using last sample")

            # Concatenate patches
            concatenated = torch.cat([input_patch.unsqueeze(0), target_patch.unsqueeze(0)], dim=0)

            # Compute features
            features = self.vgg(concatenated)

            # Split features
            vgg_input, vgg_target = torch.chunk(features, 2, dim=0)

            # Compute loss for the current patch
            patch_loss = self.l1_loss(vgg_input, vgg_target.detach())

            # Accumulate loss
            total_loss += patch_loss / batch_size

        return total_loss


def extract_mask_from_image(image, background_value=1.0, tolerance=0.2):
    """
    Extract mask from image (regions where the background matches a specific value), no gradient needed

    Args:
        image: input image [C, H, W]
        background_value: background value (white is typically 1.0)
        tolerance: tolerance range

    Returns:
        binary mask [1, H, W], foreground=1, background=0 (no gradient needed)
    """
    with torch.no_grad():
        # Compute difference between each pixel and the background value
        diff = torch.abs(image - background_value)

        # For RGB images, take the maximum difference across all channels
        if image.dim() == 3 and image.size(0) > 1:
            diff = torch.max(diff, dim=0, keepdim=True)[0]

        # Create mask (background region=0, foreground region=1)
        mask = torch.where(diff > tolerance, 1.0, 0.0)

        # Ensure mask is 2D or 3D
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

    return mask.detach()


def get_largest_connected_component(mask, ksize=3):
    """
    Efficient GPU-accelerated largest connected component extraction

    Uses morphological operations + region growing instead of traditional connected component analysis
    Suitable for PyTorch GPU compute environments

    Args:
        mask: input mask [1, H, W] (0/1 values)
        ksize: morphological operation kernel size (default 3)

    Returns:
        mask containing only the largest connected component [1, H, W]
    """
    with torch.no_grad():
        # Ensure single channel
        if mask.shape[0] > 1:
            mask = mask[:1]

        # 1. Morphological opening for denoising
        kernel = torch.ones(ksize, ksize, device=mask.device) / (ksize ** 2)
        kernel = kernel.view(1, 1, ksize, ksize)

        # Dilation and erosion operations
        padding = ksize // 2
        smoothed_mask = F.conv2d(mask.unsqueeze(0), kernel, padding=padding).squeeze(0)
        smoothed_mask = (smoothed_mask > 0.5).float()

        # 2. Extract brightest pixel as seed point
        max_val = torch.max(smoothed_mask)
        seed_points = (smoothed_mask == max_val).float()

        # 3. GPU-accelerated region growing algorithm
        current_mask = seed_points
        max_iter = 50  # Safety bound

        for _ in range(max_iter):
            # Diffuse to adjacent regions
            expanded = F.conv2d(current_mask.unsqueeze(0),
                                torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]],
                                             dtype=torch.float32, device=mask.device),
                                padding=1).squeeze(0)

            # Constrain within original binary mask
            new_mask = torch.where(expanded > 0, smoothed_mask, torch.zeros_like(expanded))

            # Check for convergence
            if torch.all(new_mask == current_mask):
                break

            current_mask = new_mask

        # 4. Refine boundaries
        return current_mask * mask


def label_connected_components(binary_mask):
    """
    Connected component labeling using PyTorch (no gradient needed)

    Args:
        binary_mask: binary mask [H, W]

    Returns:
        labeled: labeled connected components [H, W]
        num_labels: number of connected components
    """
    from queue import Queue

    with torch.no_grad():
        # Use 4-connectivity
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Initialization
        h, w = binary_mask.shape
        labeled = torch.zeros((h, w), dtype=torch.int32)
        current_label = 1
        queue = Queue()

        # Iterate over each pixel
        for i in range(h):
            for j in range(w):
                # Skip background or already labeled
                if binary_mask[i, j] == 0 or labeled[i, j] > 0:
                    continue

                # New connected component
                queue.put((i, j))
                labeled[i, j] = current_label

                # Breadth-first search
                while not queue.empty():
                    x, y = queue.get()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < h and 0 <= ny < w and
                                binary_mask[nx, ny] == 1 and labeled[nx, ny] == 0):
                            labeled[nx, ny] = current_label
                            queue.put((nx, ny))

                current_label += 1

        return labeled, current_label - 1


def align_foreground_objects(pred_img, tgt_img,
                             background_value=1.0,
                             min_crop_size=128,
                             output_size=128,
                             pad_value=1.0,
                             large_bbox_threshold=0.5):
    """
    Align foreground objects (only the largest connected component), but skip alignment when
    the object bounding box occupies more than a threshold fraction of the image

    Note: all image operations preserve gradients, mask operations do not require gradients

    Args:
        pred_img: predicted image [C, H, W] (with gradient)
        tgt_img: target image [C, H, W] (with gradient)
        background_value: background value
        min_crop_size: minimum crop size
        output_size: output size
        pad_value: padding value
        large_bbox_threshold: large bbox area determination threshold (default 2/3=0.66)

    Returns:
        aligned_pred: aligned predicted image [C, output_size, output_size] (with gradient)
        aligned_tgt: aligned target image [C, output_size, output_size] (with gradient)
        combined_mask: combined foreground mask [1, output_size, output_size] (no gradient needed)
    """
    device = pred_img.device

    # Get original image dimensions
    _, h, w = pred_img.shape
    img_area = h * w

    # 1. Extract mask and get the largest connected component (no gradient needed)
    with torch.no_grad():
        # Extract mask
        pred_mask = extract_mask_from_image(pred_img, background_value)
        tgt_mask = extract_mask_from_image(tgt_img, background_value)

        # Extract largest connected component
        pred_mask_cc = get_largest_connected_component(pred_mask)
        tgt_mask_cc = get_largest_connected_component(tgt_mask)

        # If mask is all zero (no foreground), use the entire image
        if pred_mask_cc.sum() == 0:
            pred_mask = torch.ones_like(pred_mask)
        else:
            pred_mask = pred_mask_cc

        if tgt_mask_cc.sum() == 0:
            tgt_mask = torch.ones_like(tgt_mask)
        else:
            tgt_mask = tgt_mask_cc

        # Get bounding boxes
        pred_box = masks_to_boxes(pred_mask)
        tgt_box = masks_to_boxes(tgt_mask)

        # Compute bounding box area
        def bbox_area(bbox):
            if bbox.numel() > 0:
                x1, y1, x2, y2 = bbox[0]
                return (x2 - x1) * (y2 - y1)
            return 0.0

        pred_bbox_area = bbox_area(pred_box)
        tgt_bbox_area = bbox_area(tgt_box)

        # Determine if it is a large bounding box
        is_large_pred = pred_bbox_area > large_bbox_threshold * img_area
        is_large_tgt = tgt_bbox_area > large_bbox_threshold * img_area

        # If either bounding box is larger than the threshold, skip alignment
        if is_large_pred or is_large_tgt:
            # Directly resize the original image (preserve gradient)
            aligned_pred = adaptive_resize_and_pad(pred_img, output_size, pad_value)
            aligned_tgt = adaptive_resize_and_pad(tgt_img, output_size, pad_value)

            # Extract mask (no gradient needed)
            with torch.no_grad():
                aligned_pred_mask = extract_mask_from_image(aligned_pred, background_value)
                aligned_tgt_mask = extract_mask_from_image(aligned_tgt, background_value)
                combined_mask = torch.max(aligned_pred_mask, aligned_tgt_mask)

            return aligned_pred, aligned_tgt, combined_mask

        # Compute the minimum region covering both bounding boxes
        if pred_box.shape[1] > 0 and tgt_box.shape[1] > 0:
            combined_box = torch.cat([pred_box, tgt_box])
            min_coords = torch.min(combined_box[:, :2], dim=0)[0]
            max_coords = torch.max(combined_box[:, 2:], dim=0)[0]
        else:
            # If a mask has no bounding box, use the entire image
            min_coords = torch.tensor([0, 0], device=device)
            max_coords = torch.tensor([w, h], device=device)

        # Ensure minimum size
        w_size, h_size = max_coords - min_coords
        if w_size < min_crop_size or h_size < min_crop_size:
            center = (min_coords + max_coords) / 2
            half_size = max(min_crop_size / 2, max(w_size, h_size) / 2)
            min_coords = torch.clamp(center - half_size, 0)
            max_coords = torch.clamp(center + half_size, 0, pred_img.shape[-1])

        # Convert to integer coordinates
        min_coords = torch.floor(min_coords).int()
        max_coords = torch.ceil(max_coords).int()

        # Ensure valid crop region
        x1, y1 = min_coords[0].clamp(0), min_coords[1].clamp(0)
        x2, y2 = max_coords[0].clamp(0, w), max_coords[1].clamp(0, h)

        # If crop region is invalid, use the entire image
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h

    # 2. Crop images (preserve gradient)
    # Note: slicing preserves the original gradient
    cropped_pred = pred_img[:, y1:y2, x1:x2]
    cropped_tgt = tgt_img[:, y1:y2, x1:x2]

    # 3. Resize and pad to uniform size (preserve gradient)
    aligned_pred = adaptive_resize_and_pad(cropped_pred, output_size, pad_value)
    aligned_tgt = adaptive_resize_and_pad(cropped_tgt, output_size, pad_value)

    # 4. Create combined foreground mask (no gradient needed)
    with torch.no_grad():
        aligned_pred_mask = extract_mask_from_image(aligned_pred, background_value)
        aligned_pred_mask = get_largest_connected_component(aligned_pred_mask)

        aligned_tgt_mask = extract_mask_from_image(aligned_tgt, background_value)
        aligned_tgt_mask = get_largest_connected_component(aligned_tgt_mask)

        combined_mask = torch.max(aligned_pred_mask, aligned_tgt_mask)

    return aligned_pred, aligned_tgt, combined_mask


def adaptive_resize_and_pad(image, output_size, pad_value=1.0):
    """
    Aspect-ratio-preserving resize and padding (preserves gradient)

    Args:
        image: input image [C, H, W] (with gradient)
        output_size: target output size
        pad_value: padding value

    Returns:
        resized and padded image [C, output_size, output_size] (with gradient)
    """
    # If the image is empty, return directly
    if image.numel() == 0:
        return torch.full((image.shape[0], output_size, output_size), pad_value, device=image.device)

    # Get original dimensions
    _, h, w = image.shape

    # Compute scale ratio
    scale = min(output_size / h, output_size / w)

    # Compute new dimensions
    new_h, new_w = int(h * scale), int(w * scale)
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Resize (preserve gradient)
    resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w),
                            mode='bilinear', align_corners=False).squeeze(0)

    # Compute padding
    pad_top = (output_size - new_h) // 2
    pad_bottom = output_size - new_h - pad_top
    pad_left = (output_size - new_w) // 2
    pad_right = output_size - new_w - pad_left

    # Apply padding (preserve gradient)
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom),
                   value=pad_value)

    return padded


def pts_trans_loss(pts_trans, target, radio, lambda_l1=1.0):
    """
    Compute point cloud translation loss
    Uses the camera extrinsic translation matrix from target to compute the point cloud translation loss
    """
    device = pts_trans.device

    transl = pts_trans[:, :, :3]
    extrinsics = target["extrinsics"]

    T = extrinsics[:, :, :3, 3]  # BxSx3
    target_transl = T * (radio - 1)

    l1_criterion = nn.L1Loss(reduction='none')
    l1_loss_map = l1_criterion(transl, target_transl)
    l1_loss = l1_loss_map.mean()  # Average loss per point

    losses = {}
    losses['l1'] = l1_loss

    total_loss = torch.tensor(0.0, device=device)
    total_loss += lambda_l1 * l1_loss

    return total_loss, losses


class RenderLoss:
    """
    Improved rendering loss function: applies higher weight to masked regions
    """

    def __init__(self, device):
        self.lpips = LPIPS(net='vgg').half().to(device)
        self.lpips.requires_grad_(False)
        self.bg_color_cuda = torch.tensor([1.0, 1.0, 1.0], device=device).to(torch.float32)

    def compute_lpips_loss(self, image, gt_image):
        """
            image: [N, C, H ,W]
        """
        assert image.shape[2] == image.shape[3] and gt_image.shape[2] == gt_image.shape[3]
        lpips_loss = self.lpips.forward(
            image[:, [2, 1, 0], :, :],
            gt_image[:, [2, 1, 0], :, :],
            normalize=True
        ).mean()
        return lpips_loss

    def crop_image(self, gt_mask, patch_size, randomly, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """
        mask_uv = torch.argwhere(gt_mask > 0.)
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        if randomly and max_size > patch_size:
            random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
            random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        for image in args:
            cropped_image = self.bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype=image.dtype,
                                                                           device=image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

            if randomly and max_size > patch_size:
                cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
            else:
                cropped_image = F.interpolate(cropped_image[None], size=(patch_size, patch_size), mode='bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imshow('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imshow('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.waitKey(0)

        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]

    def forward(self, images, masks, target_images, loss_type="perceptual+l1",
                lambda_perceptual=0.1, lambda_l1=1.0, lambda_mse=0.5,
                mask_weight_factor=2.0, edge_weight_factor=3.0,
                min_crop_size=128, output_size=518,
                background_value=1.0, algining=False,
                random_patch=False, patch_size=256):
        """
        Args:
            images: predicted images [B*V, C, H, W]
            masks: predicted image masks [B, V, H, W, C]
            target: dictionary containing target images, expects "images" key
            loss_type: loss combination method ("perceptual", "l1", "perceptual+l1")
            lambda_perceptual: perceptual loss weight
            lambda_l1: L1 loss weight
            mask_weight_factor: mask region loss weight multiplier
            edge_weight_factor: edge region additional weight multiplier

        Returns:
            combined loss value and individual loss components
        """
        # Reshape target image tensor
        target_images = rearrange(target_images, "b v c h w -> (b v) c h w").contiguous()
        masks = rearrange(masks, "b v h w c -> (b v) c h w").contiguous()

        N, C, H, W = target_images.shape
        masks = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=True)

        device = images.device

        if algining:
            # Align each pair of images
            aligned_preds = []
            aligned_targets = []
            aligned_masks = []

            for i in range(len(images)):
                pred_img = images[i]
                tgt_img = target_images[i]

                # Align foreground objects
                aligned_pred, aligned_tgt, combined_mask = align_foreground_objects(
                    pred_img, tgt_img,
                    background_value=background_value,
                    min_crop_size=min_crop_size,
                    output_size=output_size,
                    pad_value=background_value
                )

                aligned_preds.append(aligned_pred)
                aligned_targets.append(aligned_tgt)
                aligned_masks.append(combined_mask)

            # Convert to tensors
            images = torch.stack(aligned_preds)
            target_images = torch.stack(aligned_targets)
            masks = torch.stack(aligned_masks)

        # Create weight map: apply higher weight to masked regions
        if mask_weight_factor > 1.0:
            # Normalize mask shape (B*V, 1, H, W)
            if masks.dim() > 2 and masks.size(1) == 3:  # RGB mask
                masks = masks.mean(dim=1, keepdim=True)  # Convert to single channel

            # Base weight map (background=1, foreground=mask_weight_factor)
            base_weights = torch.ones_like(masks)  # Background weight=1
            base_weights[masks > 0.5] = mask_weight_factor  # Weighted region

            # Edge enhancement (optional)
            if edge_weight_factor > mask_weight_factor:
                from torch.nn.functional import conv2d
                kernel = torch.tensor([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=torch.float32, device=device)
                kernel = kernel.view(1, 1, 3, 3) / kernel.sum()

                # Detect edge regions
                with torch.no_grad():
                    dilated = conv2d(masks, kernel, padding=1)
                    edges = (dilated > 0.2) & (dilated < 0.8)
                    base_weights[edges] = edge_weight_factor

            # Expand weight map to match all channels
            weights = base_weights.expand_as(images)  # Copy to RGB channels
        else:
            weights = torch.ones_like(images)  # No additional weights applied

        # Initialize loss modules
        perceptual_criterion = PerceptualLossPatch(device=device, patch_size=patch_size)
        l1_criterion = nn.L1Loss(reduction='none')  # Set to none for weighting

        # Compute individual loss components
        losses = {}

        # L1 loss (per-pixel, can be weighted)
        if "l1" in loss_type:
            l1_loss_map = l1_criterion(images, target_images)
            weighted_l1 = (l1_loss_map * weights).mean()
            losses['l1'] = weighted_l1

        # MSE loss (per-pixel, can be weighted)
        if "mse" in loss_type:
            mse_loss_map = F.mse_loss(images, target_images, reduction='none')
            weighted_mse = (mse_loss_map * weights).mean()
            losses['mse'] = weighted_mse

        # Perceptual loss (feature-level, typically not directly weighted)
        if "perceptual" in loss_type:
            # Perceptual loss is not directly weighted, but can be extracted via ROI
            if mask_weight_factor > 1.0:
                # Extract foreground region
                with torch.no_grad():
                    mask_roi = (masks > 0.5).float()

                # Compute foreground and background perceptual losses separately

                # LPIPS from animatable gaussian
                N = mask_roi.shape[0]
                image_list = []
                gt_image_list = []
                for i in range(N):
                    if mask_roi[i, 0].sum() < 100:
                        # Entirely background, skip
                        continue
                    image, gt_image = self.crop_image(mask_roi[i, 0], patch_size, random_patch, images[i],
                                                      target_images[i])
                    image_list.append(image)
                    gt_image_list.append(gt_image)
                image_list = torch.stack(image_list, dim=0)
                gt_image_list = torch.stack(gt_image_list, dim=0)
                fg_perceptual = self.compute_lpips_loss(image_list, gt_image_list)

                # fg_perceptual = perceptual_criterion(images * mask_roi, target_images * mask_roi)
                # bg_perceptual = perceptual_criterion(images * (1 - mask_roi), target_images * (1 - mask_roi))

                # Apply higher weight to foreground
                # losses['perceptual'] = (fg_perceptual * mask_weight_factor + bg_perceptual) / 2

                losses['perceptual'] = fg_perceptual
            else:
                # losses['perceptual'] = self.compute_lpips_loss(images, target_images)
                losses['perceptual'] = perceptual_criterion(images, target_images)

        # Weighted sum
        total_loss = torch.tensor(0.0, device=device)
        if 'perceptual' in losses:
            total_loss += lambda_perceptual * losses['perceptual']
        if 'l1' in losses:
            total_loss += lambda_l1 * losses['l1']
        if 'mse' in losses:
            total_loss += lambda_mse * losses['mse']  # Uses the same weight as L1

        # Optional: add foreground existence regularization to prevent all-black predictions
        if mask_weight_factor > 1.0:
            foreground_coverage = masks.mean()  # Average coverage
            # Ensure at least a certain proportion of foreground
            coverage_constraint = torch.maximum(torch.tensor(0.05, device=device) - foreground_coverage,
                                                torch.tensor(0.0, device=device))
            losses['coverage'] = coverage_constraint * mask_weight_factor * 10.0
            total_loss += losses['coverage']

        return total_loss, {k: v.item() for k, v in losses.items()}


def chamfer_distance(pred, target):
    """
    Bidirectional Chamfer distance implementation
    """
    # Pred -> target

    dist_pred_target = torch.cdist(pred, target)  # [B, N, M]
    min_dist1, _ = dist_pred_target.min(2)  # [B, N]

    # Target -> pred
    min_dist2, _ = dist_pred_target.min(1)  # [B, M]

    return 0.5 * (min_dist1.mean() + min_dist2.mean())


def estimate_normals(points):
    """
    Estimate normals of a point cloud
    Uses a simple k-NN method to estimate normals
    """
    k = 16  # Number of nearest neighbors
    B, N, _ = points.shape

    # Build k-NN graph
    dists = torch.cdist(points, points)  # [B, N, N]
    _, topk_idx = torch.topk(dists, k + 1, largest=False)  # [B, N, k+1]
    nn_points = torch.gather(points.unsqueeze(2), 2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # [B, N, k+1, 3]

    # Average normals
    normals = nn_points[:, :, 1:] - nn_points[:, :, :1]  # [B, N, k, 3]
    normals = normals.mean(dim=2)  # [B, N, 3]

    return normals / (normals.norm(dim=-1, keepdim=True) + 1e-6)  # Normalize normals


def normal_consistency_loss(pred, target):
    """
    Normal consistency between predicted point cloud and pseudo-label point cloud
    """
    pred_normals = estimate_normals(pred)  # [B, N, 3]
    target_normals = estimate_normals(target)  # [B, M, 3]

    # Find the nearest neighbor normals
    dist_matrix = torch.cdist(pred, target)
    _, nn_idx = dist_matrix.min(2)  # [B, N]

    # Gather corresponding normals
    nn_normals = torch.gather(
        target_normals,
        1,
        nn_idx.unsqueeze(-1).expand(-1, -1, 3)
    )

    # Compute cosine similarity
    cos_sim = F.cosine_similarity(pred_normals, nn_normals, dim=-1)
    return 1.0 - cos_sim.mean()


def uniform_point_loss(points, k=16):
    """
    Penalizes non-uniform point density distribution
    """
    # 1. Build k-NN graph
    dists = torch.cdist(points, points)  # [B, N, N]

    # 2. Compute local density variance
    _, topk_idx = torch.topk(dists, k + 1, largest=False)  # [B, N, k+1]
    nn_dists = torch.gather(dists, 2, topk_idx)[..., 1:]  # Exclude self

    # 3. Compute local region density variance
    density_vars = torch.var(nn_dists, dim=-1)  # [B, N]
    return density_vars.mean()


def distill_geometry_loss(points, masks, pseudo_label_points, loss_type="chamfer+uniform", lambda_chamfer=0.5,
                          lambda_uniform=0.2, downsample_ratio=0.01):
    """
    Compute geometry loss via point clouds and pseudo-label points

    Args:
        points: predicted points [B, V, H, W, C]
        target: dictionary containing target point cloud, expects "pseudo_label_points" key. target["pseudo_label_points"]: [B, V, H, W, C]
        masks: predicted image masks [B, V, H, W, C]
        loss_type: loss combination method ("chamfer", "normal", "uniform")

    Returns:
        combined loss value
    """

    # Reshape target point cloud tensor
    target_points = rearrange(pseudo_label_points, "b v h w c -> (b v) (h w) c").contiguous()
    points = rearrange(points, "b v h w c -> (b v) (h w) c").contiguous()
    masks = rearrange(masks, "b v h w c -> (b v) (h w) c").contiguous()

    device = points.device

    # Keep only valid points
    valid_mask = masks > 0.5  # Assume mask value > 0.5 indicates valid points
    valid_mask = valid_mask.expand_as(points)  # [B*V, H*W, C]
    points = points * valid_mask
    target_points = target_points * valid_mask

    # Downsample point cloud
    target_num = int(points.shape[1] * downsample_ratio)

    points = downsample_pointcloud(points, target_num, method='random')
    target_points = downsample_pointcloud(target_points, target_num, method='random')

    # Change to use chamfer + uniform loss
    losses = {}
    if "chamfer" in loss_type:
        losses['chamfer'] = chamfer_distance(points, target_points)
    if "uniform" in loss_type:
        losses['uniform'] = uniform_point_loss(points)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    # Weighted sum
    total_loss = torch.tensor(0.0, device=device)
    if 'chamfer' in losses:
        total_loss += lambda_chamfer * losses['chamfer']
    if 'uniform' in losses:
        total_loss += lambda_uniform * losses['uniform']

    return total_loss, {k: v.item() for k, v in losses.items()}


def depth_consist_loss(rendered_depth, masks, target_depth, loss_type):
    """
    Compute depth consistency loss

    Args:
        rendered_depth: rendered depth map [B*V, 1, H, W]
        masks: valid masks [B, V, 1, H, W]
        target_depth: target depth map [B, V, 1, H, W]
        loss_type: loss type ('MSE')

    Returns:
        depth_loss: depth consistency loss
    """
    B, V, H, W, C = target_depth.shape

    all_views = rendered_depth.shape[0] // B
    rendered_depth = rendered_depth.view(B, all_views, H, W, C)[:, :V]

    masks = rearrange(masks[:, :V], "b v c h w -> b v h w c").contiguous()
    valid_mask = (rendered_depth > 1e-5) & (masks > 0.5)

    if 'MSE' in loss_type:
        depth_loss = F.mse_loss(rendered_depth, target_depth, reduction='none')[valid_mask].mean()
    else:
        depth_loss = None

    loss = torch.nan_to_num(depth_loss, nan=0.0)
    return loss, {"mse_loss": loss.item()}


def distill_depth_loss(pred_depth, masks, target_depth):
    """
    Depth distillation loss function - uses only MSE loss

    Args:
        pred_depth: model predicted depth map [B, V, H, W]
        target_depth: target depth map (ground truth or pseudo-label) [B, V, H, W]
        masks: valid region masks [B, V, H, W]

    Returns:
        MSE loss value
    """

    # Apply valid mask
    valid_mask = masks > 0.5  # Assume mask value > 0.5 indicates valid points
    valid_mask = valid_mask.expand_as(pred_depth)  # Ensure dimension match

    # Count number of valid pixels (avoid division by zero)
    num_valid = torch.sum(valid_mask).clamp(min=1)

    # Compute masked mean squared error (MSE)
    diff = pred_depth - target_depth
    squared_diff = diff * diff
    masked_squared_diff = squared_diff * valid_mask.float()

    # Average loss
    loss = torch.sum(masked_squared_diff) / num_valid

    return loss, {"mse_loss": loss.item()}


def distill_transformer_loss(feature_list, feature_list_label):
    """
    Multi-feature distillation loss function - standard MSE implementation

    Args:
        feature_list: model predicted feature list [B, V, H, W] * N
        feature_list_label: target feature list (ground truth or pseudo-label) [B, V, H, W] * N

    Returns:
        average MSE loss value
    """
    total_loss = 0.0
    num_features = len(feature_list)

    # Ensure feature lists have the same length
    assert len(feature_list) == len(feature_list_label), \
        "Feature list and target list length mismatch"

    # Compute loss for each feature pair
    losses = []
    for feature, feature_label in zip(feature_list, feature_list_label):
        # Check if feature shapes match
        assert feature.shape == feature_label.shape, \
            f"Feature shape mismatch: {feature.shape} vs {feature_label.shape}"

        # Count number of elements
        num_elements = feature.numel()

        # Compute MSE loss for a single feature
        loss = F.mse_loss(feature, feature_label, reduction='sum') / num_elements
        losses.append(loss)
        total_loss += loss

    # Compute average loss
    avg_loss = total_loss / num_features

    # Collect loss metrics
    metrics = {"avg_mse_loss": avg_loss.item()}
    for i, loss in enumerate(losses):
        metrics[f"mse_loss_{i}"] = loss.item()

    return avg_loss, metrics


def mask_loss(masks, target_masks, loss_type="Dice+BCE", lambda_BCE=0.5, lambda_Dice=0.5):
    """
    Combined mask loss function

    Args:
        masks: predicted masks [B, V, H, W, C]
        target: dictionary containing target images, expects "mask" key. target["masks"]: [B, V, C, H, W]
        loss_type: loss combination method ("Dice", "BCE", "Dice+BCE")

    Returns:
        combined loss value
    """

    # Reshape target image tensor
    target_masks = rearrange(target_masks, "b v c h w -> (b v) c h w").contiguous()
    masks = rearrange(masks, "b v h w c -> (b v) c h w").contiguous()

    device = masks.device

    # Compute individual loss components
    losses = {}
    if "BCE" in loss_type:
        losses['BCE'] = nn.BCEWithLogitsLoss()(masks, target_masks)
    if "Dice" in loss_type:
        intersection = (masks * target_masks).sum()
        union = masks.sum() + target_masks.sum()
        epsilon = 1e-5
        losses['Dice'] = 1 - (2 * intersection + epsilon) / (union + epsilon)  # +epsilon to avoid division by zero

    # Weighted sum
    total_loss = torch.tensor(0.0, device=device)
    if 'BCE' in losses:
        total_loss += lambda_BCE * losses['BCE']
    if 'Dice' in losses:
        total_loss += lambda_Dice * losses['Dice']

    return total_loss, {k: v.item() for k, v in losses.items()}


def foreground_region_loss(rendered_images, target_masks, loss_type="Dice+BCE", lambda_BCE=0.5, lambda_Dice=0.5):
    """
    Combined loss function for foreground regions

    Args:
        rendered_images: predicted images [B*V, C, H, W]
        target_masks: dictionary containing target images, expects "mask" key. target["masks"]: [B, V, C, H, W]
        loss_type: loss combination method ("Dice", "BCE", "Dice+BCE")

    Returns:
        combined loss value
    """

    target_masks = rearrange(target_masks, "b v c h w -> (b v) c h w").contiguous()
    N, C, H, W = rendered_images.shape
    target_masks = F.interpolate(target_masks, size=(H, W), mode='bilinear', align_corners=True)

    intensity = rendered_images.abs().mean(dim=1, keepdim=True)

    foreground_prob = torch.sigmoid((1 - intensity) * 1000) * 2 - 1  # Use sigmoid to approximate a step function

    device = rendered_images.device

    # Compute individual loss components
    losses = {}
    if "BCE" in loss_type:
        losses['BCE'] = nn.BCEWithLogitsLoss()(foreground_prob, target_masks)
    if "Dice" in loss_type:
        intersection = (foreground_prob * target_masks).sum()
        union = foreground_prob.sum() + target_masks.sum()
        epsilon = 1e-5
        losses['Dice'] = 1 - (2 * intersection + epsilon) / (union + epsilon)  # +epsilon to avoid division by zero

    # Weighted sum
    total_loss = torch.tensor(0.0, device=device)
    if 'BCE' in losses:
        total_loss += lambda_BCE * losses['BCE']
    if 'Dice' in losses:
        total_loss += lambda_Dice * losses['Dice']

    return total_loss, {k: v.item() for k, v in losses.items()}


def depth_loss(depth, depth_conf, batch, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False,
               affine_inv=False, gradient_loss=None, valid_range=-1, disable_conf=False, all_mean=False, **kwargs):
    gt_depth = batch['depths'].clone()
    valid_mask = batch['point_masks']

    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")

    gt_depth = gt_depth[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                                   batch, normalize_pred=False, normalize_gt=False,
                                   gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss,
                                   valid_range=valid_range, postfix="_depth", disable_conf=disable_conf,
                                   all_mean=all_mean)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, batch, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False,
               gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False,
               conf_loss_type="v1", **kwargs):
    """
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    """
    # gt_pts3d: B, S, H, W, 3
    gt_pts3d = batch['world_points']
    # valid_mask: B, S, H, W
    valid_mask = batch['point_masks']
    gt_pts3d = check_and_fix_inf_nan(gt_pts3d, "gt_pts3d")

    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")

    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask,
                                  batch, normalize_pred=normalize_pred, gamma=gamma, alpha=alpha, affine_inv=affine_inv,
                                  gradient_loss=gradient_loss, valid_range=valid_range,
                                  camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean)

    return conf_loss_dict


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        # too small, just return
        return loss_tensor

    # Randomly sample if tensor is too large
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch, normalize_gt=True, normalize_pred=True, gamma=1.0,
              alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1,
              disable_conf=False, all_mean=False, postfix=""):
    # normalize
    if normalize_gt:
        gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask)

    if normalize_pred:
        pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask)

    if affine_inv:
        scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames = reg_loss(pts3d,
                                                                                                          gt_pts3d,
                                                                                                          valid_mask,
                                                                                                          gradient_loss=gradient_loss)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(
            first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[
            other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])

    if conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        if valid_range > 0:
            conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
            conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

        conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
        conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")
    else:
        conf_loss_first_frame = pts3d * 0
        conf_loss_other_frames = pts3d * 0
        print("No valid conf loss", batch["seq_name"])

    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames

    # Verified that the loss is the same

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    if gradient_loss is not None:
        # loss_grad_first_frame and loss_grad_other_frames are already meaned
        loss_grad = loss_grad_first_frame + loss_grad_other_frames
        loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
        loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
        loss_dict[f"loss_grad{postfix}"] = loss_grad

    return loss_dict


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None):
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask],
                                      dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask],
                                       dim=-1)

    if gradient_loss == "grad":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww))
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww))
    elif gradient_loss == "grad_impl2":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=gradient_loss_impl2)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=gradient_loss_impl2)
    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                          first_frame_mask.reshape(bb * ss, hh, ww),
                                                          gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_gt_pts3d.reshape(bb * ss, hh, ww, nc),
                                                           other_frames_mask.reshape(bb * ss, hh, ww),
                                                           gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0  # hard coded
            alpha = 0.2  # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]  # B,H,W,3
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        # Direction vectors
        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir, left_dir, dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir, dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir, up_dir, dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids


def downsample_pointcloud(points, target_num, method='random'):
    """
    Point cloud downsampling

    Args:
        points: [B, N, 3] input point cloud
        target_num: target number of points
        method: 'random', 'farthest_point' or 'voxel'

    Returns:
        downsampled point cloud [B, K, 3] where K=target_num
    """
    B, N, D = points.shape

    # Random sampling (fastest)
    if method == 'random':
        indices = torch.randperm(N)[:target_num]
        return points[:, indices]

    # Farthest point sampling (preserves point cloud distribution)
    elif method == 'farthest_point':
        downsampled = []
        for i in range(B):
            samples = []
            remaining = points[i].clone()

            # Randomly select the first point
            first_idx = torch.randint(0, N, (1,))
            samples.append(points[i, first_idx])
            remaining = torch.cat([points[i, :first_idx], points[i, first_idx + 1:]], dim=0)

            # Iteratively select farthest points
            for _ in range(1, target_num):
                dists = torch.cdist(samples[-1:], remaining)[0]  # [1, M]
                max_idx = torch.argmax(dists)
                samples.append(remaining[max_idx])
                remaining = torch.cat([remaining[:max_idx], remaining[max_idx + 1:]], dim=0)

            downsampled.append(torch.stack(samples))

        return torch.stack(downsampled)

    # Voxel sampling (most uniform)
    elif method == 'voxel':
        import numpy as np

        voxel_size = 0.05  # Define voxel size
        downsampled_points = []

        for i in range(B):
            # Convert points to NumPy array
            points_np = points[i].cpu().numpy()

            # Compute voxel grid indices
            voxel_indices = np.floor(points_np / voxel_size).astype(int)

            # Use unique voxel indices to downsample
            _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
            downsampled_points.append(points_np[unique_indices[:target_num]])

        downsampled_points = torch.tensor(downsampled_points, device=points.device)
        return downsampled_points

    else:
        raise ValueError(f"Unsupported sampling method: {method}")


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn=gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def torch_quantile(
        input: torch.Tensor,
        q: float | torch.Tensor,
        dim: int | None = None,
        keepdim: bool = False,
        *,
        interpolation: str = "nearest",
        out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out
