from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange
from vggt.training.lpips.lpips import LPIPS

    
def crop_and_resize_loss(
    loss_fn: Callable, 
    pred: torch.Tensor,
    tar: torch.Tensor,
    mask: torch.Tensor,
    compute_size: int,
    border: int = 16,
):
    B, _, H, W = mask.shape

    # Normalize mask to [B, H, W] boolean
    valid = mask[:, 0] > 0.5 if mask.dim() == 4 else mask > 0.5

    cropped_pred = []
    cropped_tar = []
    
    for i in range(B):
        vi = valid[i]
        if vi.any():
            ys, xs = torch.nonzero(vi, as_tuple=True)
            y1 = int(ys.min().item())
            y2 = int(ys.max().item())
            x1 = int(xs.min().item())
            x2 = int(xs.max().item())
        else:
            continue

        y1p = max(0, y1 - border)
        x1p = max(0, x1 - border)
        y2p = min(H - 1, y2 + border)
        x2p = min(W - 1, x2 + border)
        
        l_side = max(x2p + 1 - x1p, y2p + 1 - y1p)
        pad_h = l_side - (y2p + 1 - y1p)
        pad_w = l_side - (x2p + 1 - x1p)
        pad = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]

        # Resize to out size        
        _tar = tar[i : i + 1, :, y1p : y2p + 1, x1p : x2p + 1]
        resized_tar = F.interpolate(
            F.pad(_tar, pad, mode="replicate"), 
            size=(compute_size, compute_size), mode="bilinear", align_corners=False
        )[0]
        
        _pred = pred[i : i + 1, :, y1p : y2p + 1, x1p : x2p + 1]
        resized_pred = F.interpolate(
            F.pad(_pred, pad, mode="replicate"),
            size=(compute_size, compute_size), mode="bilinear", align_corners=False
        )[0]
        
        cropped_pred.append(resized_pred)
        cropped_tar.append(resized_tar)
    
    if len(cropped_pred) == 0:
        return pred.new_zeros((), dtype=pred.dtype)

    cropped_pred = torch.stack(cropped_pred, dim=0)
    cropped_tar = torch.stack(cropped_tar, dim=0)
    
    return loss_fn(cropped_pred, cropped_tar)


DINOV3_MODEL_FAMILY = [
    'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16',
]


class GradientInjector(Function):
    @staticmethod
    def forward(ctx, x, grad_x):
        ctx.save_for_backward(grad_x)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, = ctx.saved_tensors
        return grad_output + grad_x, None


class PerceptualLossDINOv3(nn.Module):
    def __init__(self, repo_dir, ckpt_dir, arch='dinov3_vitb16'):
        """
        Perceptual Loss using a pre-trained DINOv3 model
        """
        assert arch in DINOV3_MODEL_FAMILY, f"Model architecture must be one of {DINOV3_MODEL_FAMILY}"
        super(PerceptualLossDINOv3, self).__init__()
        self.dino_model = torch.hub.load(repo_dir, arch, source='local', weights=ckpt_dir)
        self.dino_model.eval()  # Set to evaluation mode
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        _n_blocks = self.dino_model.n_blocks
        self._layers = [_n_blocks // 4, _n_blocks // 2, _n_blocks * 3 // 4, _n_blocks - 1]

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted images
            target: Ground truth images
            
        Returns:
            Perceptual loss
        """
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            feat_pred = self.dino_model.get_intermediate_layers(pred, n=self._layers)
            feat_target = self.dino_model.get_intermediate_layers(target, n=self._layers)
            
            loss = 0.0
            for p, t in zip(feat_pred, feat_target):
                loss += F.l1_loss(p, t)
        return loss.float()


class RenderLoss(nn.Module):
    def __init__(self, lambda_perceptual=0.1, lambda_l1=1.0, mask_weight_factor=2.0, edge_weight_factor=3.0, ):
        super().__init__()
        
        self.lambda_perceptual = lambda_perceptual
        self.lambda_l1 = lambda_l1
        self.mask_weight_factor = mask_weight_factor
        self.edge_weight_factor = edge_weight_factor
        
        self.register_buffer(
            "edge_kernel", 
            torch.tensor([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]], dtype=torch.float32).view(1, 1, 3, 3), 
            persistent=False,            
        )
        
        self.lpips = LPIPS(net='vgg')
        self.lpips.requires_grad_(False)
        # self.lpips = PerceptualLossDINOv3(
        #     repo_dir="/home/user/project/VGGT_Human/dinov3/", 
        #     ckpt_dir="/home/user/project/VGGT_Human/dinov3/models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 
        #     arch="dinov3_vitb16"
        # )

    def compute_lpips_loss_detach(self, images, target_images, masks, lambda_perceptual, compute_size=768, border=16):
        grad_container = torch.zeros_like(images)
        total_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
        valid_count = 0
        grad_clip_value = 1.0

        for i in range(images.shape[0]):
            mask_i = masks[i:i + 1]
            if not (mask_i > 0.5).any():
                continue

            image_i = images[i:i + 1].detach()
            image_i.requires_grad_(True)
            target_i = target_images[i:i + 1]

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss_i = crop_and_resize_loss(
                    self.lpips,
                    image_i,
                    target_i,
                    mask_i,
                    compute_size=compute_size,
                    border=border,
                ).mean()

            if not torch.isfinite(loss_i):
                del image_i, target_i, loss_i
                continue

            grad_i = torch.autograd.grad(loss_i, image_i, retain_graph=False, create_graph=False)[0]
            grad_i = torch.nan_to_num(grad_i, nan=0.0, posinf=grad_clip_value, neginf=-grad_clip_value)
            grad_i = grad_i.clamp(min=-grad_clip_value, max=grad_clip_value)
            grad_container[i:i + 1] = grad_i
            total_loss = total_loss + loss_i.detach().float()
            valid_count += 1

            del image_i, target_i, loss_i, grad_i

        if valid_count > 0:
            avg_loss = total_loss / valid_count
            grad_container = grad_container / float(valid_count)
        else:
            avg_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)

        avg_loss = torch.nan_to_num(avg_loss, nan=0.0, posinf=100.0, neginf=0.0)
        grad_container = torch.nan_to_num(grad_container, nan=0.0, posinf=grad_clip_value, neginf=-grad_clip_value)

        images_surgeried = GradientInjector.apply(images, grad_container * lambda_perceptual)
        return avg_loss, images_surgeried

    def forward(self, images, masks, target_images):
        target_images = rearrange(target_images, "b v c h w -> (b v) c h w")
        masks = rearrange(masks, "b v c h w -> (b v) c h w")

        pred_h, pred_w = images.shape[-2:]
        if target_images.shape[-2:] != (pred_h, pred_w):
            target_images = F.interpolate(
                target_images,
                size=(pred_h, pred_w),
                mode="area",
            )
        if masks.shape[-2:] != (pred_h, pred_w):
            masks = F.interpolate(
                masks,
                size=(pred_h, pred_w),
                mode="bilinear",
                align_corners=False,
            )

        if self.mask_weight_factor > 1.0:
            base_weights = torch.ones_like(masks)
            base_weights[masks > 0.5] = self.mask_weight_factor

            if self.edge_weight_factor > self.mask_weight_factor:
                with torch.no_grad():
                    dilated = F.conv2d(masks, self.edge_kernel, padding=1)
                    edges = (dilated > 0.2) & (dilated < 0.8)
                    base_weights[edges] = self.edge_weight_factor
            weights = base_weights
        else:
            weights = None

        losses = {}

        loss_lpips, images_surgeried = self.compute_lpips_loss_detach(
            images,
            target_images,
            masks,
            lambda_perceptual=self.lambda_perceptual,
            compute_size=768,
            border=16,
        )
        losses['lpips'] = loss_lpips.item()

        l1_loss_map = F.l1_loss(images_surgeried, target_images, reduction="none")
        if weights is not None:
            weighted_l1 = (l1_loss_map * weights).mean()
        else:
            weighted_l1 = l1_loss_map.mean()
        weighted_l1 = torch.nan_to_num(weighted_l1, nan=0.0, posinf=100.0, neginf=0.0)
        losses['l1'] = weighted_l1.item()

        total_loss = weighted_l1 * self.lambda_l1 + loss_lpips.detach() * self.lambda_perceptual
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=100.0, neginf=0.0)

        return total_loss, losses
