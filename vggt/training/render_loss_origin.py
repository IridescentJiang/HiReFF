from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vggt.training.lpips.lpips import LPIPS
from einops import repeat

    
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
        
        resized_pred = repeat(_tar[0, :, 0, 0], "c -> 1 c h w", h=l_side, w=l_side).clone()
        resized_pred[..., pad[2]:pad[2]+_pred.shape[2], pad[0]:pad[0]+_pred.shape[3]] = _pred
        
        resized_pred = F.interpolate(
            resized_pred, 
            size=(compute_size, compute_size), mode="bilinear", align_corners=False
        )[0]
        
        cropped_pred.append(resized_pred)
        cropped_tar.append(resized_tar)
    
    cropped_pred = torch.stack(cropped_pred, dim=0)
    cropped_tar = torch.stack(cropped_tar, dim=0)
    
    return loss_fn(cropped_pred, cropped_tar)


DINOV3_MODEL_FAMILY = [
    'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16',
]
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

    def forward(self, images, masks, target_images):
        target_images = rearrange(target_images, "b v c h w -> (b v) c h w")
        masks = rearrange(masks, "b v c h w -> (b v) c h w")

        if self.mask_weight_factor > 1.0:
            base_weights = torch.ones_like(masks)
            base_weights[masks > 0.5] = self.mask_weight_factor

            if self.edge_weight_factor > self.mask_weight_factor:
                with torch.no_grad():
                    dilated = F.conv2d(masks, self.edge_kernel, padding=1)
                    edges = (dilated > 0.2) & (dilated < 0.8)
                    base_weights[edges] = self.mask_weight_factor
            weights = base_weights
        else:
            weights = None

        losses = {}

        l1_loss_map = F.l1_loss(images, target_images, reduction="none")
        weighted_l1 = (l1_loss_map * weights).mean()
        losses['l1'] = weighted_l1.item()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss_lpips = crop_and_resize_loss(self.lpips, images, target_images, masks, compute_size=768, border=16).mean()
        losses['lpips'] = loss_lpips.item()
        
        total_loss = loss_lpips * self.lambda_perceptual + weighted_l1 * self.lambda_l1
        
        return total_loss, losses
