# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from hireff.heads.dpt_head import DPTHead
from timm.models.layers import trunc_normal_
from .edgenext.edgenext import EdgeNeXt
from .edgenext.layers import LayerNorm


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.relu = nn.GELU()

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_planes//8)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_planes == planes):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_planes == planes:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class HiReFF_DPT_GS_Head(DPTHead):
    def __init__(self, 
            dim_in: int,
            patch_size: tuple[int, int] = (14, 14),
            output_dim: int = 83,
            activation: str = "inv_log",
            conf_activation: str = "expp1",
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            feature_only: bool = False,
            down_ratio: int = 1,
            img_size: int = 518,
    ):
        super().__init__(dim_in, patch_size, output_dim, activation, conf_activation, features, out_channels, intermediate_layer_idx, pos_embed, feature_only, down_ratio)
        
        self.hr_feat = EdgeNeXt(
            levels=2, depths=[2, 4], dims=[32, 64], expan_ratio=4,
            global_block=[0, 1, ],
            global_block_type=['None', 'SDTA', ],
            use_pos_embd_xca=[False, True, ],
            kernel_sizes=[3, 3, ],
            heads=[4, 4, ],
            d2_scales=[2, 2, ],
        )
        
        self.feature_fuse0 = ResidualBlock(64 + 128, 128, 'instance')
        self.feature_fuse1 = ResidualBlock(64 + 32, 64, 'instance')
        
        self.upsample0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False, ), 
            ResidualBlock(128, 64, 'instance'), 
            ResidualBlock(64, 64, 'instance'), 
        )
        
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False, ), 
            ResidualBlock(64, 32, 'instance'), 
        )
        
        self.rot_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.scale_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.scale_head[-1].bias, -0.02)
        nn.init.trunc_normal_(self.opacity_head[-1].weight, mean=0.0, std=1e-2)
        nn.init.constant_(self.opacity_head[-1].bias, -2.0)
    
    def _init_weights(self, m):  # TODO: MobileViT is using 'kaiming_normal' for initializing conv layers
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, encoder_tokens: List[torch.Tensor], imgs, ds_imgs, patch_start_idx: int = 5, image_size=None, conf=None, frames_chunk_size: int = 8):
        # H, W = input_info['image_size']
        B, S, _, H, W = imgs.shape
        image_size = self.image_size if image_size is None else image_size
    
        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(encoder_tokens, imgs, ds_imgs, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            chunk_output = self._forward_impl(
                encoder_tokens, imgs, ds_imgs, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
        
        # Concatenate results along the sequence dimension
        return torch.cat(all_preds, dim=1)
    
    def _forward_impl(self, encoder_tokens: List[torch.Tensor], imgs, ds_imgs, patch_start_idx: int = 5, frames_start_idx: int = None, frames_end_idx: int = None):
        
        if frames_start_idx is not None and frames_end_idx is not None:
            imgs = imgs[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = imgs.shape
        _, _, _, ds_H, ds_W = ds_imgs.shape

        patch_h, patch_w = ds_H // self.patch_size[0], ds_W // self.patch_size[1]

        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            # x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            if len(encoder_tokens) > 10:
                x = encoder_tokens[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = encoder_tokens[list_idx][:, :, patch_start_idx:]
            
            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()
            
            x = x.view(B * S, -1, x.shape[-1])

            x = self.norm(x)
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, ds_W, ds_H)
            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        
        direct_img_feat = self.hr_feat.forward_features_multi_res(imgs.flatten(0, 1))
        
        rs_h, rs_w = direct_img_feat[-1].shape[-2:]
        out = F.interpolate(out, size=(rs_h, rs_w), mode='bilinear', align_corners=True)

        if out.shape[-2:] != direct_img_feat[1].shape[-2:]:
            out = F.interpolate(out, size=direct_img_feat[1].shape[-2:], mode='bilinear', align_corners=False)
        
        fused = self.feature_fuse0(torch.cat([out, direct_img_feat[1]], dim=1))
        fused = self.upsample0(fused)

        if direct_img_feat[0].shape[-2:] != fused.shape[-2:]:
            skip0 = F.interpolate(direct_img_feat[0], size=fused.shape[-2:], mode='bilinear', align_corners=False)
        else:
            skip0 = direct_img_feat[0]
        
        fused = self.feature_fuse1(torch.cat([fused, skip0], dim=1))
        fused = self.upsample1(fused)

        color_out = self.color_head(fused)

        # rot head
        rot_out = self.rot_head(fused)

        # scale head
        scale_out = self.scale_head(fused)

        # opacity head
        opacity_out = self.opacity_head(fused)

        out = rearrange(
            torch.cat([opacity_out, scale_out, rot_out, color_out], dim=1), 
            "(b v) ... -> b v ...", 
            b=B
        )
        
        return out