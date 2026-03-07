# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch
import torch.nn as nn
from einops import rearrange
import copy
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.vggt_dpt_gs_head import VGGT_DPT_GS_Head
from vggt.heads.gs_adaptor import process_gs_map
from vggt.utils.depth2points import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.rendering.render_image import adjust_transl
from vggt.utils.interpolate import interpolate_images


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator_img_size = img_size
        self.sh_degree = 0
        self.d_sh = (self.sh_degree + 1) ** 2
        self.opacity_ch = 1
        self.scale_ch = 3
        self.rotate_ch = 4
        self.color_ch = 3 * self.d_sh
        self.gs_para_ch = self.opacity_ch + self.scale_ch + self.rotate_ch + self.color_ch

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type="absT_quaR_FoV")
        self.point_head = None  # DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.activate_depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        
        self.gs_para_head = VGGT_DPT_GS_Head(
            dim_in=2048,
            patch_size=(14, 14),
            output_dim=self.gs_para_ch,
            activation="norm_exp",
            conf_activation="expp1",
            features=256,
            img_size=img_size
        )
        self.mask_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="sigmoid", conf_activation="expp1")

    def forward(
            self,
            images: torch.Tensor, 
            images_hr: torch.Tensor,
            mask_gaussian: bool = True, 
            gt_masks: torch.Tensor | None = None, 
            use_gt_mask: bool = False, 
            gt_extrinsic: torch.Tensor | None = None,
            gt_intrinsic: torch.Tensor | None = None,
            if_train=True,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, V, C, H, W = images.shape

        predictions = {}

        with torch.amp.autocast(device_type="cuda", enabled=True):

            # start_time = time.time_ns() // 1_000_000
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
            # end_time = time.time_ns() // 1_000_000
            # print(f"Processed aggregator in {end_time - start_time:.2f} ms")

            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc_pre"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                if if_train:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = interpolate_images(depth, target_size=(H, W), mode='bilinear', align_corners=True)
                    depth_conf = interpolate_images(depth_conf, target_size=(H, W), mode='bilinear', align_corners=True)
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                    predictions["pseudo_label_depth"] = depth.detach()

                if self.activate_depth_head is not None:
                    activate_depth, activate_depth_conf = self.activate_depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                    )
                    activate_depth = interpolate_images(activate_depth, target_size=(H, W), mode='bilinear', align_corners=True)
                    activate_depth_conf = interpolate_images(activate_depth_conf, target_size=(H, W), mode='bilinear', align_corners=True)
                    # predictions["pseudo_label_depth_conf"] = predictions["world_depth_conf"]
                    predictions["depth"] = activate_depth
                    predictions["depth_conf"] = activate_depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                # predictions["world_points_conf"] = pts3d_conf
            else:
                depth_map = predictions["depth"]  # (B, V, H, W, 1)
                depth_map = rearrange(depth_map, "B V H W C -> (B V) H W C")
                if gt_intrinsic is not None and gt_extrinsic is not None:
                    adjusted_extrinsics, _ = adjust_transl(predictions["pose_enc_pre"], gt_extrinsic, None)
                    adjusted_extrinsics = rearrange(adjusted_extrinsics, "B V H W -> (B V) H W")
                    gt_intrinsic = rearrange(gt_intrinsic, "B V H W -> (B V) H W")
                    world_points = unproject_depth_map_to_point_map(depth_map, adjusted_extrinsics, gt_intrinsic)
                else:
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc_pre"], images.shape[-2:])
                    extrinsic = rearrange(extrinsic, "B V H W -> (B V) H W")
                    intrinsic = rearrange(intrinsic, "B V H W -> (B V) H W")
                    world_points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
                world_points = rearrange(world_points, "(B V) H W C -> B V H W C", B=images.shape[0])
                predictions["world_points"] = world_points

            if self.mask_head is not None:
                masks, masks_conf = self.mask_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx,
                )
                masks = interpolate_images(masks, target_size=(H, W), mode='bilinear', align_corners=True)
                predictions["masks"] = masks
                predictions["masks_conf"] = masks_conf

            if self.gs_para_head is not None:
                b, v, _, h, w = images.shape
                out = self.gs_para_head(
                    aggregated_tokens_list,
                    images_hr,
                    images,
                    patch_start_idx=patch_start_idx,
                    image_size=(h, w),
                )
                out = out.float().permute(0, 1, 3, 4, 2).contiguous()  # (b, v, h, w, gs_para_ch)
                
                gs_map_h, gs_map_w = out.shape[2:4]
                
                if mask_gaussian:
                    if gt_masks is not None and use_gt_mask:
                        gs_mask = torch.nn.functional.interpolate(gt_masks.flatten(0, 1), (gs_map_h, gs_map_w), mode="bilinear", align_corners=False)
                        gs_mask = rearrange(gs_mask, "(b v) c h w -> b v c h w", b=B)
                    else:
                        gs_mask = rearrange(predictions["masks"], "b v h w c -> (b v) c h w")
                        gs_mask = torch.nn.functional.interpolate(gs_mask, (gs_map_h, gs_map_w), mode="bilinear", align_corners=False)
                        gs_mask = rearrange(gs_mask, "(b v) c h w -> b v c h w", b=B)
                else:
                    gs_mask = None
                    
                gs_pos = interpolate_images(predictions["world_points"], (gs_map_h, gs_map_w), align_corners=False)

                predictions["flat_gs"] = process_gs_map(out, gs_pos, gs_mask, images_hr)

        predictions["images"] = images

        return predictions

    @classmethod
    def from_checkpoint(cls, pretrained_path, map_location="cpu"):
        """
        从本地文件加载预训练参数

        Args:
            pretrained_path: 本地模型文件路径
            map_location: 加载设备 [cpu/cuda]
        """
        model = cls()  # 初始化空模型

        # 加载checkpoint
        checkpoint = torch.load(pretrained_path, map_location=map_location)

        # 自动处理常见checkpoint格式
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        # 处理多GPU训练保存的参数名
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        # 加载参数
        model.load_state_dict(state_dict)
        model.eval()
        return model
