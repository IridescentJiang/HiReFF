import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024):
        super().__init__()
        self.output_dim = embed_dim

        # 高效CNN架构
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images):
        # images: (B, S, C, H, W) -> (B*S, C, H, W)
        B, S, C, H, W = images.shape
        x = images.reshape(B * S, C, H, W)

        # 提取特征 (B*S, embed_dim, H', W')
        x = self.net(x)
        return x


class FeatureFuser(nn.Module):
    def __init__(self, embed_dim, rope_freq=100):
        super().__init__()
        self.embed_dim = embed_dim

        # 位置嵌入组件
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # 自适应池化以适应不同尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, tokens, cnn_features, images, patch_size):
        """
        tokens: list of tensors each (B, S, P, 2C)
        cnn_features: (B*S, embed_dim, H', W')
        images: 原始图像 (B, S, C, H, W)
        patch_size: 来自Aggregator的patch_size
        """
        B, S, C, H, W = images.shape
        P = tokens[0].shape[2]  # 总token数
        num_patches = (H // patch_size) * (W // patch_size)

        # 计算patch_start_idx (从您的Aggregator代码中获取)
        num_register_tokens = self.embed_dim // 256  # 假设每个register token 256维
        patch_start_idx = 1 + num_register_tokens

        # 调整CNN特征尺寸以匹配patch网格
        H_patch = H // patch_size
        W_patch = W // patch_size
        cnn_features = F.interpolate(
            cnn_features,
            size=(H_patch, W_patch),
            mode='bilinear',
            align_corners=False
        )

        # 将CNN特征转换为token格式 (B*S, H_patch*W_patch, embed_dim)
        cnn_features = cnn_features.flatten(2).permute(0, 2, 1)
        cnn_features = cnn_features.reshape(B, S, H_patch * W_patch, self.embed_dim)

        # 准备位置嵌入 (如果需要)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H_patch, W_patch, device=images.device)
            # 为特殊token添加零位置
            pos_special = torch.zeros(B * S, patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # 对每个attention block的输出进行融合
        fused_tokens = []
        for token_block in tokens:
            # token_block: (B, S, P, 2C)
            # 分离frame和global特征
            frame_features = token_block[..., :self.embed_dim]
            global_features = token_block[..., self.embed_dim:]

            # 为frame特征添加CNN信息
            frame_fused = self._fuse_single_stream(
                frame_features, cnn_features, patch_start_idx, num_patches, pos
            )

            # 为global特征添加CNN信息
            global_fused = self._fuse_single_stream(
                global_features, cnn_features, patch_start_idx, num_patches, pos
            )

            # 重新拼接
            fused_block = torch.cat([frame_fused, global_fused], dim=-1)
            fused_tokens.append(fused_block)

        return fused_tokens

    def _fuse_single_stream(self, tokens, cnn_features, patch_start_idx, num_patches, pos=None):
        """
        融合单个特征流 (frame或global)
        tokens: (B, S, P, embed_dim)
        cnn_features: (B, S, num_patches, embed_dim)
        """
        B, S, P, C = tokens.shape

        # 仅融合图像patch部分 (排除特殊token)
        patch_tokens = tokens[:, :, patch_start_idx:patch_start_idx + num_patches, :]

        # 确保CNN特征与patch token形状匹配
        if cnn_features.shape[2] != num_patches:
            cnn_features = F.interpolate(
                cnn_features.permute(0, 1, 3, 2),
                size=num_patches,
                mode='linear'
            ).permute(0, 1, 3, 2)

        # 门控融合
        combined = torch.cat([patch_tokens, cnn_features], dim=-1)
        fusion_gate = self.gate(combined)

        # 融合特征: (1-g)*tokens + g*cnn_features
        fused_patches = (1 - fusion_gate) * patch_tokens + fusion_gate * cnn_features

        # 重建完整token集 (特殊token + 融合后的patch token)
        special_tokens = tokens[:, :, :patch_start_idx]
        return torch.cat([special_tokens, fused_patches], dim=2)