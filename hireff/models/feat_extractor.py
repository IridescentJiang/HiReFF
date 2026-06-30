import torch
import torch.nn as nn
import torch.nn.functional as F
from hireff.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, embed_dim=1024):
        super().__init__()
        self.output_dim = embed_dim

        # Efficient CNN architecture
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

        # Extract features (B*S, embed_dim, H', W')
        x = self.net(x)
        return x


class FeatureFuser(nn.Module):
    def __init__(self, embed_dim, rope_freq=100):
        super().__init__()
        self.embed_dim = embed_dim

        # Position embedding components
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Adaptive pooling for different sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, None))

        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, tokens, cnn_features, images, patch_size):
        """
        tokens: list of tensors each (B, S, P, 2C)
        cnn_features: (B*S, embed_dim, H', W')
        images: Original images (B, S, C, H, W)
        patch_size: patch_size from Aggregator
        """
        B, S, C, H, W = images.shape
        P = tokens[0].shape[2]  # Total number of tokens
        num_patches = (H // patch_size) * (W // patch_size)

        # Compute patch_start_idx (obtained from the Aggregator code)
        num_register_tokens = self.embed_dim // 256  # Assume each register token is 256-dim
        patch_start_idx = 1 + num_register_tokens

        # Resize CNN features to match patch grid
        H_patch = H // patch_size
        W_patch = W // patch_size
        cnn_features = F.interpolate(
            cnn_features,
            size=(H_patch, W_patch),
            mode='bilinear',
            align_corners=False
        )

        # Convert CNN features to token format (B*S, H_patch*W_patch, embed_dim)
        cnn_features = cnn_features.flatten(2).permute(0, 2, 1)
        cnn_features = cnn_features.reshape(B, S, H_patch * W_patch, self.embed_dim)

        # Prepare position embeddings (if needed)
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H_patch, W_patch, device=images.device)
            # Add zero positions for special tokens
            pos_special = torch.zeros(B * S, patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # Fuse each attention block output
        fused_tokens = []
        for token_block in tokens:
            # token_block: (B, S, P, 2C)
            # Separate frame and global features
            frame_features = token_block[..., :self.embed_dim]
            global_features = token_block[..., self.embed_dim:]

            # Add CNN information to frame features
            frame_fused = self._fuse_single_stream(
                frame_features, cnn_features, patch_start_idx, num_patches, pos
            )

            # Add CNN information to global features
            global_fused = self._fuse_single_stream(
                global_features, cnn_features, patch_start_idx, num_patches, pos
            )

            # Re-concatenate
            fused_block = torch.cat([frame_fused, global_fused], dim=-1)
            fused_tokens.append(fused_block)

        return fused_tokens

    def _fuse_single_stream(self, tokens, cnn_features, patch_start_idx, num_patches, pos=None):
        """
        Fuse a single feature stream (frame or global).
        tokens: (B, S, P, embed_dim)
        cnn_features: (B, S, num_patches, embed_dim)
        """
        B, S, P, C = tokens.shape

        # Only fuse image patch portion (exclude special tokens)
        patch_tokens = tokens[:, :, patch_start_idx:patch_start_idx + num_patches, :]

        # Ensure CNN features match patch token shape
        if cnn_features.shape[2] != num_patches:
            cnn_features = F.interpolate(
                cnn_features.permute(0, 1, 3, 2),
                size=num_patches,
                mode='linear'
            ).permute(0, 1, 3, 2)

        # Gated fusion
        combined = torch.cat([patch_tokens, cnn_features], dim=-1)
        fusion_gate = self.gate(combined)

        # Fused features: (1-g)*tokens + g*cnn_features
        fused_patches = (1 - fusion_gate) * patch_tokens + fusion_gate * cnn_features

        # Reconstruct complete token set (special tokens + fused patch tokens)
        special_tokens = tokens[:, :, :patch_start_idx]
        return torch.cat([special_tokens, fused_patches], dim=2)