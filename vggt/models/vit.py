from torchvision.models.vision_transformer import VisionTransformer
import torch.nn as nn
import torch


def create_small_vit_s(output_dim=128, patch_size=14, img_size=518):
    """
    Create a lightweight ViT-S model.

    Args:
        output_dim: Output feature dimension.
        patch_size: Patch size for input images.
        img_size: Input image size.
    """
    # Compute number of patches (e.g., 256/16 = 16 → 16x16 = 256 patches)
    num_patches = (img_size // patch_size) ** 2

    # Small ViT-S configuration
    vit_config = {
        'image_size': img_size,
        'patch_size': patch_size,
        'num_layers': 6,         # fewer layers for a lightweight model
        'num_heads': 8,          # fewer attention heads
        'hidden_dim': 384,       # smaller hidden dimension
        'mlp_dim': 1536,         # typically 4x hidden_dim
        'num_classes': output_dim,
        'dropout': 0.1,
        'attention_dropout': 0.1,
    }

    model = VisionTransformer(**vit_config)

    # Replace classification head with a linear projection to output_dim
    # The output shape will be (B, 8, 256)
    model.heads = nn.Sequential(
        nn.Linear(vit_config['hidden_dim'], vit_config['hidden_dim']),
        nn.GELU(),
        nn.Linear(vit_config['hidden_dim'], output_dim)
    )

    # Print parameter counts
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Small ViT-S Total parameters: {total_params:,}")
    # print(f"Small ViT-S Trainable parameters: {trainable_params:,}")

    # Define custom forward to return patch-level features
    def forward_custom(x):
        # Extract features via ViT
        x = model._process_input(x)
        n = x.shape[1]

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = model.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Pass through Transformer encoder
        x = model.encoder(x)

        # Remove class token, keep patch tokens only
        x = x[:, 1:, :]  # shape: (B, 256, 384)

        # Apply head projection
        x = model.heads(x)  # shape: (B, 256, 8)

        # Transpose to (B, 8, 256)
        return x.transpose(1, 2)

    model.forward = forward_custom
    return model