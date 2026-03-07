# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from merging.merge import (
    token_merge_bipartite2d,
)

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        patch_width: int = 37,
        patch_height: int = 37,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.patch_width = patch_width
        self.patch_height = patch_height

    def forward(self, x: Tensor, pos=None, global_merging=None) -> Tensor:
        merge_num = list(range(24))
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:

            if global_merging is not None and global_merging in merge_num:
                generator = torch.Generator(device=x.device)
                generator.manual_seed(33)

                merge_ratio = 0.1
                r = int(x.shape[1] * merge_ratio)

                m, u = token_merge_bipartite2d(
                    x,
                    self.patch_width,
                    self.patch_height,
                    2,
                    2,
                    r,
                    False,
                    generator,
                    enable_protection=True,
                )

                m_a, u_a = (m, u)

                B_q, H_q, N_q, D_q = q.shape

                q_merge_in = q.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                k_merge_in = k.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)
                v_merge_in = v.permute(0, 2, 1, 3).reshape(B_q, N_q, H_q * D_q)

                q_out, k_out, v_out = m_a(
                    q_merge_in,
                    mode="mean",
                    extra_tensors=k_merge_in,
                    extra_tensors_2=v_merge_in,
                )

                del q_merge_in, k_merge_in, v_merge_in

                N_m = q_out.shape[1]
                q = q_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                k = k_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)
                v = v_out.reshape(B_q, N_m, H_q, D_q).permute(0, 2, 1, 3)

                del q_out, k_out, v_out

                N = N_m

            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if global_merging is not None and global_merging in merge_num:
            x = u_a(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, global_merging=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
