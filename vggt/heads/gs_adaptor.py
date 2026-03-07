import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class GaussianAdapter(nn.Module):
    def __init__(self, sh_degree, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_sh = (sh_degree + 1) ** 2
        self.sh_mask = None

        if sh_degree:
            n_sh_coeffs = (sh_degree + 1) ** 2
            self.sh_mask = torch.zeros(n_sh_coeffs)
            for degree in range(1, sh_degree + 1):
                self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree
        else:
            self.sh_mask = 1
    def forward(
            self,
            xyz,
            raw_gaussians,
            eps: float = 1e-8,
    ):
        opacities, scales, rotations, colors = raw_gaussians.split((1, 3, 4, 3 * self.d_sh), dim=-1)

        opacities = opacities.sigmoid()

        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        colors = rearrange(colors, "... (xyz d_sh) -> ... xyz d_sh", xyz=3).contiguous()
        colors = torch.sigmoid(colors)

        # covariances = build_covariance(scales, rotations)

        return {"xyz": xyz,
                "rotations": rotations,
                "colors": colors,
                "opacities": opacities,
                "scales": scales}
        
def process_gs_map(raw_map, pos, mask=None, gt_images=None):
    b, v, h, w, c = raw_map.shape
    sh_c = c - 8
    
    opacities, scales, rotations, colors = rearrange(raw_map, "b v h w c -> b (v h w) c").split((1, 3, 4, sh_c), dim=-1)  # b n c

    pcs = []
    fake_mask = torch.ones(h * w * v, dtype=torch.bool, device=raw_map.device) if mask is None else None
    for idx in range(b):
        _mask = mask[idx].flatten() > 0.5 if mask is not None else fake_mask
        
        _xyz = rearrange(pos[idx], "v h w c -> (v h w) c")[_mask]
        
        _opa = torch.sigmoid(opacities[idx][_mask])
        _sca = F.softplus(scales[idx][_mask], beta=100)
        _rot = torch.nn.functional.normalize(rotations[idx][_mask], dim=-1)
                
        _clr = rearrange(colors[idx][_mask], "n (xyz d_sh) -> n xyz d_sh", xyz=3).contiguous()
        _clr = torch.sigmoid(_clr)
        
        if gt_images is not None:
            _gt_clr = torch.nn.functional.interpolate(gt_images[idx], size=(h, w), mode="bilinear", align_corners=False)
            _gt_clr = rearrange(_gt_clr, "v c h w -> (v h w) c")[_mask]
        else:
            _gt_clr = None
        
        pcs.append({
            "xyz": _xyz, 
            "rotations": _rot, 
            "colors": _clr, 
            "opacities": _opa, 
            "scales": _sca, 
            "gt_colors": _gt_clr, 
        })

    return pcs


def build_covariance(
        scale,
        rotation_xyzw,
):
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
            rotation
            @ scale
            @ rearrange(scale, "... i j -> ... j i").contiguous()
            @ rearrange(rotation, "... i j -> ... j i").contiguous()
    )


def quaternion_to_matrix(
        quaternions,
        eps: float = 1e-8,
):
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3).contiguous()


def flatten_gaussians(gaussians_dict):
    """将多维高斯参数重组为扁平化字典格式"""

    # 重组各参数
    flattened = {
        "xyz": rearrange(
            gaussians_dict["xyz"],
            "b v h w xyz -> b (v h w) xyz"
        ).contiguous(),
        "rotations": rearrange(
            gaussians_dict["rotations"],
            "b v h w rotations -> b (v h w) rotations"
        ).contiguous(),
        "colors": rearrange(
            gaussians_dict["colors"],
            "b v h w c d_sh-> b (v h w) (c d_sh)"
        ).contiguous(),
        "opacities": rearrange(
            gaussians_dict["opacities"],
            "b v h w e-> b (v h w) e"
        ).contiguous(),
        "scales": rearrange(
            gaussians_dict["scales"],
            "b v h w scales -> b (v h w) scales"
        ).contiguous()
    }

    return flattened