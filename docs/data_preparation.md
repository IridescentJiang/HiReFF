# NPZ Data Format Specification

HiReFF training and inference use `.npz` files to store multi-view data.
This document specifies the format and provides conversion examples.

## File Structure

Each NPZ file represents one frame (timestep) of a multi-view capture:

```
frame_0000.npz
  ├── view_00  → Python dict
  ├── view_01
  ├── view_02
  └── ...
```

## Per-View Dictionary Keys

| Key | Shape | Dtype | Required | Description |
|---|---|---|---|---|
| `image` | (N,) | uint8 | Yes | JPEG-encoded RGB image bytes |
| `intrinsic` | (3, 3) | float32 | Yes | Camera intrinsic matrix |
| `extrinsic` | (4, 4) | float32 | Yes | Camera extrinsic matrix (camera-to-world) |
| `mask` | (M,) | uint8 | Train only | PNG-encoded grayscale foreground mask (white = foreground) |

### Intrinsic Matrix

Standard pinhole camera intrinsics:

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

where `fx`, `fy` are focal lengths in pixels and `(cx, cy)` is the principal point.

### Extrinsic Matrix

4×4 camera-to-world transformation matrix:

```
[[R_00, R_01, R_02, tx],
 [R_10, R_11, R_12, ty],
 [R_20, R_21, R_22, tz],
 [   0,    0,    0,  1]]
```

> **Note:** The training pipeline inverts this to world-to-camera internally.

## Directory Structure for Training

```
data_root/
  ├── dna-rendering/
  │   ├── 0001_01/           # Subject directory (pattern: ^\d{4}_\d{2}$ for DNA)
  │   │   ├── frame_000000.npz
  │   │   ├── frame_000001.npz
  │   │   └── ...
  │   └── 0002_03/
  │       └── ...
  ├── zju-mocap/
  │   ├── CoreView_313/      # Subject directory (pattern: ^CoreView_\d+$ for ZJU)
  │   │   ├── frame_000000.npz
  │   │   └── ...
  │   └── CoreView_315/
  │       └── ...
  └── mvhuman/
      ├── subject_001/       # Any directory containing .npz files
      │   ├── frame_000000.npz
      │   └── ...
      └── ...
```

### Dataset-Specific View Counts

| Dataset | Views per frame | View Naming |
|---|---|---|
| DNA-Rendering | 48 | `view_00` to `view_47` |
| ZJU-MoCap | 24 | `view_00` to `view_23` |
| MVHuman | 16 | `view_00` to `view_15` |

## Conversion Script Example

```python
"""Convert a directory of multi-view images to HiReFF NPZ format."""

import io
import json
import os
import numpy as np
from PIL import Image


def images_to_npz(
    frame_dir: str,
    output_path: str,
    intrinsics: dict[int, np.ndarray],   # view_id -> 3x3 float32
    extrinsics: dict[int, np.ndarray],   # view_id -> 4x4 float32 (c2w)
    masks: dict[int, np.ndarray] | None = None,  # view_id -> HxW uint8
):
    """Pack a multi-view frame into HiReFF NPZ format.

    Args:
        frame_dir: Directory containing view_XX.png (or .jpg) files.
        output_path: Path to the output .npz file.
        intrinsics: Dict mapping view id to 3x3 intrinsic matrix.
        extrinsics: Dict mapping view id to 4x4 extrinsic matrix (c2w).
        masks: Optional dict mapping view id to HxW uint8 mask (255=fg).
    """
    views = {}
    for view_id in sorted(intrinsics.keys()):
        # Read image
        img_path = os.path.join(frame_dir, f"view_{view_id:02d}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(frame_dir, f"view_{view_id:02d}.jpg")
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # Read or create mask
        mask_bytes = None
        if masks is not None and view_id in masks:
            mask = masks[view_id]
            mask_img = Image.fromarray(mask)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_bytes = buf.getvalue()

        entry = {
            "image": np.frombuffer(image_bytes, dtype=np.uint8),
            "intrinsic": intrinsics[view_id].astype(np.float32),
            "extrinsic": extrinsics[view_id].astype(np.float32),
        }
        if mask_bytes is not None:
            entry["mask"] = np.frombuffer(mask_bytes, dtype=np.uint8)

        views[f"view_{view_id:02d}"] = entry

    np.savez_compressed(output_path, **views)
    print(f"Saved {output_path} with {len(views)} views")


# Example usage
if __name__ == "__main__":
    # Assume you have:
    #   ./my_data/frame_0000/view_00.png ... view_47.png
    #   ./my_data/cameras.json  (contains intrinsics and extrinsics)

    with open("./my_data/cameras.json") as f:
        cam_data = json.load(f)

    intrinsics = {i: np.array(c["intrinsic"]) for i, c in enumerate(cam_data)}
    extrinsics = {i: np.array(c["extrinsic"]) for i, c in enumerate(cam_data)}

    images_to_npz(
        frame_dir="./my_data/frame_0000",
        output_path="./my_data/frame_0000.npz",
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )
```
