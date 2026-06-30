# NPZ Data Format & Preprocessing

HiReFF training and inference use `.npz` files to store multi-view data.
This document specifies the NPZ format and how to use the preprocessing scripts
in `preprocessing/` to convert from common human datasets.

## Quick Start

Preprocessing scripts are provided for three datasets under `preprocessing/`:

| Dataset | Script | Input Format |
|---|---|---|
| DNA-Rendering | `preprocessing/dna/smc2npz.py` | `.smc` (proprietary HDF5) |
| ZJU-MoCap | `preprocessing/zju/zju2npz.py` | Images + `annots.json` |
| MVHuman | `preprocessing/mvh/mvh2npz.py` | Images + masks + `camera.npz` |

Each directory has a `README.md` with detailed usage instructions.

## NPZ File Structure

Each NPZ file represents one frame (timestep) of a multi-view capture:

```
frame_0000.npz
  ├── view_00  → Python dict
  ├── view_01
  ├── view_02
  └── ...
```

### Per-View Dictionary Keys

| Key | Shape | Dtype | Required | Description |
|---|---|---|---|---|
| `image` | (N,) | uint8 | Yes | JPEG-encoded RGB image bytes |
| `intrinsic` | (3, 3) | float32 | Yes | Camera intrinsic matrix |
| `extrinsic` | (4, 4) | float32 | Yes | Camera extrinsic matrix (camera-to-world) |
| `mask` | (M,) | uint8 | Train only | PNG-encoded grayscale foreground mask (white = foreground) |

### Intrinsic Matrix

```
[[fx,  0, cx],
 [ 0, fy, cy],
 [ 0,  0,  1]]
```

### Extrinsic Matrix

4×4 camera-to-world transformation. The training pipeline inverts this to world-to-camera internally.

## Directory Structure for Training

```
data_root/
  ├── dna-rendering/
  │   ├── 0001_01/           # Subject directory (DNA: ^\d{4}_\d{2}$)
  │   │   ├── frame_000000.npz
  │   │   └── ...
  │   └── ...
  ├── zju-mocap/
  │   ├── CoreView_313/      # Subject directory (ZJU: ^CoreView_\d+$)
  │   │   ├── frame_000000.npz
  │   │   └── ...
  │   └── ...
  └── mvhuman/
      ├── 101286/             # Subject directory
      │   ├── frame_000000.npz
      │   └── ...
      └── ...
```

### Dataset-Specific View Counts

| Dataset | Views per frame | View Naming |
|---|---|---|
| DNA-Rendering | 48 | `view_00` to `view_47` |
| ZJU-MoCap | 23 | `view_00` to `view_22` |
| MVHuman | 16 | `view_00` to `view_15` |

## Custom Dataset Conversion

For datasets not covered by the provided scripts, use the following pattern to
create NPZ files:

```python
import io
import numpy as np
from PIL import Image


def images_to_npz(output_path, images, intrinsics, extrinsics, masks=None):
    """Pack multi-view data into HiReFF NPZ format.

    Args:
        output_path: Path to the output .npz file.
        images: Dict[int, bytes] — view_id → JPEG-encoded image bytes.
        intrinsics: Dict[int, np.ndarray] — view_id → (3,3) float32.
        extrinsics: Dict[int, np.ndarray] — view_id → (4,4) float32 c2w.
        masks: Optional Dict[int, np.ndarray] — view_id → (H,W) uint8.
    """
    views = {}
    for view_id in sorted(images):
        entry = {
            "image": np.frombuffer(images[view_id], dtype=np.uint8),
            "intrinsic": intrinsics[view_id].astype(np.float32),
            "extrinsic": extrinsics[view_id].astype(np.float32),
        }
        if masks and view_id in masks:
            mask = masks[view_id]
            buf = io.BytesIO()
            Image.fromarray(mask).save(buf, format="PNG")
            entry["mask"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        views[f"view_{view_id:02d}"] = entry

    np.savez_compressed(output_path, **views)
```
