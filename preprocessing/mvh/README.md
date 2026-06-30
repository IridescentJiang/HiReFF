# MVHuman Preprocessing

Convert [MVHumanNet](https://github.com/GAP-LAB-CUHK-SZ/MVHumanNet) data to HiReFF NPZ format.

## Directory Structure Required

```
mvhuman/<subject_id>/
  images/cam_XX/*.jpg     — RGB images per camera
  masks/cam_XX/*.png      — Binary foreground masks per camera
  cameras/cam_XX/camera.npz  — Camera parameters
```

Each `camera.npz` must contain:
- `intrinsic` — `(3, 3)` float32
- `extrinsic` — `(3, 4)` float32

## Scripts

| Script | Purpose |
|---|---|
| `mvh2npz.py` | Main conversion script with argparse |
| `mvh2npz.sh` | Batch wrapper (edit variables at top of file) |

## Usage

**Single subject:**

```bash
python mvh2npz.py \
    --data-root /path/to/mvhuman/<subject_id> \
    --output-dir /path/to/output \
    --camera-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
    --start-frame 0 --end-frame 100
```

**Batch (edit `mvh2npz.sh` variables first):**

```bash
bash mvh2npz.sh
```

Sh script variables: `PY`, `DATA_ROOT`, `SUBJECT_PATTERN`, `CAMERA_IDS`, `START_FRAME`, `END_FRAME`, `STEP`, `TARGET_SIZE`, `JPEG_QUALITY`, `OUTPUT_ROOT`.

## Output

```
output_root/<subject>/frame_0000.npz
output_root/<subject>/frame_0001.npz
  └── view_00, view_01, ...  (image, mask, intrinsic, extrinsic, camera_id)
```
