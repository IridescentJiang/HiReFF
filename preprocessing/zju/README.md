# ZJU-MoCap Preprocessing

Convert [ZJU-MoCap](https://github.com/zju3dv/NeuralBody) data to HiReFF NPZ format.

Supports both directory layouts used by the dataset:
- `CoreView_313` style: `Camera (x)/` + `annots.json`
- `CoreView_377` style: `Camera_Bx/` + `annots.npy`

## Scripts

| Script | Purpose |
|---|---|
| `zju2npz.py` | Main conversion script with argparse |
| `zju2npz.sh` | Batch wrapper (edit variables at top of file) |

## Usage

**Single sequence (23 cameras):**

```bash
python zju2npz.py \
    --data-root /path/to/zju-mocap/CoreView_313 \
    --output-dir /path/to/output/CoreView_313 \
    --camera-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
```

**Batch (edit `zju2npz.sh` variables first):**

```bash
bash zju2npz.sh
```

**Subsampling (process every Nth frame):**

```bash
python zju2npz.py \
    --data-root /path/to/CoreView_313 \
    --output-dir /path/to/output \
    --camera-ids 1,...,23 \
    --start-frame 0 --end-frame 300 --step 3
```

Output filenames are numbered sequentially regardless of step (`frame_0000.npz`, `frame_0001.npz`, ...).

## Output

```
output_root/<subject>/frame_0000.npz
  └── view_00, view_01, ...  (image, mask, intrinsic, extrinsic)
```

## Notes

- If a scene is missing some camera files, omit `--camera-ids` to auto-detect available cameras.
- Camera ids in the NPZ are mapped sequentially: `view_00` corresponds to the first camera id, `view_01` to the second, etc.
