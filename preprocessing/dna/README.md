# DNA-Rendering Preprocessing

Scripts for converting the [DNA-Rendering](https://dna-rendering.github.io/) dataset to HiReFF's NPZ training format.

## Data Format Pipeline

```
SMC (raw)  ──→  CPS (compressed)  ──→  NPZ (training)
```

| Format | Description |
|---|---|
| `smc` | DNA-Rendering's original storage format (HDF5-based) |
| `cps` | Compressed format: images encoded as H.264 video for efficient storage |
| `npz` | Final training format: per-frame `.npz` archives with JPEG-encoded views |

## Scripts

| Script | Input | Output | Purpose |
|---|---|---|---|
| `SMCReader.py` | `.smc` file | Frames (numpy) | Read SMC/HDF5 format |
| `smc2cps.py` | `.smc` directory | `.cps` video files | Compress SMC to CPS for storage |
| `cps2npz.py` | `.cps` directory | `frame_XXXX.npz` | Decode CPS and pack into NPZ |
| `smc2npz.py` | `.smc` directory | `frame_XXXX.npz` | Direct SMC → NPZ (single step) |

## Example

```bash
# Option A: Two-step (SMC → CPS → NPZ), saves disk space
python smc2cps.py --input /path/to/smc_data --output /path/to/cps_data
python cps2npz.py --input /path/to/cps_data --output /path/to/npz_data

# Option B: Direct (SMC → NPZ), faster if disk is not a concern
python smc2npz.py --input /path/to/smc_data --output /path/to/npz_data
```
