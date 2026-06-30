#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="$WORK_ROOT/.venv/bin/python"
DATA_ROOT="$SCRIPT_DIR/raw_data"
SUBJECT_PATTERN="*"
CAMERA_IDS="$(seq -s, 0 15)"
STEP="1"
START_FRAME="0"
END_FRAME=""
TARGET_SIZE="2072"
JPEG_QUALITY="95"
FID_OFFSET="0"
STRICT_MISSING="0"
CLEAN_OUTPUT="0"
SKIP_DONE_CASE="1"
OUTPUT_ROOT="$SCRIPT_DIR/data_npz_mvh"

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python executable not found or not executable: $PY"
  exit 1
fi

subject_dirs=()
if [[ -d "$DATA_ROOT/images" && -d "$DATA_ROOT/masks" && -d "$DATA_ROOT/cameras" ]]; then
  subject_dirs+=("$DATA_ROOT")
else
  shopt -s nullglob
  for d in "$DATA_ROOT"/$SUBJECT_PATTERN; do
    if [[ -d "$d/images" && -d "$d/masks" && -d "$d/cameras" ]]; then
      subject_dirs+=("$d")
    fi
  done
  shopt -u nullglob
fi

if [[ ${#subject_dirs[@]} -eq 0 ]]; then
  echo "[ERROR] No MVHuman subject directories found. DATA_ROOT=$DATA_ROOT SUBJECT_PATTERN=$SUBJECT_PATTERN"
  exit 1
fi

for subject_dir in "${subject_dirs[@]}"; do
  subject_name="$(basename "$subject_dir")"
  out_dir="$OUTPUT_ROOT/$subject_name"

  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -rf "$out_dir"
  fi

  if [[ "$SKIP_DONE_CASE" == "1" ]] && [[ -d "$out_dir" ]] && compgen -G "$out_dir/frame_*.npz" > /dev/null; then
    echo "[INFO] skip $subject_name: already processed at $out_dir"
    continue
  fi

  echo "[INFO] converting $subject_name (cams=$CAMERA_IDS, step=$STEP, start=$START_FRAME, end=${END_FRAME:-all})"

  cmd=(
    "$PY" "$SCRIPT_DIR/mvh2npz.py"
    --data-root "$subject_dir"
    --output-dir "$out_dir"
    --camera-ids "$CAMERA_IDS"
    --start-frame "$START_FRAME"
    --step "$STEP"
    --target-size "$TARGET_SIZE"
    --jpeg-quality "$JPEG_QUALITY"
    --fid-offset "$FID_OFFSET"
  )

  if [[ -n "$END_FRAME" ]]; then
    cmd+=(--end-frame "$END_FRAME")
  fi

  if [[ "$STRICT_MISSING" == "1" ]]; then
    cmd+=(--strict-missing)
  fi

  "${cmd[@]}"
done

echo "[INFO] done. output root: $OUTPUT_ROOT"
