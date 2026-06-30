#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="$WORK_ROOT/.venv/bin/python"
DATA_ROOT="/media/china/Extreme SSD/zju-mocap"
SCENE_PATTERN="CoreView_*"
CAMERA_IDS="$(seq -s, 1 23)"
STEP=5
START_FRAME=0
END_FRAME=""
TARGET_SIZE=2072
OUTPUT_ROOT="$DATA_ROOT/data_npz_zju"
MASK_DIR="mask_cihp"
CLEAN_OUTPUT=1

if [[ ! -x "$PY" ]]; then
  echo "[ERROR] Python executable not found or not executable: $PY"
  exit 1
fi

scene_dirs=()
if [[ -d "$DATA_ROOT" ]] && ([[ -f "$DATA_ROOT/annots.npy" ]] || [[ -f "$DATA_ROOT/annots.json" ]]); then
  scene_dirs+=("$DATA_ROOT")
else
  shopt -s nullglob
  for scene_dir in "$DATA_ROOT"/$SCENE_PATTERN; do
    [[ -d "$scene_dir" ]] && scene_dirs+=("$scene_dir")
  done
  shopt -u nullglob
fi

if [[ ${#scene_dirs[@]} -eq 0 ]]; then
  echo "[ERROR] No scene directories found. DATA_ROOT=$DATA_ROOT SCENE_PATTERN=$SCENE_PATTERN"
  exit 1
fi

for scene_dir in "${scene_dirs[@]}"; do
  scene_name=$(basename "$scene_dir")
  echo "[INFO] converting $scene_name (step=$STEP, start=$START_FRAME, end=${END_FRAME:-all}) ..."

  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -rf "$OUTPUT_ROOT/$scene_name"
  fi

  cmd=(
    "$PY" "$SCRIPT_DIR/zju2npz.py"
    --data-root "$scene_dir"
    --output-dir "$OUTPUT_ROOT/$scene_name"
    --camera-ids "$CAMERA_IDS"
    --start-frame "$START_FRAME"
    --step "$STEP"
    --target-size "$TARGET_SIZE"
    --mask-dir "$MASK_DIR"
  )

  if [[ -n "$END_FRAME" ]]; then
    cmd+=(--end-frame "$END_FRAME")
  fi

  "${cmd[@]}"
done