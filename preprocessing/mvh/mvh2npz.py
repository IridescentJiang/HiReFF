import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def parse_camera_id(cam_name: str) -> int:
	if not cam_name.startswith("cam_"):
		raise ValueError(f"Invalid camera folder name: {cam_name}")
	token = cam_name.split("_")[-1]
	return int(token)


def discover_camera_names(data_root: Path) -> list[str]:
	image_root = data_root / "images"
	if not image_root.exists():
		raise FileNotFoundError(f"Missing images directory: {image_root}")
	names = sorted([d.name for d in image_root.iterdir() if d.is_dir() and d.name.startswith("cam_")])
	if not names:
		raise FileNotFoundError(f"No camera folders found under: {image_root}")
	return names


def parse_camera_ids_arg(camera_ids_arg: str | None, auto_cam_names: list[str]) -> list[str]:
	if camera_ids_arg is None:
		return auto_cam_names

	selected = []
	for token in camera_ids_arg.split(","):
		token = token.strip()
		if not token:
			continue
		if token.startswith("cam_"):
			selected.append(token)
		else:
			selected.append(f"cam_{int(token):02d}")

	selected = sorted(set(selected))
	available = set(auto_cam_names)
	missing = [name for name in selected if name not in available]
	if missing:
		print(f"[warn] Requested cameras not found and will be ignored: {missing}")
	return [name for name in selected if name in available]


def list_frame_stems(folder: Path, suffixes: tuple[str, ...]) -> set[str]:
	stems = set()
	if not folder.exists():
		return stems
	for p in folder.iterdir():
		if p.is_file() and p.suffix.lower() in suffixes:
			stems.add(p.stem)
	return stems


def build_shared_frame_stems(data_root: Path, camera_names: list[str]) -> list[str]:
	common_stems = None
	for cam_name in camera_names:
		img_dir = data_root / "images" / cam_name
		mask_dir = data_root / "masks" / cam_name

		image_stems = list_frame_stems(img_dir, IMAGE_SUFFIXES)
		mask_stems = list_frame_stems(mask_dir, MASK_SUFFIXES)
		stems = image_stems & mask_stems
		if not stems:
			raise FileNotFoundError(f"No shared image/mask stems for {cam_name} under {img_dir} and {mask_dir}")

		if common_stems is None:
			common_stems = stems
		else:
			common_stems &= stems

	if not common_stems:
		raise ValueError("No frame stems are shared across all selected cameras")

	def sort_key(stem: str):
		if stem.isdigit():
			return (0, int(stem), stem)
		return (1, stem)

	return sorted(common_stems, key=sort_key)


def read_image(path: Path) -> np.ndarray:
	image = cv2.imread(str(path), cv2.IMREAD_COLOR)
	if image is None:
		raise RuntimeError(f"Failed to read image: {path}")
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, hw: tuple[int, int]) -> np.ndarray:
	mask_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if mask_raw is None:
		h, w = hw
		return np.ones((h, w), dtype=np.uint8) * 255

	if mask_raw.ndim == 2:
		mask_gray = mask_raw
	elif mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
		mask_gray = mask_raw[:, :, 3]
	else:
		mask_gray = cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
	return (mask_gray > 0).astype(np.uint8) * 255


def center_crop_square(image: np.ndarray) -> tuple[np.ndarray, int, int]:
	h, w = image.shape[:2]
	if h == w:
		return image, 0, 0
	if h > w:
		y0 = (h - w) // 2
		return image[y0:y0 + w, :], 0, y0
	x0 = (w - h) // 2
	return image[:, x0:x0 + h], x0, 0


def preprocess_image_mask(image_rgb: np.ndarray, mask_gray: np.ndarray, target_size: int) -> tuple[np.ndarray, np.ndarray, int, int, int]:
	image_sq, crop_x, crop_y = center_crop_square(image_rgb)
	mask_sq, _, _ = center_crop_square(mask_gray)

	side = image_sq.shape[0]
	if target_size > 0 and target_size != side:
		image_sq = cv2.resize(image_sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
		mask_sq = cv2.resize(mask_sq, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
		side = target_size

	mask_bin = (mask_sq > 0).astype(np.uint8) * 255
	return image_sq, mask_bin, crop_x, crop_y, side


def load_camera_params(data_root: Path, cam_name: str) -> tuple[np.ndarray, np.ndarray]:
	cam_file = data_root / "cameras" / cam_name / "camera.npz"
	if not cam_file.exists():
		raise FileNotFoundError(f"Missing camera file: {cam_file}")

	cam = np.load(cam_file)
	if "intrinsic" not in cam or "extrinsic" not in cam:
		raise KeyError(f"camera.npz must contain intrinsic and extrinsic: {cam_file}")

	intrinsic = np.asarray(cam["intrinsic"], dtype=np.float32)
	extrinsic_3x4 = np.asarray(cam["extrinsic"], dtype=np.float32)
	if intrinsic.shape != (3, 3):
		raise ValueError(f"Unexpected intrinsic shape {intrinsic.shape} in {cam_file}")
	if extrinsic_3x4.shape != (3, 4):
		raise ValueError(f"Unexpected extrinsic shape {extrinsic_3x4.shape} in {cam_file}")

	extrinsic = np.eye(4, dtype=np.float32)
	extrinsic[:3, :4] = extrinsic_3x4
	return intrinsic, extrinsic


def update_intrinsic_after_crop_resize(
	intrinsic: np.ndarray,
	crop_x: int,
	crop_y: int,
	original_hw: tuple[int, int],
	final_side: int,
) -> np.ndarray:
	k = intrinsic.copy().astype(np.float32)
	k[0, 2] -= float(crop_x)
	k[1, 2] -= float(crop_y)

	original_side = min(original_hw)
	scale = float(final_side) / float(original_side)
	k[0, 0] *= scale
	k[1, 1] *= scale
	k[0, 2] *= scale
	k[1, 2] *= scale
	return k


def find_existing_file(folder: Path, stem: str, suffixes: tuple[str, ...]) -> Path:
	for suffix in suffixes:
		candidate = folder / f"{stem}{suffix}"
		if candidate.exists():
			return candidate
	raise FileNotFoundError(f"Cannot find file for stem '{stem}' in {folder}")


def convert_mvhuman_to_npz(
	data_root: Path,
	output_dir: Path,
	target_size: int,
	start_frame: int,
	end_frame: int | None,
	step: int,
	camera_names: list[str],
	jpeg_quality: int,
	fid_offset: int,
	strict_missing: bool,
):
	frame_stems = build_shared_frame_stems(data_root, camera_names)

	if end_frame is None:
		end_frame = len(frame_stems)
	if start_frame < 0 or end_frame > len(frame_stems) or start_frame >= end_frame:
		raise ValueError(
			f"Invalid frame range: start={start_frame}, end={end_frame}, total={len(frame_stems)}"
		)

	sampled_indices = list(range(start_frame, end_frame, step))
	output_dir.mkdir(parents=True, exist_ok=True)

	camera_cache = {}
	for cam_name in camera_names:
		k, e = load_camera_params(data_root, cam_name)
		camera_cache[cam_name] = {"intrinsic": k, "extrinsic": e, "camera_id": np.int32(parse_camera_id(cam_name))}

	for sampled_idx, frame_list_idx in enumerate(tqdm(sampled_indices, desc="Converting MVHuman frames")):
		stem = frame_stems[frame_list_idx]
		frame_data = {}
		missing = []

		for view_idx, cam_name in enumerate(camera_names):
			image_dir = data_root / "images" / cam_name
			mask_dir = data_root / "masks" / cam_name

			try:
				image_path = find_existing_file(image_dir, stem, IMAGE_SUFFIXES)
				mask_path = find_existing_file(mask_dir, stem, MASK_SUFFIXES)
				image_rgb = read_image(image_path)
				h, w = image_rgb.shape[:2]
				mask_gray = read_mask(mask_path, (h, w))

				image_proc, mask_proc, crop_x, crop_y, final_side = preprocess_image_mask(
					image_rgb,
					mask_gray,
					target_size,
				)

				intrinsic_adj = update_intrinsic_after_crop_resize(
					camera_cache[cam_name]["intrinsic"],
					crop_x,
					crop_y,
					(h, w),
					final_side,
				)

				ok_img, encoded_img = cv2.imencode(
					".jpg",
					cv2.cvtColor(image_proc, cv2.COLOR_RGB2BGR),
					[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
				)
				ok_mask, encoded_mask = cv2.imencode(".png", mask_proc)
				if not ok_img or not ok_mask:
					raise RuntimeError("Encoding image/mask failed")

				key = f"view_{view_idx:02d}"
				frame_data[key] = {
					"image": encoded_img,
					"mask": encoded_mask,
					"intrinsic": intrinsic_adj.astype(np.float32),
					"extrinsic": camera_cache[cam_name]["extrinsic"],
					"camera_id": camera_cache[cam_name]["camera_id"],
				}
			except Exception as exc:
				missing.append(f"{cam_name}:{exc}")

		if len(frame_data) != len(camera_names):
			message = f"[skip] stem {stem} due to missing views: {missing}"
			if strict_missing:
				raise RuntimeError(message)
			print(message)
			continue

		out_fid = sampled_idx + fid_offset
		out_path = output_dir / f"frame_{out_fid:04d}.npz"
		np.savez(str(out_path), **frame_data)


def main():
	parser = argparse.ArgumentParser(description="Convert MVHuman data to DNA-rendering NPZ format.")
	parser.add_argument("--data-root", type=str, required=True, help="Path to one MVHuman subject, e.g. mvhuman/101286")
	parser.add_argument("--output-dir", type=str, required=True, help="Output folder for frame_xxxx.npz")
	parser.add_argument("--target-size", type=int, default=518 * 4, help="Final square resolution; <=0 keeps original square")
	parser.add_argument("--start-frame", type=int, default=0, help="Start index in sorted shared frame list (inclusive)")
	parser.add_argument("--end-frame", type=int, default=None, help="End index in sorted shared frame list (exclusive)")
	parser.add_argument("--step", type=int, default=1, help="Frame sampling step")
	parser.add_argument("--camera-ids", type=str, default=None, help="Comma-separated camera ids or names, e.g. 0,1,2 or cam_00,cam_01")
	parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for encoded image")
	parser.add_argument("--fid-offset", type=int, default=0, help="Output frame id offset")
	parser.add_argument("--strict-missing", action="store_true", help="Raise error when any view is missing instead of skipping frame")
	args = parser.parse_args()

	data_root = Path(args.data_root).resolve()
	output_dir = Path(args.output_dir).resolve()

	auto_cam_names = discover_camera_names(data_root)
	camera_names = parse_camera_ids_arg(args.camera_ids, auto_cam_names)
	if not camera_names:
		raise ValueError("No valid cameras selected")

	convert_mvhuman_to_npz(
		data_root=data_root,
		output_dir=output_dir,
		target_size=args.target_size,
		start_frame=args.start_frame,
		end_frame=args.end_frame,
		step=args.step,
		camera_names=camera_names,
		jpeg_quality=args.jpeg_quality,
		fid_offset=args.fid_offset,
		strict_missing=args.strict_missing,
	)


if __name__ == "__main__":
	main()
