import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_CAMERA_FILE_INDEX_CACHE: dict[str, dict[int, list[Path]]] = {}


def parse_camera_id(text: str) -> int:
    match = re.search(r"Camera\s*\((\d+)\)|Camera_B(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse camera id from: {text}")
    if match.group(1) is not None:
        return int(match.group(1))
    return int(match.group(2))


def resolve_camera_dir(data_root: Path, cam_id: int) -> str | None:
    candidates = [f"Camera ({cam_id})", f"Camera_B{cam_id}"]
    for name in candidates:
        if (data_root / name).exists():
            return name
    return None


def discover_camera_ids(data_root: Path) -> list[int]:
    ids = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("Camera (") or child.name.startswith("Camera_B"):
            ids.append(parse_camera_id(child.name))
    if not ids:
        raise FileNotFoundError(f"Cannot find camera folders under: {data_root}")
    return sorted(ids)


def _extract_frame_token(file_name: str) -> int | None:
    stem = Path(file_name).stem
    if stem.isdigit():
        return int(stem)
    match = re.search(r"_(\d{4,6})(?:_|$)", stem)
    if match:
        return int(match.group(1))
    return None


def _build_camera_file_index(camera_dir: Path) -> dict[int, list[Path]]:
    key = str(camera_dir.resolve())
    if key in _CAMERA_FILE_INDEX_CACHE:
        return _CAMERA_FILE_INDEX_CACHE[key]

    token_to_paths: dict[int, list[Path]] = {}
    if camera_dir.exists():
        for path in sorted(camera_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            token = _extract_frame_token(path.name)
            if token is None:
                continue
            token_to_paths.setdefault(token, []).append(path)

    _CAMERA_FILE_INDEX_CACHE[key] = token_to_paths
    return token_to_paths


def _resolve_with_frame_token(path: Path) -> Path | None:
    camera_dir = path.parent
    if not camera_dir.exists():
        return None

    token = _extract_frame_token(path.name)
    if token is None:
        return None

    token_to_paths = _build_camera_file_index(camera_dir)
    candidates = token_to_paths.get(token)
    if not candidates:
        return None

    same_suffix = [candidate for candidate in candidates if candidate.suffix.lower() == path.suffix.lower()]
    if same_suffix:
        return same_suffix[0]
    return candidates[0]


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    def _with_scene_name_fallback(path: Path) -> Path:
        candidates = [path]
        fallback_name = re.sub(r"^CoreView_\d+", data_root.name, path.name)
        if fallback_name != path.name:
            candidates.append(path.parent / fallback_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        for candidate in candidates:
            resolved = _resolve_with_frame_token(candidate)
            if resolved is not None:
                return resolved

        return candidates[-1]

    rel_path = rel_path.replace("\\", "/").lstrip("./")

    path_obj = Path(rel_path)
    if path_obj.is_absolute():
        return _with_scene_name_fallback(path_obj)

    parts = list(path_obj.parts)

    for idx, part in enumerate(parts):
        if part.startswith("Camera (") or part.startswith("Camera_B"):
            return _with_scene_name_fallback(data_root / Path(*parts[idx:]))

    if parts and (parts[0] == data_root.name or parts[0].startswith("CoreView_")):
        parts = parts[1:]

    return _with_scene_name_fallback(data_root / Path(*parts))


def _find_first_readable_image(
    data_root: Path,
    ims: list[dict],
    start_frame: int,
    end_frame: int,
    camera_ids: list[int],
) -> tuple[np.ndarray, Path]:
    camera_id_set = set(camera_ids)
    for frame_idx in range(start_frame, end_frame):
        for rel in ims[frame_idx]["ims"]:
            cam_id = parse_camera_id(rel)
            if cam_id not in camera_id_set:
                continue
            image_path = resolve_image_path(data_root, rel)
            if not image_path.exists():
                continue
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is not None:
                return image_bgr, image_path
    raise FileNotFoundError(
        "Cannot find any readable image in selected frame range and camera ids. "
        f"range=[{start_frame}, {end_frame}), camera_ids={camera_ids}"
    )


def get_new_intrinsic(ori_intr: np.ndarray, dist_coeff: np.ndarray, ori_size: tuple[int, int], target_size: int):
    new_k_0, _ = cv2.getOptimalNewCameraMatrix(
        ori_intr,
        dist_coeff,
        ori_size,
        alpha=0.0,
        newImgSize=ori_size,
        centerPrincipalPoint=True,
    )
    new_k = new_k_0.copy()

    height, width = ori_size[1], ori_size[0]
    if height > width:
        start_y = (height - width) // 2
        new_k[1, 2] -= start_y
    elif width > height:
        start_x = (width - height) // 2
        new_k[0, 2] -= start_x

    scale = target_size / min(height, width)
    new_k[0, 0] *= scale
    new_k[1, 1] *= scale
    new_k[0, 2] *= scale
    new_k[1, 2] *= scale
    return new_k, new_k_0


def center_crop_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image
    if h > w:
        y0 = (h - w) // 2
        return image[y0:y0 + w, :]
    x0 = (w - h) // 2
    return image[:, x0:x0 + h]


def preprocess_image_mask(
    image_rgb: np.ndarray,
    mask_gray: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    image_ud = cv2.remap(image_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_ud = cv2.remap(mask_gray, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    image_sq = center_crop_square(image_ud)
    mask_sq = center_crop_square(mask_ud)

    image_rs = cv2.resize(image_sq, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask_sq, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_rs > 0).astype(np.uint8) * 255
    return image_rs, mask_bin


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_mask_name_candidates(image_path: Path, frame_idx: int) -> list[str]:
    stem = image_path.stem
    names = [
        stem,
        image_path.name,
        f"{frame_idx:06d}",
        f"{frame_idx:04d}",
        f"{frame_idx + 1:06d}",
        f"{frame_idx + 1:04d}",
    ]

    digit_match = re.search(r"_(\d{4,6})(?:_|$)", stem)
    if digit_match:
        token = digit_match.group(1)
        names.append(token)
        token_int = int(token)
        names.append(f"{token_int:06d}")
        names.append(f"{token_int:04d}")
        if token_int > 0:
            names.append(f"{token_int - 1:06d}")
            names.append(f"{token_int - 1:04d}")

    return _dedupe_keep_order(names)


def resolve_mask_path(
    data_root: Path,
    camera_dir_name: str,
    image_path: Path,
    frame_idx: int,
    mask_dir_name: str,
    fallback_mask_dir_name: str | None,
) -> Path | None:
    dir_candidates = [mask_dir_name]
    if fallback_mask_dir_name:
        dir_candidates.append(fallback_mask_dir_name)

    name_candidates = build_mask_name_candidates(image_path, frame_idx)

    for dir_name in dir_candidates:
        cam_mask_dir = data_root / dir_name / camera_dir_name
        if not cam_mask_dir.exists():
            continue

        for name in name_candidates:
            candidate = cam_mask_dir / name
            if candidate.is_file():
                return candidate

            stem = Path(name).stem
            for suffix in MASK_SUFFIXES:
                candidate = cam_mask_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate

    return None


def read_mask_as_grayscale(mask_path: Path | None, image_shape_hw: tuple[int, int]) -> np.ndarray:
    if mask_path is None or not mask_path.exists():
        h, w = image_shape_hw
        return np.ones((h, w), dtype=np.uint8) * 255

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        h, w = image_shape_hw
        return np.ones((h, w), dtype=np.uint8) * 255

    if mask_raw.ndim == 2:
        return mask_raw
    if mask_raw.ndim == 3 and mask_raw.shape[2] == 4:
        return mask_raw[:, :, 3]
    return cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)


def _normalize_annots(annots: dict, date_key: str | None = None):
    if "cams" not in annots or "ims" not in annots:
        raise KeyError("Annotation file must contain keys 'cams' and 'ims'")

    cams_all = annots["cams"]
    if isinstance(cams_all, dict) and all(key in cams_all for key in ["K", "D", "R", "T"]):
        cams = cams_all
    elif isinstance(cams_all, dict):
        if date_key is None:
            if len(cams_all) != 1:
                raise ValueError(
                    f"Annotation cams has multiple date keys: {list(cams_all.keys())}, please set --date-key"
                )
            date_key = next(iter(cams_all.keys()))
        if date_key not in cams_all:
            raise KeyError(f"date key {date_key} not found in annotation cams")
        cams = cams_all[date_key]
    else:
        raise TypeError("Unsupported annotation cams format")

    ims = annots["ims"]
    return cams, ims


def load_annots(data_root: Path, date_key: str | None = None):
    annots_json_path = data_root / "annots.json"
    if annots_json_path.exists():
        with annots_json_path.open("r", encoding="utf-8") as f:
            annots = json.load(f)
        return _normalize_annots(annots, date_key=date_key)

    annots_npy_path = data_root / "annots.npy"
    if annots_npy_path.exists():
        annots = np.load(str(annots_npy_path), allow_pickle=True).item()
        return _normalize_annots(annots, date_key=date_key)

    raise FileNotFoundError(f"Cannot find annots.json or annots.npy under: {data_root}")


def parse_camera_ids_arg(camera_ids_arg: str | None, auto_ids: list[int]) -> list[int]:
    if camera_ids_arg is None:
        return auto_ids
    result = []
    for token in camera_ids_arg.split(","):
        token = token.strip()
        if token:
            result.append(int(token))
    return sorted(set(result))


def build_calib_index_map(ims: list[dict]) -> dict[int, int]:
    if not ims:
        raise ValueError("annots['ims'] is empty")

    first_frame = ims[0]
    if "ims" not in first_frame:
        raise KeyError("annots['ims'][0] does not contain key 'ims'")

    camera_ids = []
    seen = set()
    for rel in first_frame["ims"]:
        cam_id = parse_camera_id(rel)
        if cam_id not in seen:
            seen.add(cam_id)
            camera_ids.append(cam_id)

    return {cam_id: idx for idx, cam_id in enumerate(camera_ids)}


def build_camera_cache(
    cams: dict,
    camera_ids: list[int],
    calib_index_map: dict[int, int],
    ori_size: tuple[int, int],
    target_size: int,
):
    cache = {}
    for cam_id in camera_ids:
        if cam_id not in calib_index_map:
            raise KeyError(f"Camera id {cam_id} does not exist in annots calibration map")

        idx = calib_index_map[cam_id]
        k = np.asarray(cams["K"][idx], dtype=np.float32)
        d = np.asarray(cams["D"][idx], dtype=np.float32).reshape(-1, 1)
        r = np.asarray(cams["R"][idx], dtype=np.float32)
        t = np.asarray(cams["T"][idx], dtype=np.float32).reshape(3)

        new_k, new_k0 = get_new_intrinsic(k, d, ori_size, target_size)
        map_x, map_y = cv2.initUndistortRectifyMap(
            cameraMatrix=k,
            distCoeffs=d,
            R=np.eye(3, dtype=np.float32),
            newCameraMatrix=new_k0,
            size=ori_size,
            m1type=cv2.CV_32FC1,
        )

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = r
        extrinsic[:3, 3] = t

        cache[cam_id] = {
            "map_x": map_x,
            "map_y": map_y,
            "intrinsic": new_k.astype(np.float32),
            "extrinsic": extrinsic,
        }
    return cache


def convert_zju_to_npz(
    data_root: Path,
    output_dir: Path,
    target_size: int,
    start_frame: int,
    end_frame: int | None,
    step: int,
    camera_ids: list[int],
    mask_dir_name: str,
    fallback_mask_dir_name: str | None,
    jpeg_quality: int,
    fid_offset: int,
    date_key: str | None,
):
    cams, ims = load_annots(data_root, date_key=date_key)
    calib_index_map = build_calib_index_map(ims)

    available_camera_ids = sorted(calib_index_map.keys())
    missing_camera_ids = sorted(set(camera_ids) - set(available_camera_ids))
    if missing_camera_ids:
        print(
            "[warn] Requested camera ids not found in this sequence and will be skipped: "
            f"{missing_camera_ids}"
        )
        camera_ids = [camera_id for camera_id in camera_ids if camera_id in calib_index_map]

    if not camera_ids:
        raise ValueError(
            "After filtering unavailable cameras, no valid camera id remains. "
            f"Available camera ids: {available_camera_ids}"
        )

    if end_frame is None:
        end_frame = len(ims)

    if start_frame < 0 or end_frame > len(ims) or start_frame >= end_frame:
        raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}, total={len(ims)}")

    first_img, first_path = _find_first_readable_image(
        data_root=data_root,
        ims=ims,
        start_frame=start_frame,
        end_frame=end_frame,
        camera_ids=camera_ids,
    )
    h, w = first_img.shape[:2]
    ori_size = (w, h)

    camera_cache = build_camera_cache(cams, camera_ids, calib_index_map, ori_size, target_size)

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = list(range(start_frame, end_frame, step))
    for sampled_idx, frame_idx in enumerate(tqdm(frame_indices, desc="Converting ZJU frames")):
        frame_info = ims[frame_idx]
        frame_image_paths = {}
        for rel in frame_info["ims"]:
            cam_id = parse_camera_id(rel)
            frame_image_paths[cam_id] = resolve_image_path(data_root, rel)

        codec_result = {}
        missing = []

        for view_idx, cam_id in enumerate(camera_ids):
            if cam_id not in frame_image_paths:
                missing.append(f"cam{cam_id}:missing_in_annots")
                continue

            image_path = frame_image_paths[cam_id]
            if not image_path.exists():
                missing.append(f"cam{cam_id}:missing_image")
                continue

            camera_dir_name = resolve_camera_dir(data_root, cam_id)
            if camera_dir_name is None:
                missing.append(f"cam{cam_id}:camera_dir_not_found")
                continue

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                missing.append(f"cam{cam_id}:read_image_failed")
                continue

            mask_path = resolve_mask_path(
                data_root=data_root,
                camera_dir_name=camera_dir_name,
                image_path=image_path,
                frame_idx=frame_idx,
                mask_dir_name=mask_dir_name,
                fallback_mask_dir_name=fallback_mask_dir_name,
            )
            if mask_path is None:
                missing.append(f"cam{cam_id}:mask_not_found_using_fullwhite")

            mask_gray = read_mask_as_grayscale(mask_path, (image_bgr.shape[0], image_bgr.shape[1]))

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            cam_cache = camera_cache[cam_id]
            image_proc, mask_proc = preprocess_image_mask(
                image_rgb,
                mask_gray,
                cam_cache["map_x"],
                cam_cache["map_y"],
                target_size,
            )

            ok_img, encoded_img = cv2.imencode(
                ".jpg",
                cv2.cvtColor(image_proc, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            ok_mask, encoded_mask = cv2.imencode(".png", mask_proc)
            if not ok_img or not ok_mask:
                missing.append(f"cam{cam_id}:encode_failed")
                continue

            key = f"view_{view_idx:02d}"
            codec_result[key] = {
                "image": encoded_img,
                "mask": encoded_mask,
                "intrinsic": cam_cache["intrinsic"],
                "extrinsic": cam_cache["extrinsic"],
                "camera_id": np.int32(cam_id),
            }

        if len(codec_result) != len(camera_ids):
            print(f"[skip] frame {frame_idx} due to missing views: {missing}")
            continue

        out_fid = sampled_idx + fid_offset
        out_path = output_dir / f"frame_{out_fid:04d}.npz"
        np.savez(str(out_path), **codec_result)


def main():
    parser = argparse.ArgumentParser(description="Convert ZJU-MoCap image dataset to DNA-rendering NPZ format.")
    parser.add_argument("--data-root", type=str, required=True, help="Path to one subject folder, e.g., zju-mocap/CoreView_313")
    parser.add_argument("--output-dir", type=str, required=True, help="Output folder for frame_xxxx.npz files")
    parser.add_argument("--target-size", type=int, default=518 * 4, help="Final square resolution")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index (0-based, inclusive)")
    parser.add_argument("--end-frame", type=int, default=None, help="End frame index (0-based, exclusive)")
    parser.add_argument("--step", type=int, default=1, help="Frame sampling step")
    parser.add_argument("--camera-ids", type=str, default=None, help="Comma-separated camera ids, e.g. 1,2,3,4")
    parser.add_argument("--mask-dir", type=str, default="mask_cihp", help="Primary mask directory name under data root")
    parser.add_argument("--fallback-mask-dir", type=str, default=None, help="Fallback mask directory name")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for image encoding")
    parser.add_argument("--fid-offset", type=int, default=0, help="Output frame id offset")
    parser.add_argument("--date-key", type=str, default=None, help="Date key in annots.json cams field")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    auto_ids = discover_camera_ids(data_root)
    camera_ids = parse_camera_ids_arg(args.camera_ids, auto_ids)
    if not camera_ids:
        raise ValueError("No camera ids selected")

    convert_zju_to_npz(
        data_root=data_root,
        output_dir=output_dir,
        target_size=args.target_size,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        step=args.step,
        camera_ids=camera_ids,
        mask_dir_name=args.mask_dir,
        fallback_mask_dir_name=args.fallback_mask_dir,
        jpeg_quality=args.jpeg_quality,
        fid_offset=args.fid_offset,
        date_key=args.date_key,
    )


if __name__ == "__main__":
    main()
