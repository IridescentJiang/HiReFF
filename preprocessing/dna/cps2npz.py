import os
import json
import io
import shutil
import subprocess
import tempfile
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import warnings

import numpy as np
import torch
import torchvision
import cv2
from tqdm import tqdm

from SMCReader import SMCReader

FFMPEG_EXECUTABLE = "ffmpeg"
TARGET_SIZE = 518 * 4
NUM_5MP_VIEWS = 48


def _load_images_npz(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    result = {}
    for k in data.keys():
        cam_key = k.replace(".npy", "")
        arr = data[k]
        if arr.dtype != np.uint8:
            if arr.dtype == object:
                arr = np.frombuffer(arr.tobytes(), dtype=np.uint8)
            else:
                arr = arr.astype(np.uint8)
        result[cam_key] = arr
    return result


def _decode_camera_task(args) -> Tuple[str, int, str]:
    """
    独立进程解码，返回临时目录路径
    """
    hevc_bytes, tmp_root, ext, input_format, vk = args
    # 每个相机独立的子目录
    cam_dir = os.path.join(tmp_root, f"cam_{vk}")
    os.makedirs(cam_dir, exist_ok=True)
    
    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", input_format,
        "-i", "-",
        "-vsync", "0",
        "-q:v", "2",
        "-threads", "2",
        os.path.join(cam_dir, f"frame_%04d.{ext}"),
    ]
    
    proc = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    
    try:
        out, err = proc.communicate(hevc_bytes.tobytes())
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed for {vk}: {err.decode('utf-8', 'ignore')}")
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    
    n = len([f for f in os.listdir(cam_dir) if f.endswith(f'.{ext}')])
    return vk, n, cam_dir  # 返回目录路径


def get_new_intrinsic(ori_intr, dist_coeff, ori_size, target_size):
    new_K_0, _ = cv2.getOptimalNewCameraMatrix(
        ori_intr, dist_coeff, ori_size, alpha=0.0, newImgSize=ori_size, centerPrincipalPoint=True
    )
    new_K = new_K_0.copy()
    height, width = ori_size[1], ori_size[0]
    if height > width:
        start_y = (height - width) // 2
        new_K[1, 2] -= start_y
    elif width > height:
        start_x = (width - height) // 2
        new_K[0, 2] -= start_x
    scale = target_size / min(height, width)
    new_K[0, 0] *= scale
    new_K[1, 1] *= scale
    new_K[0, 2] *= scale
    new_K[1, 2] *= scale
    return new_K, new_K_0


def get_undist_map(ori_intr, dist_coeff, ori_size, target_size, device):
    dist_coeff = np.zeros_like(dist_coeff)
    new_K, new_K_0 = get_new_intrinsic(ori_intr, dist_coeff, ori_size, target_size)
    map_x, map_y = cv2.initUndistortRectifyMap(
        ori_intr, dist_coeff, R=np.eye(3), newCameraMatrix=new_K_0,
        size=ori_size, m1type=cv2.CV_32FC1
    )
    map_x_tensor = torch.from_numpy(map_x).to(device)
    map_y_tensor = torch.from_numpy(map_y).to(device)
    return map_x_tensor, map_y_tensor, new_K


def process_images(image, mask, map_x, map_y, target_size=TARGET_SIZE):
    from kornia.geometry.transform import remap
    image_dist = remap(image, map_x, map_y, mode='bilinear', padding_mode='zeros')
    mask_dist = remap(mask, map_x, map_y, mode='bilinear', padding_mode='zeros')
    height, width = image_dist.shape[2], image_dist.shape[3]
    if height > width:
        start_y = (height - width) // 2
        image_dist = image_dist[:, :, start_y: start_y + width, :]
        mask_dist = mask_dist[:, :, start_y: start_y + width, :]
    elif width > height:
        start_x = (width - height) // 2
        image_dist = image_dist[:, :, :, start_x: start_x + height]
        mask_dist = mask_dist[:, :, :, start_x: start_x + height]
    image_dist = torch.nn.functional.interpolate(image_dist, size=(target_size, target_size), mode='bilinear', align_corners=False)
    mask_dist = torch.nn.functional.interpolate(mask_dist, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return image_dist, mask_dist


def delete_camera_frames(cam_dir: str):
    """删除相机帧释放磁盘空间"""
    if os.path.exists(cam_dir):
        shutil.rmtree(cam_dir, ignore_errors=True)


def convert_images_npz_to_training(
    input_dir: str,
    output_dir: str,
    device: Optional[torch.device] = None,
    num_decode_workers: int = 4,  # 降低并行度，避免内存爆炸
    batch_size: int = 4,          # 每批解码多少个相机
):
    """
    分批解码-处理-清理，严格控制内存和显存
    
    Args:
        num_decode_workers: 并行解码 workers（建议 2-4）
        batch_size: 每批处理的相机数（建议 4-8）
    """
    os.makedirs(output_dir, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    npz_path = os.path.join(input_dir, 'videos.npz')
    masks_npz_path = os.path.join(input_dir, 'masks.npz')
    cameras_path = os.path.join(input_dir, 'cameras.json')

    if not os.path.isfile(npz_path) or not os.path.isfile(cameras_path):
        raise FileNotFoundError('videos.npz or cameras.json not found')

    print(f"Loading NPZ files...")
    cam_bytes = _load_images_npz(npz_path)
    mask_bytes = _load_images_npz(masks_npz_path) if os.path.isfile(masks_npz_path) else None
    
    with open(cameras_path, 'r') as f:
        cameras = json.load(f)

    view_keys = [f"{i:02d}" for i in range(NUM_5MP_VIEWS)]

    # 预计算校准
    print(f"Precomputing calibration maps...")
    undistort_maps = {}
    extrinsics_arr = {}
    new_intrinsics_arr = {}
    
    for vk in view_keys:
        calib = cameras[vk]
        K = np.array(calib['K'], dtype=np.float32)
        D = np.array(calib['dist'], dtype=np.float32)
        w2c = np.array(calib['w2c'], dtype=np.float32)
        size = calib.get('img_size', None)
        if size is None:
            raise ValueError(f"img_size missing for view {vk}")
        ori_size = (int(size[1]), int(size[0]))
        
        map_x, map_y, new_K = get_undist_map(K, D, ori_size, TARGET_SIZE, device)
        undistort_maps[vk] = (map_x, map_y)
        extrinsics_arr[vk] = w2c
        new_intrinsics_arr[vk] = new_K.astype(np.float32)

    # 使用长期存在的临时目录，但分批清理
    tmp_root = tempfile.mkdtemp(prefix='decode_tmp_')
    
    try:
        # 分批处理相机组
        n_frames = None
        
        for batch_start in range(0, len(view_keys), batch_size):
            batch_end = min(batch_start + batch_size, len(view_keys))
            batch_keys = view_keys[batch_start:batch_end]
            
            print(f"\n{'='*60}")
            print(f"Processing camera batch {batch_start//batch_size + 1}: {batch_keys}")
            print(f"{'='*60}")
            
            # ========== Phase 1: 并行解码这批相机 ==========
            decode_tasks = []
            for vk in batch_keys:
                task = (cam_bytes[vk], tmp_root, "jpg", "hevc", vk)
                decode_tasks.append(task)
                
                if mask_bytes and vk in mask_bytes:
                    task_mask = (mask_bytes[vk], tmp_root, "png", "hevc", f"mask_{vk}")
                    decode_tasks.append(task_mask)
            
            cam_dirs = {}      # vk -> cam_dir
            mask_dirs = {}     # vk -> mask_dir
            frame_counts = []
            
            with ProcessPoolExecutor(max_workers=num_decode_workers) as executor:
                futures = {executor.submit(_decode_camera_task, task): task[-1] for task in decode_tasks}
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Decoding batch"):
                    vk, n, cam_dir = future.result()
                    
                    if vk.startswith("mask_"):
                        real_vk = vk.replace("mask_", "")
                        mask_dirs[real_vk] = cam_dir
                    else:
                        cam_dirs[vk] = cam_dir
                        frame_counts.append(n)
            
            if not frame_counts:
                continue
                
            current_n_frame = min(frame_counts)
            if n_frames is None:
                n_frames = current_n_frame
                print(f"Total frames to process: {n_frames}")
            else:
                # 确保所有相机帧数一致
                current_n_frame = min(current_n_frame, n_frames)
            
            # ========== Phase 2: 逐帧处理这批相机，GPU 串行 ==========
            print(f"GPU processing {current_n_frame} frames for {len(batch_keys)} cameras...")
            
            for fid in tqdm(range(current_n_frame), desc='Processing frames'):
                # 加载这批相机的当前帧
                frame_images = []
                frame_masks = []
                valid_keys = []
                
                for vk in batch_keys:
                    if vk not in cam_dirs:
                        continue
                        
                    cam_dir = cam_dirs[vk]
                    img_path = os.path.join(cam_dir, f"frame_{fid+1:04d}.jpg")
                    
                    if not os.path.exists(img_path):
                        continue
                    
                    # 读取到 GPU
                    img = torchvision.io.read_image(img_path).to(device).float() / 255.0
                    frame_images.append(img)
                    valid_keys.append(vk)
                    
                    # Mask
                    if vk in mask_dirs:
                        mask_dir = mask_dirs[vk]
                        mask_path = os.path.join(mask_dir, f"frame_{fid+1:04d}.png")
                        if os.path.exists(mask_path):
                            mask = torchvision.io.read_image(mask_path).to(device).float() / 255.0
                        else:
                            mask = torch.ones((1, img.shape[1], img.shape[2]), device=device)
                    else:
                        mask = torch.ones((1, img.shape[1], img.shape[2]), device=device)
                    frame_masks.append(mask)
                
                if not frame_images:
                    continue
                
                # 批量处理
                frame_images_t = torch.stack(frame_images, dim=0)
                frame_masks_t = torch.stack(frame_masks, dim=0)
                
                # 获取这批相机的 remap 参数
                batch_map_x = torch.stack([undistort_maps[vk][0] for vk in valid_keys], dim=0)
                batch_map_y = torch.stack([undistort_maps[vk][1] for vk in valid_keys], dim=0)
                
                processed_images, processed_masks = process_images(
                    frame_images_t, frame_masks_t,
                    batch_map_x, batch_map_y,
                    target_size=TARGET_SIZE,
                )
                
                processed_images = (processed_images * 255.0).to(torch.uint8)
                processed_masks = (processed_masks > 0.5).to(torch.uint8) * 255
                
                # 保存结果（追加到已有文件或新建）
                codec_result = {}
                for i, vk in enumerate(valid_keys):
                    key = f"view_{vk}"
                    codec_result[key] = {
                        "image": torchvision.io.encode_jpeg(processed_images[i], quality=100).cpu().numpy(),
                        "mask": torchvision.io.encode_png(processed_masks[i].cpu()).numpy(),
                        "intrinsic": new_intrinsics_arr[vk],
                        "extrinsic": extrinsics_arr[vk],
                    }
                
                # 如果是第一批，新建文件；否则加载并追加（或者用更聪明的方式）
                out_path = os.path.join(output_dir, f"frame_{fid:04d}.npz")
                if os.path.exists(out_path) and batch_start > 0:
                    # 加载已有数据，合并
                    old_data = dict(np.load(out_path, allow_pickle=True))
                    old_data.update(codec_result)
                    np.savez(out_path, **old_data)
                else:
                    np.savez(out_path, **codec_result)
                
                # 立即释放 GPU 内存
                del frame_images_t, frame_masks_t, processed_images, processed_masks
                del frame_images, frame_masks
                torch.cuda.empty_cache()
                
                # 每 10 帧强制垃圾回收
                if fid % 10 == 0:
                    gc.collect()
            
            # ========== Phase 3: 清理这批相机的临时文件 ==========
            print(f"Cleaning up batch {batch_start//batch_size + 1}...")
            for vk in batch_keys:
                if vk in cam_dirs:
                    delete_camera_frames(cam_dirs[vk])
                if vk in mask_dirs:
                    delete_camera_frames(mask_dirs[vk])
            
            # 强制系统回收内存
            gc.collect()
            torch.cuda.empty_cache()
            
            # 删除已处理的 bytes 数据释放内存
            for vk in batch_keys:
                if vk in cam_bytes:
                    del cam_bytes[vk]
                if mask_bytes and vk in mask_bytes:
                    del mask_bytes[vk]
        
        print(f"\nDone! Results saved to {output_dir}")
        
    finally:
        # 清理临时目录
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)
        torch.cuda.empty_cache()
        gc.collect()

    # 保存 cameras.json
    shutil.copy2(cameras_path, os.path.join(output_dir, 'cameras.json'))


def _split_ids_round_robin(ids: List[str], n_parts: int) -> List[List[str]]:
    parts = [[] for _ in range(n_parts)]
    for idx, item in enumerate(ids):
        parts[idx % n_parts].append(item)
    return parts


def _already_converted(output_dir: str) -> bool:
    if not os.path.isdir(output_dir):
        return False
    for name in os.listdir(output_dir):
        if name.startswith("frame_") and name.endswith(".npz"):
            return True
    return False


def _worker_process_ids(
    worker_rank: int,
    gpu_id: Optional[int],
    ids: List[str],
    base_dir: str,
    output_base_dir: str,
    num_decode_workers: int,
    batch_size: int,
    skip_done_case: bool,
):
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    print(
        f"\n[Worker {worker_rank}] start: device={device}, "
        f"num_ids={len(ids)}, decode_workers={num_decode_workers}, batch_size={batch_size}"
    )

    for id_name in ids:
        input_dir = os.path.join(base_dir, id_name)
        output_dir = os.path.join(output_base_dir, id_name)

        if skip_done_case and _already_converted(output_dir):
            print(f"[Worker {worker_rank}] skip {id_name}: already converted")
            continue

        print(f"\n{'='*60}")
        print(f"[Worker {worker_rank}] Processing {id_name}")
        print(f"{'='*60}")

        try:
            convert_images_npz_to_training(
                input_dir,
                output_dir,
                device=device,
                num_decode_workers=num_decode_workers,
                batch_size=batch_size,
            )
        except Exception as exc:
            print(f"[Worker {worker_rank}] ERROR on {id_name}: {exc}")

    print(f"[Worker {worker_rank}] done")


if __name__ == '__main__':
    base_dir = './dna_rendering_part2_cps'
    output_base_dir = './data_npz_part2_cps'

    NUM_DECODE_WORKERS = 4
    BATCH_SIZE = 24
    SKIP_DONE_CASE = True
    MAX_WORKERS = 0
    GPU_IDS = []

    cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if cuda_count > 0:
        if len(GPU_IDS) == 0:
            selected_gpu_ids = list(range(cuda_count))
        else:
            selected_gpu_ids = []
            for gpu_id in GPU_IDS:
                if not isinstance(gpu_id, int):
                    raise ValueError(f"GPU_IDS must be int list, but got: {gpu_id}")
                if gpu_id < 0 or gpu_id >= cuda_count:
                    raise ValueError(f"GPU id out of range: {gpu_id}, available: 0..{cuda_count - 1}")
                selected_gpu_ids.append(gpu_id)
            selected_gpu_ids = sorted(set(selected_gpu_ids))

        if MAX_WORKERS > 0:
            NUM_WORKERS = min(MAX_WORKERS, len(selected_gpu_ids))
        else:
            NUM_WORKERS = len(selected_gpu_ids)
    else:
        selected_gpu_ids = []
        NUM_WORKERS = 1

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    ids = sorted([name for name in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, name))])

    print(f"{len(ids)} IDs found")
    print(f"CUDA devices: {cuda_count}")
    print(f"Selected GPUs: {selected_gpu_ids if selected_gpu_ids else 'CPU'}")
    print(f"Worker processes: {NUM_WORKERS} (MAX_WORKERS={MAX_WORKERS})")

    if len(ids) == 0:
        raise RuntimeError("No IDs found to process")

    if NUM_WORKERS == 1:
        single_gpu_id = selected_gpu_ids[0] if len(selected_gpu_ids) > 0 else None
        _worker_process_ids(
            worker_rank=0,
            gpu_id=single_gpu_id,
            ids=ids,
            base_dir=base_dir,
            output_base_dir=output_base_dir,
            num_decode_workers=NUM_DECODE_WORKERS,
            batch_size=BATCH_SIZE,
            skip_done_case=SKIP_DONE_CASE,
        )
    else:
        mp.set_start_method('spawn', force=True)
        chunks = _split_ids_round_robin(ids, NUM_WORKERS)
        processes = []
        for worker_rank in range(NUM_WORKERS):
            if len(chunks[worker_rank]) == 0:
                continue
            gpu_id = selected_gpu_ids[worker_rank % len(selected_gpu_ids)] if len(selected_gpu_ids) > 0 else None
            p = mp.Process(
                target=_worker_process_ids,
                args=(
                    worker_rank,
                    gpu_id,
                    chunks[worker_rank],
                    base_dir,
                    output_base_dir,
                    NUM_DECODE_WORKERS,
                    BATCH_SIZE,
                    SKIP_DONE_CASE,
                ),
            )
            p.start()
            processes.append(p)

        failed = False
        for p in processes:
            p.join()
            if p.exitcode != 0:
                failed = True
                print(f"[main] worker pid={p.pid} exited with code {p.exitcode}")

        if failed:
            raise RuntimeError("One or more worker processes failed")