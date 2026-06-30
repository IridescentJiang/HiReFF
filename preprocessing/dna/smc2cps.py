import subprocess
import numpy as np
import torch
import json
import torchvision
import os
import io
import zipfile
import ctypes
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from SMCReader import SMCReader

FFMPEG_EXECUTABLE = "ffmpeg"


def malloc_trim():
    """强制 Python 归还内存给操作系统"""
    gc.collect()
    try:
        libc = ctypes.CDLL('libc.so.6')
        libc.malloc_trim(0)
    except:
        pass


def encode_frames_to_video(
    frames_array,
    fps=30,
    pix_fmt="rgb24",
    codec="libx265",
    preset="ultrafast",  # 最快预设
    x265_params="",
    container="hevc",
):
    if frames_array is None or frames_array.size == 0:
        return b""

    if not frames_array.flags["C_CONTIGUOUS"]:
        frames_array = np.ascontiguousarray(frames_array)

    h, w = frames_array.shape[1:3]
    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-s", f"{w}x{h}",
        "-framerate", str(fps),
        "-i", "-",
        "-an",
        "-c:v", codec,
        "-preset", preset,
    ]
    if codec == "libx265" and x265_params:
        cmd += ["-x265-params", x265_params]
    cmd += [
        "-pix_fmt", pix_fmt,
        "-f", container,
        "-",
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        out, err = proc.communicate(frames_array.tobytes())
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode('utf-8', 'ignore')}")
    return out


def process_single_camera(args):
    """独立进程处理单个相机"""
    smc_path, annots_path, rgb_vidx, n_frame, device_id = args
    
    # 每个进程独立打开 SMCReader
    rd = SMCReader(smc_path)
    rd_annots = SMCReader(annots_path) if annots_path and os.path.exists(annots_path) else None
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    if rgb_vidx < 48:
        camera_str = "Camera_5mp"
    else:
        return None, None, None
    
    frames_array = None
    masks_array = None
    
    try:
        for fid in range(n_frame):
            view_img = rd.get_img(camera_str, rgb_vidx, 'color', Frame_id=fid, return_bytes=True)
            view_img = torchvision.io.decode_jpeg(
                torch.from_numpy(view_img),
                mode=torchvision.io.ImageReadMode.RGB,
                device=device,
            )
            frame = view_img.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)

            if rd_annots is not None:
                view_mask = rd_annots.get_mask(rgb_vidx, Frame_id=fid, return_bytes=True)
                view_mask = torchvision.io.decode_image(
                    torch.from_numpy(view_mask),
                    mode=torchvision.io.ImageReadMode.GRAY,
                ).to(device)
                mask = view_mask.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.uint8)
            else:
                mask = None
            
            if frames_array is None:
                h, w = frame.shape[:2]
                frames_array = np.empty((n_frame, h, w, 3), dtype=np.uint8)
                masks_array = np.empty((n_frame, h, w, 1), dtype=np.uint8)
            
            frames_array[fid] = frame
            masks_array[fid] = mask if rd_annots is not None else 255
            
            del view_img, frame
            if rd_annots is not None:
                del view_mask, mask
        
        # 使用 ultrafast 预设加速
        encoded_video = encode_frames_to_video(
            frames_array,
            pix_fmt="rgb24",
            codec="libx265",
            preset="ultrafast",  # 最快，文件稍大
            x265_params="crf=18:pools=+",
            container="hevc",
        )
        encoded_bytes = np.frombuffer(encoded_video, dtype=np.uint8)
        
        encoded_mask_video = encode_frames_to_video(
            masks_array,
            pix_fmt="gray",
            codec="libx265",
            preset="ultrafast",
            x265_params="lossless=1:pools=+",
            container="hevc",
        )
        encoded_mask_bytes = np.frombuffer(encoded_mask_video, dtype=np.uint8)
        
        # 返回序列化后的结果（多进程必须）
        return rgb_vidx, encoded_bytes.tobytes(), encoded_mask_bytes.tobytes()
        
    finally:
        del frames_array, masks_array
        rd.close() if hasattr(rd, 'close') else None
        if rd_annots:
            rd_annots.close() if hasattr(rd_annots, 'close') else None
        malloc_trim()


def get_all_frames_data(smc_path, annots_path=None, save_path=None, gpu_ids=None, num_workers=None):
    """
    多进程并行编码加速
    """
    print("===========================================================")
    print(f"=== Multi-Process CPU Encoding ===")
    print("===========================================================")
    
    # 读取元数据（主进程）
    rd_temp = SMCReader(smc_path)
    n_frame = rd_temp.get_Camera_5mp_info()['num_frame']
    # rd_temp.close()
    
    # 处理校准数据（主进程）
    if annots_path and os.path.exists(annots_path):
        rd_annots_temp = SMCReader(annots_path)
        rgb_cam_dict = rd_annots_temp.get_Calibration_all()
        # rd_annots_temp.close()
    else:
        rd_temp = SMCReader(smc_path)
        rgb_cam_dict = rd_temp.get_Calibration_all()
        # rd_temp.close()
    
    if rgb_cam_dict is None:
        raise ValueError("Cannot get calibration data")
    
    calibs = {}
    for rgb_vidx in range(48):
        calib = rgb_cam_dict[f"{rgb_vidx:02d}"]
        RT = calib["RT"]
        w2c = np.linalg.inv(RT)
        img_size = [int(v) for v in rd_temp.get_Camera_5mp_info()['resolution']]
        calibs[f"{rgb_vidx:02d}"] = {
            "K": np.asarray(calib["K"], dtype=float).tolist(),
            "w2c": np.asarray(w2c, dtype=float).tolist(),
            "dist": np.asarray(calib["D"], dtype=float).tolist(),
            "img_size": img_size,
        }
    
    device_id = gpu_ids[0] if gpu_ids else (0 if torch.cuda.is_available() else -1)
    num_workers = num_workers or min(cpu_count(), 8)  # 默认使用 8 核
    
    print(f"Using {num_workers} workers for encoding...")
    
    os.makedirs(save_path, exist_ok=True)
    zip_path = os.path.join(save_path, "videos.npz")
    mask_zip_path = os.path.join(save_path, "masks.npz")
    zip_file = zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED)
    mask_zip_file = zipfile.ZipFile(mask_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED)
    
    # 准备任务参数
    tasks = [(smc_path, annots_path, i, n_frame, device_id) for i in range(48)]
    
    try:
        # 多进程池
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_camera, tasks),
                total=48,
                desc="Encoding cameras"
            ))
        
        # 写入 zip
        for cam_idx, video_bytes, mask_bytes in results:
            if cam_idx is not None:
                cam_key = f"{cam_idx:02d}"
                # 从 bytes 恢复 numpy array
                video_arr = np.frombuffer(video_bytes, dtype=np.uint8)
                mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8)
                
                _write_npy_to_zip(zip_file, video_arr, f"{cam_key}.npy")
                _write_npy_to_zip(mask_zip_file, mask_arr, f"{cam_key}.npy")
                
                del video_bytes, mask_bytes, video_arr, mask_arr
                malloc_trim()
                
    finally:
        zip_file.close()
        mask_zip_file.close()
        malloc_trim()
    
    json.dump(calibs, open(os.path.join(save_path, "cameras.json"), "w"), indent=4)
    print(f"Done! Results saved to {save_path}")


def _write_npy_to_zip(zip_file, array, arcname):
    buffer = io.BytesIO()
    np.save(buffer, array)
    zip_file.writestr(arcname, buffer.getvalue())
    buffer.close()
    del buffer


if __name__ == "__main__":
    id = "0022_10"
    smc_path = f"./data_smc/{id}.smc"
    annots_path = f"./data_smc/{id}_annots.smc"
    
    get_all_frames_data(
        smc_path=smc_path,
        annots_path=annots_path if os.path.exists(annots_path) else None,
        save_path=f"./data_cps/{id}",
        gpu_ids=[0],
        num_workers=4,  # 根据 CPU 核心数调整，建议 4-8
    )
    
# # 使用示例
# if __name__ == "__main__":
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     main_dir = os.path.join(base_dir, "dna_rendering_part2_main")
#     annots_dir = os.path.join(base_dir, "dna_rendering_part2_annotations")
#     out_dir = os.path.join(base_dir, "dna_rendering_part2_cps")

#     if not os.path.isdir(main_dir):
#         raise FileNotFoundError(f"Main dir not found: {main_dir}")

#     os.makedirs(out_dir, exist_ok=True)

#     main_files = [f for f in os.listdir(main_dir) if f.endswith(".smc")]
#     main_files.sort()

#     for filename in main_files:
#         sample_id = os.path.splitext(filename)[0]
#         main_path = os.path.join(main_dir, filename)
#         annots_path = os.path.join(annots_dir, f"{sample_id}_annots.smc")
#         sample_out_dir = os.path.join(out_dir, sample_id)
#         if os.path.isdir(sample_out_dir):
#             print(f"Skip {sample_id}: output already exists at {sample_out_dir}")
#             continue

#         rd = SMCReader(main_path)
#         rd_annots = SMCReader(annots_path) if os.path.exists(annots_path) else None

#         get_all_frames_data(
#             rd,
#             rd_annots,
#             save_path=sample_out_dir,
#             gpu_ids=[0],
#             save_mode="npz_stream",
#         )