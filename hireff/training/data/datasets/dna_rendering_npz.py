import os
import os.path as osp
import json
import zipfile
import numpy as np
import random
import re
import logging
import torch
import torch.distributed as dist
import itertools
import torchvision
from glob import glob
from torch.utils.data import Dataset
from hireff.utils.load_fn import convert_extrinsics_to_relative_tensor, adjust_intrinsic, adjust_intrinsic_batch


def dna_collate_fn(batch):
    """
    Custom collate function for the DNA dataset.
    batch: List of samples, each being the return value of read_dna_npz_entry.
    """
    # Initialize result dictionary
    collated_batch = {
        'image_bytes': [],
        'masks': [],
        'intrinsics': [],
        'extrinsics': [],
        # 'bg_color': [],
        'supervise_image_bytes': [],
        'supervise_masks': [],
        'supervise_intrinsics': [],
        'supervise_extrinsics': []
    }

    # Iterate over each sample in the batch
    for sample in batch:
        image_bytes, masks, intrinsics, extrinsics = sample['image_bytes'], sample['masks'], sample['intrinsics'], \
        sample['extrinsics']

        # For image_bytes and bg_color, append directly to list (no stacking)
        collated_batch['image_bytes'].append(image_bytes)

        # For other tensors, collect first then stack
        # collated_batch['bg_color'].append(sample['bg_color'])
        collated_batch['masks'].append(masks)
        collated_batch['intrinsics'].append(intrinsics)
        collated_batch['extrinsics'].append(extrinsics)

        # Process supervision view data
        collated_batch['supervise_image_bytes'].append(sample['supervise_image_bytes'])
        collated_batch['supervise_masks'].append(sample['supervise_masks'])
        collated_batch['supervise_intrinsics'].append(sample['supervise_intrinsics'])
        collated_batch['supervise_extrinsics'].append(sample['supervise_extrinsics'])

    # Stack stackable tensors
    # collated_batch['bg_color'] = torch.stack(collated_batch['bg_color'], dim=0)[0]
    collated_batch['masks'] = torch.stack(collated_batch['masks'], dim=0)
    collated_batch['intrinsics'] = torch.stack(collated_batch['intrinsics'], dim=0)
    collated_batch['extrinsics'] = torch.stack(collated_batch['extrinsics'], dim=0)
    collated_batch['supervise_masks'] = torch.stack(collated_batch['supervise_masks'], dim=0)
    collated_batch['supervise_intrinsics'] = torch.stack(collated_batch['supervise_intrinsics'], dim=0)
    collated_batch['supervise_extrinsics'] = torch.stack(collated_batch['supervise_extrinsics'], dim=0)

    return collated_batch


def read_dna_npz_entry(npz_path: str, view_ids: list[int]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_bytes = []
    masks = []
    intrinsics = []
    extrinsics = []

    with np.load(npz_path, allow_pickle=True) as archive:
        for view in view_ids:
            data_key = f"view_{view:02d}"
            if data_key not in archive:
                raise KeyError(f"{data_key} not found in {npz_path}.")
            entry = archive[data_key].item()

            mask = torchvision.io.decode_image(torch.from_numpy(entry["mask"]),
                                               mode=torchvision.io.ImageReadMode.GRAY).float() / 255.0

            image_bytes.append(torch.from_numpy(entry["image"]))
            masks.append(mask)
            intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))
            extrinsics.append(torch.from_numpy(entry["extrinsic"]).to(dtype=torch.float32))
            
    masks = torch.stack(masks, dim=0)  # (n_views, H, W)
    intrinsics = torch.stack(intrinsics, dim=0)  # (n_views, 3, 3)
    extrinsics = torch.stack(extrinsics, dim=0)  # (n_views, 4, 4)

    return image_bytes, masks, intrinsics, extrinsics


class DnaRenderingDatasetNpz(Dataset):
    def __init__(
            self,
            data_root: str,
            min_frames: int = 1,
            target_size: int = 2048,
            sr_target_size: int = 2048,
            frame_numbers: list | None = None,
            split: str = "train",
            n_views_supervise: int = 2
    ):
        super().__init__()
        self.data_root = data_root
        self.sr_target_size = sr_target_size
        self.split = split
        self.frame_numbers = frame_numbers
        self.samples = []  # List to store all samples
        self.supervise_view_groups = None
        self.n_views_supervise = n_views_supervise  # Select N view pairs for supervision
        self.target_size = target_size
        self.origin_width = 2048
        self.origin_height = 2048

        # Initialize view groups (1-48 step 6 pick 8, divided into 4 groups)
        base_views = list(range(1, 48, 3))
        self.view_groups = [
            [f"{v:02d}" for v in base_views[::4][:4]],  # 1, 13, 25, 37
            [f"{v:02d}" for v in base_views[1::4][:4]], # 4, 16, 28, 40
            [f"{v:02d}" for v in base_views[2::4][:4]], # 7, 19, 31, 43
            [f"{v:02d}" for v in base_views[3::4][:4]], # 10, 22, 34, 46
        ]

        # if self.n_views_supervise > 0:
        #     # Generate random supervision view groups
        #     base_views = list(range(0, 24))
        #     all_combinations = list(itertools.combinations(base_views, self.n_views_supervise))
        #     valid_combinations = [combo for combo in all_combinations if
        #                           len(combo) == 1 or max(combo) - min(combo) >= 3]

        #     self.supervise_view_groups = []
        #     for combo in valid_combinations:
        #         relative_combo = [v + 24 for v in combo]
        #         group = [f"{v:02d}" for v in combo] + [f"{v:02d}" for v in relative_combo]
        #         self.supervise_view_groups.append(group)

        # Discover all valid sequences and frames
        self._discover_samples(min_frames)

        # Split data based on train/val
        if len(self.samples) > 10:
            if split == "train":
                self.samples = self.samples[:int(len(self.samples) * 0.95)]
            else:  # val
                self.samples = self.samples[int(len(self.samples) * 0.95):]
        else:
            if split == "train":
                self.samples = self.samples
            else:  # val
                self.samples = self.samples

        logging.info(f"Loaded {len(self.samples)} valid samples for {split}")

    @staticmethod
    def _dist_rank_world() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _discover_samples_check(self, min_frames):
        """Discover all valid sequence and frame combinations."""
        rank, world_size = self._dist_rank_world()
        local_samples = []

        seq_dirs = [
            seq_dir
            for seq_dir in sorted(os.listdir(self.data_root))
            if re.match(r"^\d{4}_\d{2}$", seq_dir)
        ]

        # Shard sequences across GPUs to avoid redundant traversal by each process.
        if world_size > 1:
            seq_dirs = seq_dirs[rank::world_size]

        for seq_dir in seq_dirs:

            seq_path = osp.join(self.data_root, seq_dir)
            logging.debug(f"Processing sequence: {seq_dir}")

            cached_valid_frames = self._try_load_valid_frames_cache(seq_path, min_frames)
            if cached_valid_frames is not None:
                if len(cached_valid_frames) >= min_frames:
                    for frame in cached_valid_frames:
                        local_samples.append({
                            "seq_dir": seq_dir,
                            "frame": frame,
                        })
                continue

            if self.frame_numbers:
                frame_numbers = self.frame_numbers
            else:
                frame_numbers = sorted([
                    int(f.split("_")[1].split(".")[0])
                    for f in os.listdir(seq_path)
                    if f.startswith("frame_") and f.endswith(".npz")
                ])

            # Validate frame completeness
            valid_frames = []
            key_list = ["mask", "image", "intrinsic", "extrinsic"]
            for frame in frame_numbers:
                valid = True
                npz_path = osp.join(seq_path, f"frame_{int(frame):04d}.npz")
                if not osp.exists(npz_path):
                    continue
                # Check all required view groups
                try:
                    with np.load(npz_path, allow_pickle=True) as archive:
                        for view_group in self.view_groups:
                            for view in range(1, 48, 1):
                                data_key = f"view_{int(view):02d}"
                                if data_key not in archive:
                                    valid = False
                                    break
                                entry = archive[data_key].item()
                                # Validate key_list in the entry
                                if any(entry.get(key, None) is None for key in key_list):
                                    valid = False
                                    break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                except (zipfile.BadZipFile, OSError, ValueError, EOFError):
                    logging.warning("Skip broken npz file: %s", npz_path)
                    valid = False

                if valid:
                    valid_frames.append(frame)
                else:
                    print(f"Invalid frame {frame} in sequence {seq_dir} due to missing/broken data.")
                    logging.debug("Invalid frame %s in sequence %s due to missing/broken data.", frame, seq_dir)

            print(f"Rank {rank}: Valid sequence {seq_dir} at {seq_path}")

            # Record valid frames
            if len(valid_frames) >= min_frames:
                self._save_valid_frames_cache(seq_path, min_frames, valid_frames)
                for frame in valid_frames:
                    local_samples.append({
                        "seq_dir": seq_dir,
                        "frame": frame,
                    })

        if world_size > 1:
            gathered_samples = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_samples, local_samples)
            self.samples = []
            for part in gathered_samples:
                self.samples.extend(part)
            self.samples.sort(key=lambda x: (x["seq_dir"], x["frame"]))
        else:
            self.samples = local_samples

        logging.info(
            f"Discovered {len(self.samples)} valid samples "
            f"(rank {rank}/{world_size}, local {len(local_samples)})"
        )

    def _validation_cache_path(self, seq_path: str) -> str:
        return osp.join(seq_path, ".hireff_dna_valid_frames_cache.json")

    def _validation_cache_key(self, min_frames: int) -> dict:
        return {
            "version": 1,
            "min_frames": int(min_frames),
            "frame_numbers": self.frame_numbers,
            "n_views_supervise": int(self.n_views_supervise),
        }

    def _try_load_valid_frames_cache(self, seq_path: str, min_frames: int) -> list[int] | None:
        cache_path = self._validation_cache_path(seq_path)
        if not osp.exists(cache_path):
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None

        if payload.get("cache_key") != self._validation_cache_key(min_frames):
            return None

        valid_frames = payload.get("valid_frames")
        if not isinstance(valid_frames, list):
            return None

        return [int(frame) for frame in valid_frames]

    def _save_valid_frames_cache(self, seq_path: str, min_frames: int, valid_frames: list[int]):
        cache_path = self._validation_cache_path(seq_path)
        payload = {
            "cache_key": self._validation_cache_key(min_frames),
            "valid_frames": [int(frame) for frame in sorted(valid_frames)],
        }
        tmp_path = f"{cache_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True)
            os.replace(tmp_path, cache_path)
        except Exception:
            logging.exception("Failed to write validation cache: %s", cache_path)
            if osp.exists(tmp_path):
                os.remove(tmp_path)
    
    def _discover_samples(self, min_frames):
        self.samples = glob(f"{self.data_root}/**/frame*.npz", recursive=True)

        logging.info(f"Discovered {len(self.samples)} valid samples")

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Get sample by index."""
        sample = self.samples[idx % len(self.samples)]
        return self.load_sample(sample)

    def load_sample(self, npz_path):
        """Load a single sample."""

        # Randomly select a view group
        view_group = random.choice(self.view_groups)
        view_group = [(int(_view) + random.randint(-1, 1) * 3) % 48 for _view in view_group]
        view_group_copy = view_group.copy()
        random.shuffle(view_group)

        image_bytes, masks, intrinsics, extrinsics = read_dna_npz_entry(npz_path, view_group)

        intrinsics[:, :2] *= (self.target_size / self.origin_width)

        base_c2w = torch.linalg.inv(extrinsics[0])
        extrinsics = [extrinsics[_idx] @ base_c2w for _idx in range(extrinsics.shape[0])]
        extrinsics = torch.stack(extrinsics, dim=0)

        # Convert to numpy array
        data = {
            "image_bytes": image_bytes,
            "masks": masks,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

        if True:
            # view_group_supervise = random.choice(self.supervise_view_groups)
            # random.shuffle(view_group_supervise)
            EXCLUDE_VIEWS = {24, 42}
            view_group_supervise = []
            _n = len(view_group)
            for _idx in range(_n):
                _v1 = view_group_copy[_idx] // 3
                _v2 = view_group_copy[(_idx + 1) % _n] // 3
                if _v1 > _v2:
                    _v2 = _v2 + 16

                # Exclude views 24 and 42 (i.e., view_24 and view_42), which may have anomalies
                _novel = 24
                while _novel in EXCLUDE_VIEWS:
                    _novel = random.randint(_v1+1, _v2-1) * 3
                    _novel = _novel + random.randint(-1, 1)
                    _novel = _novel % 48
                view_group_supervise.append(_novel)
            # print(view_group_copy, view_group_supervise)
            random.shuffle(view_group_supervise)

            sv_image_bytes, sv_masks, sv_intrinsics, sv_extrinsics = read_dna_npz_entry(npz_path, [int(v) for v in view_group_supervise])

            sv_intrinsics[:, :2] *= (self.target_size / self.origin_width)

            sv_extrinsics = [sv_extrinsics[_idx] @ base_c2w for _idx in range(sv_extrinsics.shape[0])]
            sv_extrinsics = torch.stack(sv_extrinsics, dim=0)

            data.update({
                "supervise_image_bytes": sv_image_bytes,
                "supervise_masks": sv_masks,
                "supervise_intrinsics": sv_intrinsics,
                "supervise_extrinsics": sv_extrinsics,
            })
            
        return data


# Usage example
if __name__ == "__main__":
    dataset = DnaRenderingDatasetNpz(
        data_root="/home/china/lab/VGGT_human/dna_rendering/data_processing/data_npz",
        min_frames=24,
        sr_target_size=518
    )

    sample = dataset[0]
    print("Loaded keys:", sample.keys())
    print("Images shape:", sample["images"].shape)  # (4, 3, 512, 512)
    print("Camera K shape:", sample["extrinsic"].shape)  # (4, 3, 3)
