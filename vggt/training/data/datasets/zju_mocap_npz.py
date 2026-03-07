import itertools
import logging
import os
import os.path as osp
import random
import re

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset

from vggt.utils.load_fn import adjust_intrinsic_batch, convert_extrinsics_to_relative_tensor


def zju_collate_fn(batch):
	collated_batch = {
		"image_bytes": [],
		"masks": [],
		"intrinsics": [],
		"extrinsics": [],
		"bg_color": [],
		"supervise_image_bytes": [],
		"supervise_masks": [],
		"supervise_intrinsics": [],
		"supervise_extrinsics": [],
	}

	for sample in batch:
		collated_batch["image_bytes"].append(sample["image_bytes"])
		collated_batch["bg_color"].append(sample["bg_color"])
		collated_batch["masks"].append(sample["masks"])
		collated_batch["intrinsics"].append(sample["intrinsics"])
		collated_batch["extrinsics"].append(sample["extrinsics"])

		if "supervise_image_bytes" in sample:
			collated_batch["supervise_image_bytes"].append(sample["supervise_image_bytes"])
			collated_batch["supervise_masks"].append(sample["supervise_masks"])
			collated_batch["supervise_intrinsics"].append(sample["supervise_intrinsics"])
			collated_batch["supervise_extrinsics"].append(sample["supervise_extrinsics"])

	collated_batch["bg_color"] = torch.stack(collated_batch["bg_color"], dim=0)[0]
	collated_batch["masks"] = torch.stack(collated_batch["masks"], dim=0)
	collated_batch["intrinsics"] = torch.stack(collated_batch["intrinsics"], dim=0)
	collated_batch["extrinsics"] = torch.stack(collated_batch["extrinsics"], dim=0)

	if collated_batch["supervise_masks"]:
		collated_batch["supervise_masks"] = torch.stack(collated_batch["supervise_masks"], dim=0)
		collated_batch["supervise_intrinsics"] = torch.stack(collated_batch["supervise_intrinsics"], dim=0)
		collated_batch["supervise_extrinsics"] = torch.stack(collated_batch["supervise_extrinsics"], dim=0)

	return collated_batch


def read_zju_npz_entry(npz_path: str, view_ids: list[int]) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
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
			mask = torchvision.io.decode_image(
				torch.from_numpy(entry["mask"]),
				mode=torchvision.io.ImageReadMode.GRAY,
			).float() / 255.0
   
			image_bytes.append(torch.from_numpy(entry["image"]))
			masks.append(mask)
			intrinsics.append(torch.from_numpy(entry["intrinsic"]).to(dtype=torch.float32))
			extrinsics.append(torch.linalg.inv(torch.from_numpy(entry["extrinsic"])).to(dtype=torch.float32))

	return (
		image_bytes,
		torch.stack(masks, dim=0),
		torch.stack(intrinsics, dim=0),
		torch.stack(extrinsics, dim=0),
	)


class ZjuMocapDatasetNpz(Dataset):
	def __init__(
		self,
		data_root: str,
		min_frames: int = 1,
		target_size: int = 2048,
		sr_target_size: int = 2048,
		frame_numbers: list[int] | None = None,
		split: str = "train",
		n_views_supervise: int = 2,
	):
		super().__init__()
		self.data_root = data_root
		self.sr_target_size = sr_target_size
		self.split = split
		self.frame_numbers = frame_numbers
		self.samples = []
		self.n_views_supervise = n_views_supervise
		self.target_size = target_size
		self.origin_width = 2048
		self.origin_height = 2048

		self._discover_samples(min_frames)

		if len(self.samples) > 10:
			split_idx = int(len(self.samples) * 0.95)
			if split == "train":
				self.samples = self.samples[:split_idx]
			else:
				self.samples = self.samples[split_idx:]

		logging.info("Loaded %d valid samples for %s", len(self.samples), split)

	def _discover_samples(self, min_frames: int, need_valid: bool = False):
		self.samples = []

		for seq_dir in sorted(os.listdir(self.data_root)):
			if not re.match(r"^CoreView_\d+$", seq_dir):
				continue

			seq_path = osp.join(self.data_root, seq_dir)
			if not osp.isdir(seq_path):
				continue

			if self.frame_numbers:
				frame_numbers = self.frame_numbers
			else:
				frame_numbers = sorted(
					int(f.split("_")[1].split(".")[0])
					for f in os.listdir(seq_path)
					if f.startswith("frame_") and f.endswith(".npz")
				)

			if need_valid:
				valid_frames = []
				key_list = ["mask", "image", "intrinsic", "extrinsic"]
				for frame in frame_numbers:
					npz_path = osp.join(seq_path, f"frame_{int(frame):04d}.npz")
					if not osp.exists(npz_path):
						continue
					with np.load(npz_path, allow_pickle=True) as archive:
						available_views = self._extract_available_views_from_archive(archive)
						primary_groups = self._build_primary_view_groups(available_views)
						if not primary_groups:
							continue

						valid = True
						for view in primary_groups[0]:
							entry = archive[f"view_{view:02d}"].item()
							if any(entry.get(key, None) is None for key in key_list):
								valid = False
								break
						if valid:
							valid_frames.append(frame)

				if len(valid_frames) >= min_frames:
					for frame in valid_frames:
						self.samples.append({"seq_dir": seq_dir, "frame": frame})
			else:
				if len(frame_numbers) >= min_frames:
					for frame in frame_numbers:
						self.samples.append({"seq_dir": seq_dir, "frame": frame})

		logging.info("Discovered %d valid samples", len(self.samples))

	@staticmethod
	def _extract_available_views_from_archive(archive) -> list[int]:
		available_views = []
		for key in archive.files:
			match = re.match(r"^view_(\d{2})$", key)
			if match:
				available_views.append(int(match.group(1)))
		return sorted(available_views)

	def _get_available_views(self, npz_path: str) -> list[int]:
		with np.load(npz_path, allow_pickle=True) as archive:
			return self._extract_available_views_from_archive(archive)

	@staticmethod
	def _build_primary_view_groups(available_views: list[int]) -> list[list[int]]:
		available_set = set(available_views)
		groups = []
		for start in range(1, 7):
			group = [start + 6 * step for step in range(4)]
			if all(view in available_set for view in group):
				groups.append(group)

		if groups:
			return groups

		if len(available_views) >= 4:
			return [sorted(random.sample(available_views, 4))]

		return []

	def _build_supervise_view_groups(self, available_views: list[int], primary_group: list[int]) -> list[list[int]]:
		if self.n_views_supervise <= 0:
			return []

		available_set = set(available_views)

		def opposite_view(view_id: int) -> int:
			return ((view_id - 1 + 12) % 24) + 1

		pair_bases = []
		for view in available_views:
			opposite = opposite_view(view)
			if opposite in available_set and view < opposite:
				pair_bases.append(view)

		if len(pair_bases) >= self.n_views_supervise:
			pair_groups = []
			for combo in itertools.combinations(pair_bases, self.n_views_supervise):
				if len(combo) > 1 and max(combo) - min(combo) < 3:
					continue

				views = []
				for base in combo:
					views.extend([base, opposite_view(base)])
				pair_groups.append(views)

			different_pair_groups = [
				group for group in pair_groups if set(group) != set(primary_group)
			]
			if different_pair_groups:
				return different_pair_groups
			if pair_groups:
				return pair_groups

		required_views = self.n_views_supervise * 2
		if len(available_views) < required_views:
			return []

		all_combinations = itertools.combinations(available_views, required_views)
		valid_combinations = [
			list(combo)
			for combo in all_combinations
			if len(combo) == 1 or max(combo) - min(combo) >= 3
		]

		different_combinations = [
			combo for combo in valid_combinations if set(combo) != set(primary_group)
		]
		return different_combinations if different_combinations else valid_combinations

	def __len__(self):
		return len(self.samples) * 20

	def __getitem__(self, idx):
		sample = self.samples[idx % len(self.samples)]
		return self.load_sample(sample["seq_dir"], sample["frame"])

	def load_sample(self, seq_dir: str, frame: int):
		base_path = osp.join(self.data_root, seq_dir)
		npz_path = osp.join(base_path, f"frame_{frame:04d}.npz")

		available_views = self._get_available_views(npz_path)
		view_groups = self._build_primary_view_groups(available_views)
		if not view_groups:
			raise ValueError(f"No valid primary view group in {npz_path}.")

		view_group = random.choice(view_groups)
		random.shuffle(view_group)

		image_bytes, masks, intrinsics, extrinsics = read_zju_npz_entry(npz_path, view_group)

		adjusted_intrinsic = adjust_intrinsic_batch(
			intrinsics,
			self.origin_width,
			self.origin_height,
			self.target_size,
			self.target_size,
		)
		relatived_extrinsics = convert_extrinsics_to_relative_tensor(extrinsics)

		data = {
			"image_bytes": image_bytes,
			"masks": masks,
			"intrinsics": adjusted_intrinsic,
			"extrinsics": relatived_extrinsics,
			"bg_color": torch.rand(3),
		}

		supervise_view_groups = self._build_supervise_view_groups(available_views, view_group)
		if supervise_view_groups:
			view_group_supervise = random.choice(supervise_view_groups)
			random.shuffle(view_group_supervise)

			sv_image_bytes, sv_masks, sv_intrinsics, sv_extrinsics = read_zju_npz_entry(npz_path, view_group_supervise)
			adjusted_sv_intrinsic = adjust_intrinsic_batch(
				sv_intrinsics,
				self.origin_width,
				self.origin_height,
				self.target_size,
				self.target_size,
			)

			all_extrinsics = torch.cat([extrinsics, sv_extrinsics], dim=0)
			relatived_all_extrinsics = convert_extrinsics_to_relative_tensor(all_extrinsics)
			relatived_sv_extrinsics = relatived_all_extrinsics[-len(sv_extrinsics):]

			data.update(
				{
					"supervise_image_bytes": sv_image_bytes,
					"supervise_masks": sv_masks,
					"supervise_intrinsics": adjusted_sv_intrinsic,
					"supervise_extrinsics": relatived_sv_extrinsics,
				}
			)

		return data

