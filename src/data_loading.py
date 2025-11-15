"""
Data loading utilities for the mmWave MIMO Radar dataset.

Supports:
- Loading per-day train/test .mat files
- Concatenating per-device training files or keeping device partitions
- Train/val split with stratification
- tf.data pipelines with optional per-sample normalization and channel dimension

Assumed shapes (from dataset description):
- Train files per device k contain:
  - mmwave_data_train_k: (N_k, 256, 63)
  - label_train_k: (N_k,) in {0..9}
- Test file per day contains:
  - mmwave_data_test: (N_test, 256, 63)
  - label_test: (N_test,) in {0..9}

Notes:
- We do not ship the dataset in Git. Place files under data/day{0,1,2}/... locally.
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Tuple, Optional, List

import numpy as np
from scipy.io import loadmat


def _find_array_by_name_or_shape(mat: dict, prefer_prefix: List[str], shape_suffix: Tuple[int, int]) -> Optional[np.ndarray]:
	"""
	Try to find an array in a loaded .mat dict by name prefix or by matching trailing shape.
	shape_suffix is expected to be (256, 63) for data arrays.
	"""
	# 1) Prefer named keys
	for pfx in prefer_prefix:
		for key in mat.keys():
			if key.lower().startswith(pfx):
				arr = np.array(mat[key])
				if arr.size > 0:
					return arr
	# 2) Fall back to shape-based search
	for key, val in mat.items():
		if key.startswith("__"):
			continue
		arr = np.array(val)
		if arr.ndim >= 3 and tuple(arr.shape[-2:]) == shape_suffix:
			return arr
	return None


def _find_labels(mat: dict, prefer_prefix: List[str]) -> Optional[np.ndarray]:
	# 1) Name-based
	for pfx in prefer_prefix:
		for key in mat.keys():
			if key.lower().startswith(pfx):
				arr = np.array(mat[key]).squeeze()
				if arr.size > 0:
					return arr
	# 2) Fallback: first 1D array of ints
	for key, val in mat.items():
		if key.startswith("__"):
			continue
		arr = np.array(val).squeeze()
		if arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
			return arr
	return None


def _ensure_xy_shapes(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	x = np.array(x)
	y = np.array(y).squeeze()
	if x.ndim != 3 or x.shape[1:] != (256, 63):
		raise ValueError(f"Expected X shape (N,256,63), got {x.shape}")
	if y.ndim != 1 or x.shape[0] != y.shape[0]:
		raise ValueError(f"Mismatched X/Y shapes: X{ x.shape } vs y{ y.shape }")
	return x.astype(np.float32), y.astype(np.int64)


def load_train_device_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Load one device training .mat file containing mmwave_data_train_k and label_train_k."""
	mat = loadmat(path)
	x = _find_array_by_name_or_shape(mat, ["mmwave_data_train", "x_train", "data_train"], (256, 63))
	y = _find_labels(mat, ["label_train", "y_train", "labels"])
	if x is None or y is None:
		raise ValueError(f"Could not find train arrays in {path}")
	return _ensure_xy_shapes(x, y)


def load_test_file(path: str) -> Tuple[np.ndarray, np.ndarray]:
	"""Load a day test .mat file containing mmwave_data_test and label_test."""
	mat = loadmat(path)
	x = _find_array_by_name_or_shape(mat, ["mmwave_data_test", "x_test", "data_test"], (256, 63))
	y = _find_labels(mat, ["label_test", "y_test", "labels"])
	if x is None or y is None:
		raise ValueError(f"Could not find test arrays in {path}")
	return _ensure_xy_shapes(x, y)


def load_day_train_as_clients(day_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
	"""
	Load training data for a day, returning a mapping device_id -> (X, y).
	day_dir should contain a train_data/ folder with mmwave_data_train_*.mat files.
	"""
	train_glob = os.path.join(day_dir, "train_data", "mmwave_data_train_*.mat")
	files = sorted(glob.glob(train_glob))
	if not files:
		raise FileNotFoundError(f"No training files found at {train_glob}")
	clients = {}
	for f in files:
		dev = os.path.splitext(os.path.basename(f))[0]  # e.g., mmwave_data_train_3
		clients[dev] = load_train_device_file(f)
	return clients


def load_day_train_concatenated(day_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Load and concatenate all device training files for a day.
	Returns X, y, device_ids where device_ids[i] is the integer device index for sample i.
	"""
	clients = load_day_train_as_clients(day_dir)
	x_list, y_list, d_list = [], [], []
	# Try to parse device index from name suffix, default to incremental order
	for idx, (dev, (x, y)) in enumerate(clients.items(), start=1):
		x_list.append(x)
		y_list.append(y)
		d_idx = idx
		# parse trailing _N
		parts = dev.rsplit("_", 1)
		if len(parts) == 2 and parts[1].isdigit():
			d_idx = int(parts[1])
		d_list.append(np.full(shape=(x.shape[0],), fill_value=d_idx, dtype=np.int32))
	X = np.concatenate(x_list, axis=0)
	y = np.concatenate(y_list, axis=0)
	device_ids = np.concatenate(d_list, axis=0)
	return X, y, device_ids


def load_day_test(day_dir: str) -> Tuple[np.ndarray, np.ndarray]:
	test_path = os.path.join(day_dir, "test_data", "mmwave_data_test.mat")
	if not os.path.exists(test_path):
		raise FileNotFoundError(f"Test file not found: {test_path}")
	return load_test_file(test_path)


def train_val_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	from sklearn.model_selection import train_test_split
	xtr, xval, ytr, yval = train_test_split(X, y, test_size=val_ratio, random_state=seed, stratify=y)
	return xtr, xval, ytr, yval


def _normalize_numpy(x: np.ndarray, method: str = "zscore", eps: float = 1e-6) -> np.ndarray:
	if method == "none" or method is None:
		return x
	if method == "zscore":
		mu = x.mean()
		std = x.std()
		return (x - mu) / (std + eps)
	if method == "minmax":
		xmin = x.min()
		xmax = x.max()
		return (x - xmin) / (xmax - xmin + eps)
	if method == "log1p":
		return np.log1p(np.maximum(x, 0.0))
	raise ValueError(f"Unknown normalization method: {method}")


def make_tf_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, normalize: str = "zscore"):
	import tensorflow as tf

	X = X.astype(np.float32)
	y = y.astype(np.int64)

	def _map_fn(sample, label):
		# per-sample normalization
		if normalize and normalize != "none":
			if normalize == "zscore":
				mean = tf.reduce_mean(sample)
				std = tf.math.reduce_std(sample)
				sample = (sample - mean) / (std + 1e-6)
			elif normalize == "minmax":
				mn = tf.reduce_min(sample)
				mx = tf.reduce_max(sample)
				sample = (sample - mn) / (mx - mn + 1e-6)
			elif normalize == "log1p":
				sample = tf.math.log1p(tf.maximum(sample, 0.0))
		# add channel dim -> (256,63,1)
		sample = tf.expand_dims(sample, axis=-1)
		return sample, label

	ds = tf.data.Dataset.from_tensor_slices((X, y))
	if shuffle:
		ds = ds.shuffle(buffer_size=min(10000, X.shape[0]), seed=42, reshuffle_each_iteration=True)
	ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
	ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return ds


def make_client_tf_datasets(day_dir: str, batch_size: int = 32, normalize: str = "zscore") -> Dict[str, object]:
	"""Return dict of device_name -> tf.data.Dataset for training clients."""
	clients = load_day_train_as_clients(day_dir)
	ds_map = {}
	for dev, (x, y) in clients.items():
		ds_map[dev] = make_tf_dataset(x, y, batch_size=batch_size, shuffle=True, normalize=normalize)
	return ds_map


def make_centralized_tf_datasets(day_dir: str, val_ratio: float = 0.2, batch_size: int = 32, normalize: str = "zscore") -> Tuple[object, object, Tuple[np.ndarray, np.ndarray]]:
	"""
	Build (train_ds, val_ds, (X_test, y_test)) for a given day directory.
	"""
	X, y, _ = load_day_train_concatenated(day_dir)
	xtr, xval, ytr, yval = train_val_split(X, y, val_ratio=val_ratio)
	tr_ds = make_tf_dataset(xtr, ytr, batch_size=batch_size, shuffle=True, normalize=normalize)
	val_ds = make_tf_dataset(xval, yval, batch_size=batch_size, shuffle=False, normalize=normalize)
	Xte, yte = load_day_test(day_dir)
	return tr_ds, val_ds, (Xte, yte)

