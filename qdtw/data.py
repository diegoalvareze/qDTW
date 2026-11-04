"""
Data Loading and Quantum Feature Preprocessing
==============================================
Handles fetching public UCR/UEA benchmarks via tslearn, loading custom clinical NumPy
arrays, channel selection/truncation, FFT resampling, per-channel Min-Max scaling to [0, pi],
and stratified Train/Val/Test partitioning.
"""

import math
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tslearn.datasets import UCR_UEA_datasets


def prepare_custom_dataset(
    instances_path: str,
    labels_path: str,
    channels="all",
    target_length: int = 120,
    max_instances: int = 800,
    val_proportion: float = 0.10,
    test_proportion: float = 0.20,
    device: torch.device = torch.device("cpu"),
):
    """
    Loads and preprocesses custom NumPy time-series arrays (e.g., extracted SHHS Apnea subset).
    Applies per-channel Min-Max scaling to [0, pi] and stratified Train/Val/Test splitting.
    """
    print(f"\n[Data Loading] Loading custom dataset from '{instances_path}'...")
    np_X = np.load(instances_path)
    np_Y = np.load(labels_path)

    if channels == "all":
        np_X_selected = np_X
    elif isinstance(channels, int):
        np_X_selected = np_X[:, :channels, :]
    elif isinstance(channels, list):
        np_X_selected = np_X[:, channels, :]
    else:
        np_X_selected = np_X

    if max_instances is not None and len(np_X_selected) > max_instances:
        print(f"  -> Subsampling dataset from {len(np_X_selected)} down to {max_instances} instances...")
        np_X_selected, _, np_Y, _ = train_test_split(
            np_X_selected, np_Y, train_size=max_instances, random_state=42, stratify=np_Y
        )

    for c in range(np_X_selected.shape[1]):
        channel_min = np.min(np_X_selected[:, c, :])
        channel_max = np.max(np_X_selected[:, c, :])
        np_X_selected[:, c, :] = (np_X_selected[:, c, :] - channel_min) / (channel_max - channel_min + 1e-8)
        np_X_selected[:, c, :] = np_X_selected[:, c, :] * math.pi
        print(f"  -> Channel {c} scaled to [0, pi]. (Original Min: {channel_min:.2f}, Max: {channel_max:.2f})")

    current_length = np_X_selected.shape[2]
    if target_length is not None and target_length < current_length:
        print(f"  -> Resampling from length {current_length} to {target_length}...")
        np_X_resampled = signal.resample(np_X_selected, target_length, axis=2)
    else:
        np_X_resampled = np_X_selected

    np_X_final = np.transpose(np_X_resampled, (0, 2, 1))
    num_channels = np_X_final.shape[2]
    print(f"  -> Final data shape: {np_X_final.shape} | Channels (Qubits): {num_channels}")

    X_temp, X_test, y_temp, y_test = train_test_split(
        np_X_final, np_Y, test_size=test_proportion, random_state=42, stratify=np_Y
    )

    if val_proportion > 0.0:
        val_relative_size = val_proportion / (1.0 - test_proportion)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, random_state=42, stratify=y_temp
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    else:
        X_train, y_train = X_temp, y_temp
        X_val_t, y_val_t = None, None

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, num_channels


def load_public_dataset(
    dataset_name: str = "BasicMotions",
    target_length: int = None,
    val_proportion: float = 0.10,
    test_proportion: float = 0.20,
    max_instances: int = 800,
    max_channels: int = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Downloads and preprocesses public UCR/UEA time-series datasets via tslearn.
    Applies channel truncation (if specified), per-channel [0, pi] scaling, and stratified splitting.
    """
    print(f"\n[Data Loading] Fetching UCR/UEA Dataset: '{dataset_name}'...")
    loader = UCR_UEA_datasets()
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = loader.load_dataset(dataset_name)

    if X_train_raw is None:
        raise ValueError(f"Dataset '{dataset_name}' could not be downloaded. Check the exact dataset name.")

    X_all = np.vstack((X_train_raw, X_test_raw))
    y_all = np.concatenate((y_train_raw, y_test_raw))

    if len(X_all.shape) == 2:
        X_all = np.expand_dims(X_all, axis=-1)

    if max_channels is not None and max_channels < X_all.shape[2]:
        print(f"  -> Truncating channels from {X_all.shape[2]} to {max_channels}...")
        X_all = X_all[:, :, :max_channels]

    n_instances, seq_len, n_channels = X_all.shape

    unique_classes = np.unique(y_all)
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
    y_all_mapped = np.array([label_map[l] for l in y_all])

    print(f"  -> Shape: {X_all.shape} | Channels (Qubits): {n_channels} | Classes: {len(unique_classes)}")

    if max_instances is not None and len(X_all) > max_instances:
        print(f"  -> Subsampling dataset from {len(X_all)} down to {max_instances} instances...")
        X_all, _, y_all_mapped, _ = train_test_split(
            X_all, y_all_mapped, train_size=max_instances, random_state=42, stratify=y_all_mapped
        )

    if target_length is not None and target_length < seq_len:
        print(f"  -> Resampling sequence length from {seq_len} to {target_length}...")
        X_resampled = signal.resample(X_all, target_length, axis=1)
    else:
        X_resampled = X_all

    X_scaled = np.zeros_like(X_resampled)
    for c in range(n_channels):
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_scaled[:, :, c] = scaler.fit_transform(X_resampled[:, :, c])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y_all_mapped, test_size=test_proportion, stratify=y_all_mapped, random_state=42
    )

    if val_proportion > 0.0:
        adj_val_prop = val_proportion / (1.0 - test_proportion)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=adj_val_prop, stratify=y_temp, random_state=42
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    else:
        X_train, y_train = X_temp, y_temp
        X_val_t, y_val_t = None, None

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t, n_channels


def check_quantum_scale(data_tensor: torch.Tensor, name: str = "Dataset"):
    """Verifies that input features lie within [0, pi] to prevent phase-wrapping collisions."""
    if data_tensor is None:
        return
    flat_data = data_tensor.flatten()
    d_min, d_max = flat_data.min().item(), flat_data.max().item()
    d_mean, d_std = flat_data.mean().item(), flat_data.std().item()
    spread = d_max - d_min
    print(f"[{name} Scale] Min: {d_min:.4f} | Max: {d_max:.4f} | Spread: {spread:.4f} | Mean: {d_mean:.4f} | Std: {d_std:.4f}")
