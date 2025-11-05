import os
import csv
import glob
import json
import shutil
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# Config
# ----------------------------
OUTPUT_DIR = "processed_dataset"
STATIC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "static")
DYNAMIC_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dynamic")
RAW_DYNAMIC_DIR = os.path.join(DYNAMIC_OUTPUT_DIR, "raw")
RAW_STATIC_DIR = os.path.join(STATIC_OUTPUT_DIR, "raw")
LEGACY_STATIC_DIR = "dataset"

os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(DYNAMIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DYNAMIC_DIR, exist_ok=True)
os.makedirs(RAW_STATIC_DIR, exist_ok=True)


def _migrate_legacy_static_csvs():
    """Move CSVs from the legacy flat dataset/ folder into the structured tree."""

    if not os.path.isdir(LEGACY_STATIC_DIR):
        return

    legacy_files = sorted(glob.glob(os.path.join(LEGACY_STATIC_DIR, "*.csv")))
    if not legacy_files:
        return

    print("Migrating legacy static CSVs into processed_dataset/static/raw ...")

    for src_path in legacy_files:
        base_name = os.path.basename(src_path)
        hand_mode = "twohand" if "twohand" in base_name else "onehand"
        dest_dir = os.path.join(RAW_STATIC_DIR, hand_mode)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, base_name)

        if os.path.exists(dest_path):
            print(f"  Skipping {base_name}: already present at {dest_path}")
            continue

        shutil.move(src_path, dest_path)
        print(f"  Moved {base_name} -> {dest_path}")

    try:
        if not os.listdir(LEGACY_STATIC_DIR):
            os.rmdir(LEGACY_STATIC_DIR)
    except OSError:
        pass


_migrate_legacy_static_csvs()

OUTPUT_FILE_ONEHAND = os.path.join(STATIC_OUTPUT_DIR, "gesture_dataset_onehand.npz")
OUTPUT_FILE_TWOHAND = os.path.join(STATIC_OUTPUT_DIR, "gesture_dataset_twohand.npz")

LABEL_MAP_FILE_ONEHAND = os.path.join(STATIC_OUTPUT_DIR, "label_map_onehand.json")
LABEL_MAP_FILE_TWOHAND = os.path.join(STATIC_OUTPUT_DIR, "label_map_twohand.json")

NORM_PARAMS_FILE_ONEHAND = os.path.join(STATIC_OUTPUT_DIR, "norm_params_onehand.json")
NORM_PARAMS_FILE_TWOHAND = os.path.join(STATIC_OUTPUT_DIR, "norm_params_twohand.json")

DYNAMIC_DATASET_FILE = os.path.join(DYNAMIC_OUTPUT_DIR, "gesture_dataset_dynamic.npz")
LABEL_MAP_DYNAMIC_FILE = os.path.join(DYNAMIC_OUTPUT_DIR, "label_map_dynamic.json")
NORM_PARAMS_DYNAMIC_FILE = os.path.join(DYNAMIC_OUTPUT_DIR, "norm_params_dynamic.json")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
TARGET_DYNAMIC_FRAMES = 60

# ----------------------------
# Sanity check for ratios
# ----------------------------
total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if not np.isclose(total_ratio, 1.0):
    raise ValueError(f"Train/Val/Test ratios must sum to 1.0, got {total_ratio}")


# ----------------------------
# Helper: load CSVs and prepare features (static dataset)
# ----------------------------
def _gather_static_csv_files(hand_type_filter: str) -> List[str]:
    base_dir = os.path.join(RAW_STATIC_DIR, hand_type_filter)
    if not os.path.isdir(base_dir):
        return []

    pattern = os.path.join(base_dir, "**", "*.csv")
    return sorted(glob.glob(pattern, recursive=True))


def load_csv_files(hand_type_filter: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    csv_files = _gather_static_csv_files(hand_type_filter)

    if len(csv_files) == 0:
        print(
            f"No CSV files found for hand type '{hand_type_filter}' in "
            f"'{os.path.join(RAW_STATIC_DIR, hand_type_filter)}'."
        )
        return np.empty((0,)), np.empty((0,)), {}

    print(f"Loaded CSV files for {hand_type_filter}: {[os.path.basename(f) for f in csv_files]}")

    gesture_names = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in csv_files]
    unique_gestures = sorted(set(gesture_names))
    label_map = {name: idx for idx, name in enumerate(unique_gestures)}

    gesture_counts = {name: 0 for name in unique_gestures}
    X, y = [], []

    for path in csv_files:
        gesture_name = os.path.splitext(os.path.basename(path))[0].split("_")[0]
        label_idx = label_map[gesture_name]

        with open(path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                features = [float(row[col]) for col in reader.fieldnames if col not in ("timestamp", "label")]
                X.append(features)
                y.append(label_idx)
                gesture_counts[gesture_name] += 1

    print(f"Number of samples per gesture for {hand_type_filter}:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map


# ----------------------------
# Helper: split dataset
# ----------------------------
def split_dataset(X: np.ndarray, y: np.ndarray):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE, stratify=y
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# Helper: numeric normalization (static)
# ----------------------------
def normalize_dataset(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_val_norm, X_test_norm, mean, std


# ----------------------------
# Main pipeline: generate dataset for a hand type (static)
# ----------------------------
def prepare_static_dataset(hand_type_filter: str, output_file: str, label_file: str, norm_file: str):
    X, y, label_map = load_csv_files(hand_type_filter)
    if len(X) == 0:
        print(f"No data found for {hand_type_filter}. Skipping.")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Dataset shapes for {hand_type_filter}:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")

    X_train, X_val, X_test, mean, std = normalize_dataset(X_train, X_val, X_test)

    np.savez(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    print(f"{hand_type_filter.capitalize()} dataset saved to {output_file}")

    label_payload = {
        "dataset_type": "static",
        "hand_mode": hand_type_filter,
        "label_to_index": label_map,
    }
    with open(label_file, "w") as f:
        json.dump(label_payload, f, indent=2)
    print(f"Label map saved to {label_file}")

    norm_payload = {
        "dataset_type": "static",
        "hand_mode": hand_type_filter,
        "feature_dim": int(X_train.shape[1]),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(norm_file, "w") as f:
        json.dump(norm_payload, f, indent=2)
    print(f"Normalization parameters saved to {norm_file}")

    print("Label mapping:")
    for gesture, idx in label_map.items():
        print(f"  {gesture}: {idx}")


# ----------------------------
# Dynamic dataset helpers
# ----------------------------
def _extract_string(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def load_dynamic_clips() -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[str], List[str]]:
    if not os.path.isdir(RAW_DYNAMIC_DIR):
        print(f"Dynamic raw directory '{RAW_DYNAMIC_DIR}' does not exist.")
        return [], [], [], []

    clip_files = sorted(glob.glob(os.path.join(RAW_DYNAMIC_DIR, "**", "*.npz"), recursive=True))
    if len(clip_files) == 0:
        print("No dynamic clips found.")
        return [], [], [], []

    sequences: List[np.ndarray] = []
    timestamps: List[Optional[np.ndarray]] = []
    labels: List[str] = []
    hand_modes: List[str] = []

    for file_path in clip_files:
        with np.load(file_path, allow_pickle=True) as data:
            frames = np.asarray(data.get("frames", []), dtype=np.float32)
            if frames.ndim == 1:
                # handle accidental flattening
                frames = frames.reshape(-1, frames.size)
            label = _extract_string(data.get("label", ""))
            hand_mode = _extract_string(data.get("hand_mode", "onehand"))
            ts = data.get("timestamps")
            ts_array = None
            if ts is not None:
                ts_array = np.asarray(ts, dtype=np.float32)
            sequences.append(frames)
            timestamps.append(ts_array)
            labels.append(label)
            hand_modes.append(hand_mode)

    return sequences, timestamps, labels, hand_modes


def time_normalize_sequence(
    sequence: np.ndarray, target_frames: int, timestamps: Optional[np.ndarray]
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError("Expected sequence with shape (frames, features)")

    num_frames, feature_dim = sequence.shape
    if num_frames == 0:
        return np.zeros((target_frames, feature_dim), dtype=np.float32)

    if timestamps is not None and len(timestamps) == num_frames:
        time_index = timestamps.astype(np.float32) - float(timestamps[0])
        if np.ptp(time_index) < 1e-6:
            orig_idx = np.linspace(0, num_frames - 1, num_frames, dtype=np.float32)
        else:
            orig_idx = time_index
    else:
        orig_idx = np.linspace(0, num_frames - 1, num_frames, dtype=np.float32)

    if np.ptp(orig_idx) < 1e-6:
        # Degenerate case: all indices equal
        orig_idx[-1] = orig_idx[0] + 1.0

    target_idx = np.linspace(orig_idx[0], orig_idx[-1], target_frames, dtype=np.float32)
    interpolated = np.empty((target_frames, feature_dim), dtype=np.float32)
    for dim in range(feature_dim):
        interpolated[:, dim] = np.interp(target_idx, orig_idx, sequence[:, dim])

    return interpolated


def split_dynamic_dataset(
    X: np.ndarray, y: np.ndarray, hand_modes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_temp, y_train, y_temp, modes_train, modes_temp = train_test_split(
        X, y, hand_modes, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE, stratify=y
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test, modes_val, modes_test = train_test_split(
        X_temp,
        y_temp,
        modes_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, modes_train, modes_val, modes_test


def normalize_dynamic_dataset(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def flatten(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(arr.shape[0], -1)

    train_flat = flatten(X_train)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std < 1e-6] = 1.0

    def apply(arr: np.ndarray) -> np.ndarray:
        flat = flatten(arr)
        normalized = (flat - mean) / std
        return normalized.reshape(arr.shape)

    return apply(X_train), apply(X_val), apply(X_test), mean, std


def prepare_dynamic_dataset():
    sequences, timestamps, labels, hand_modes = load_dynamic_clips()
    if len(sequences) == 0:
        print("No dynamic data found. Skipping dynamic dataset preparation.")
        return

    feature_dims = {seq.shape[1] for seq in sequences if seq.ndim == 2 and seq.shape[0] > 0}
    if not feature_dims:
        print("Dynamic clips do not contain valid frames. Skipping dynamic dataset preparation.")
        return
    if len(feature_dims) > 1:
        raise ValueError(
            "Dynamic clips contain inconsistent feature dimensions. Ensure all clips are recorded with the same hand mode."
        )

    normalized_sequences = []
    for seq, ts in zip(sequences, timestamps):
        normalized_sequences.append(time_normalize_sequence(seq, TARGET_DYNAMIC_FRAMES, ts))

    X = np.stack(normalized_sequences).astype(np.float32)
    feature_dim = X.shape[2]
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in labels], dtype=np.int32)
    hand_modes_arr = np.array(hand_modes)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        modes_train,
        modes_val,
        modes_test,
    ) = split_dynamic_dataset(X, y, hand_modes_arr)

    print("Dynamic dataset shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")

    X_train, X_val, X_test, mean, std = normalize_dynamic_dataset(X_train, X_val, X_test)

    np.savez(
        DYNAMIC_DATASET_FILE,
        X_train=X_train,
        y_train=y_train,
        hand_modes_train=modes_train,
        X_val=X_val,
        y_val=y_val,
        hand_modes_val=modes_val,
        X_test=X_test,
        y_test=y_test,
        hand_modes_test=modes_test,
    )
    print(f"Dynamic dataset saved to {DYNAMIC_DATASET_FILE}")

    label_payload = {
        "dataset_type": "dynamic",
        "label_to_index": label_map,
        "hand_modes": sorted(set(hand_modes)),
    }
    with open(LABEL_MAP_DYNAMIC_FILE, "w") as f:
        json.dump(label_payload, f, indent=2)
    print(f"Dynamic label map saved to {LABEL_MAP_DYNAMIC_FILE}")

    norm_payload = {
        "dataset_type": "dynamic",
        "target_num_frames": TARGET_DYNAMIC_FRAMES,
        "feature_dim": int(feature_dim),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(NORM_PARAMS_DYNAMIC_FILE, "w") as f:
        json.dump(norm_payload, f, indent=2)
    print(f"Dynamic normalization parameters saved to {NORM_PARAMS_DYNAMIC_FILE}")


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    prepare_static_dataset("onehand", OUTPUT_FILE_ONEHAND, LABEL_MAP_FILE_ONEHAND, NORM_PARAMS_FILE_ONEHAND)
    prepare_static_dataset("twohand", OUTPUT_FILE_TWOHAND, LABEL_MAP_FILE_TWOHAND, NORM_PARAMS_FILE_TWOHAND)
    prepare_dynamic_dataset()
