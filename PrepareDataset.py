import os
import csv
import numpy as np
import json
from sklearn.model_selection import train_test_split

# ----------------------------
# Config
# ----------------------------
DATASET_DIR = "dataset"

# Output directory
OUTPUT_DIR = "processed_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE_ONEHAND = os.path.join(OUTPUT_DIR, "gesture_dataset_onehand.npz")
OUTPUT_FILE_TWOHAND = os.path.join(OUTPUT_DIR, "gesture_dataset_twohand.npz")

LABEL_MAP_FILE_ONEHAND = os.path.join(OUTPUT_DIR, "label_map_onehand.json")
LABEL_MAP_FILE_TWOHAND = os.path.join(OUTPUT_DIR, "label_map_twohand.json")

NORM_PARAMS_FILE_ONEHAND = os.path.join(OUTPUT_DIR, "norm_params_onehand.json")
NORM_PARAMS_FILE_TWOHAND = os.path.join(OUTPUT_DIR, "norm_params_twohand.json")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# ----------------------------
# Sanity check for ratios
# ----------------------------
total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
if not np.isclose(total_ratio, 1.0):
    raise ValueError(f"Train/Val/Test ratios must sum to 1.0, got {total_ratio}")

# ----------------------------
# Helper: load CSVs and prepare features
# ----------------------------
def load_csv_files(hand_type_filter):
    X, y = [], []

    csv_files = sorted([f for f in os.listdir(DATASET_DIR)
                        if f.endswith(".csv") and hand_type_filter in f])

    if len(csv_files) == 0:
        print(f"No CSV files found for hand type '{hand_type_filter}'")
        return np.array(X), np.array(y), {}

    print(f"Loaded CSV files for {hand_type_filter}: {csv_files}")

    gesture_names = [f.split("_")[0] for f in csv_files]
    unique_gestures = sorted(list(set(gesture_names)))
    label_map = {name: idx for idx, name in enumerate(unique_gestures)}

    # Counter for number of samples per gesture
    gesture_counts = {name: 0 for name in unique_gestures}

    for f in csv_files:
        gesture_name = f.split("_")[0]
        label_idx = label_map[gesture_name]
        path = os.path.join(DATASET_DIR, f)

        with open(path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                features = [float(row[col]) for col in reader.fieldnames if col not in ("timestamp", "label")]
                X.append(features)
                y.append(label_idx)
                gesture_counts[gesture_name] += 1

    # Log number of samples per gesture
    print(f"Number of samples per gesture for {hand_type_filter}:")
    for gesture, count in gesture_counts.items():
        print(f"  {gesture}: {count}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), label_map

# ----------------------------
# Helper: split dataset
# ----------------------------
def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_STATE, stratify=y
    )
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio_adjusted), random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ----------------------------
# Helper: numeric normalization
# ----------------------------
def normalize_dataset(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-6] = 1.0  # avoid division by zero

    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std

    norm_params = {"mean": mean.tolist(), "std": std.tolist()}
    return X_train_norm, X_val_norm, X_test_norm, norm_params

# ----------------------------
# Main pipeline: generate dataset for a hand type
# ----------------------------
def prepare_dataset(hand_type_filter, output_file, label_file, norm_file):
    X, y, label_map = load_csv_files(hand_type_filter)
    if len(X) == 0:
        print(f"No data found for {hand_type_filter}. Skipping.")
        return

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print(f"Dataset shapes for {hand_type_filter}:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}")

    # Numeric normalization
    X_train, X_val, X_test, norm_params = normalize_dataset(X_train, X_val, X_test)

    # Save dataset
    np.savez(output_file,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    print(f"{hand_type_filter.capitalize()} dataset saved to {output_file}")

    # Save label map
    with open(label_file, "w") as f:
        json.dump(label_map, f)
    print(f"Label map saved to {label_file}")

    # Save normalization parameters
    with open(norm_file, "w") as f:
        json.dump(norm_params, f)
    print(f"Normalization parameters saved to {norm_file}")

    print("Label mapping:")
    for gesture, idx in label_map.items():
        print(f"  {gesture}: {idx}")

# ----------------------------
# Run for one-hand and two-hand gestures separately
# ----------------------------
if __name__ == "__main__":
    prepare_dataset("onehand", OUTPUT_FILE_ONEHAND, LABEL_MAP_FILE_ONEHAND, NORM_PARAMS_FILE_ONEHAND)
    prepare_dataset("twohand", OUTPUT_FILE_TWOHAND, LABEL_MAP_FILE_TWOHAND, NORM_PARAMS_FILE_TWOHAND)
