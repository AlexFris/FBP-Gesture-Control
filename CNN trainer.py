import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from datetime import datetime

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "processed_dataset"
MODEL_DIR = "trained_model"
LOG_FILE = os.path.join(MODEL_DIR, "train_log.json")
os.makedirs(MODEL_DIR, exist_ok=True)

DATASETS = {
    "onehand": {
        "npz": os.path.join(DATA_DIR, "gesture_dataset_onehand.npz"),
        "label_map": os.path.join(DATA_DIR, "label_map_onehand.json"),
        "norm_params": os.path.join(DATA_DIR, "norm_params_onehand.json"),
        "model_path": os.path.join(MODEL_DIR, "gesture_cnn_onehand.h5"),
    },
    "twohand": {
        "npz": os.path.join(DATA_DIR, "gesture_dataset_twohand.npz"),
        "label_map": os.path.join(DATA_DIR, "label_map_twohand.json"),
        "norm_params": os.path.join(DATA_DIR, "norm_params_twohand.json"),
        "model_path": os.path.join(MODEL_DIR, "gesture_cnn_twohand.h5"),
    }
}

# ----------------------------
# Helper: load dataset
# ----------------------------
def load_dataset(path):
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        return None
    data = np.load(path)
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"]
    )

# ----------------------------
# Helper: timestamp check
# ----------------------------
def should_retrain(dataset_path, model_path):
    """Return True if dataset is newer than model or model doesn't exist."""
    if not os.path.exists(model_path):
        return True
    dataset_mtime = os.path.getmtime(dataset_path)
    model_mtime = os.path.getmtime(model_path)
    return dataset_mtime > model_mtime

# ----------------------------
# Helper: save training log
# ----------------------------
def log_training(hand_type, test_acc, epochs_run, dataset_path):
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = {}

    logs[hand_type] = {
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_accuracy": round(float(test_acc), 4),
        "epochs_run": epochs_run,
        "dataset_file": os.path.basename(dataset_path),
        "dataset_mtime": datetime.fromtimestamp(os.path.getmtime(dataset_path)).strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

# ----------------------------
# Build CNN model
# ----------------------------
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------------
# Train model
# ----------------------------
def train_model(hand_type):
    cfg = DATASETS[hand_type]
    dataset_path = cfg["npz"]
    model_path = cfg["model_path"]

    # Skip retraining if not needed
    if not os.path.exists(dataset_path):
        print(f"Skipping {hand_type}: dataset not found.")
        return
    if not should_retrain(dataset_path, model_path):
        print(f"{hand_type.capitalize()} model is up to date — skipping retrain.")
        return

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_path)
    num_classes = len(json.load(open(cfg["label_map"])))

    # Reshape input for CNN
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    input_shape = (X_train.shape[1], 1)

    print(f"\nTraining {hand_type} model:")
    print(f"  Input shape: {input_shape}")
    print(f"  Classes: {num_classes}")
    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    model = build_cnn_model(input_shape, num_classes)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = callbacks.ModelCheckpoint(model_path, save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{hand_type.capitalize()} model test accuracy: {test_acc:.4f}")

    model.save(model_path)
    print(f"Saved {hand_type} model to {model_path}")

    # ---- Export TensorFlow Lite version ----
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path = model_path.replace(".h5", ".tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"✅ Exported TFLite model to {tflite_path}")
    except Exception as e:
        print(f"⚠️ TFLite conversion failed: {e}")

    # Log the training info
    epochs_run = len(history.history['loss'])
    log_training(hand_type, test_acc, epochs_run, dataset_path)
    print(f"Training log updated for {hand_type} model.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    for hand_type in ["onehand", "twohand"]:
        train_model(hand_type)
