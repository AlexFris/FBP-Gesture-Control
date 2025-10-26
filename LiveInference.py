import numpy as np
import tensorflow as tf
import json
from PreProcessing import HandNormalizer, combine_hands_features

from HandDetector import HandDetector


# ----------------------------
# Config
# ----------------------------
MODEL_DIR = "trained_model"
DATA_DIR = "processed_dataset"

MODELS = {
    "onehand": {
        "model_path": f"{MODEL_DIR}/gesture_cnn_onehand.h5",
        "norm_params": f"{DATA_DIR}/norm_params_onehand.json",
        "label_map": f"{DATA_DIR}/label_map_onehand.json"
    },
    "twohand": {
        "model_path": f"{MODEL_DIR}/gesture_cnn_twohand.h5",
        "norm_params": f"{DATA_DIR}/norm_params_twohand.json",
        "label_map": f"{DATA_DIR}/label_map_twohand.json"
    }
}


# ----------------------------
# Load model and normalization parameters
# ----------------------------
def load_model_and_params(hand_type="onehand"):
    cfg = MODELS[hand_type]
    model = tf.keras.models.load_model(cfg["model_path"])
    norm_params = json.load(open(cfg["norm_params"]))
    label_map = json.load(open(cfg["label_map"]))
    label_map_inv = {v: k for k, v in label_map.items()}
    return model, norm_params, label_map_inv


# ----------------------------
# Prepare hand input for inference
# ----------------------------
def prepare_input_from_landmarks(landmarks, normalizer, norm_params, hand_label):
    lm_array = np.array(landmarks, dtype=np.float32)

    # Flip X for left hands if model trained on right hand only
    if hand_label == "Left":
        lm_array[:, 0] *= -1

    normalized = normalizer.normalize_hand(lm_array)
    flat_vector = normalized["flat_vector"]

    mean = np.array(norm_params["mean"], dtype=np.float32)
    std = np.array(norm_params["std"], dtype=np.float32)
    std[std < 1e-6] = 1.0
    normed_input = (flat_vector - mean) / std

    return normed_input[np.newaxis, :, np.newaxis]


# ----------------------------
# Main inference function (called from main.py)
# ----------------------------
class LiveGestureInference:
    def __init__(self, hand_type="onehand"):
        self.hand_type = hand_type
        self.model, self.norm_params, self.label_map_inv = load_model_and_params(hand_type)
        self.detector = HandDetector()
        self.normalizer = HandNormalizer(rotation_invariant=True, include_z=False)

    def predict_frame(self, frame):
        """
        Detect hands in a frame and return annotated image and prediction.
        """
        hands, img = self.detector.findHands(frame, draw=True)
        gesture, confidence = None, None

        if self.hand_type == "onehand" and len(hands) == 1:
            hand = hands[0]
            lm_list = hand["lmList"]
            hand_label = hand["type"]

            input_data = prepare_input_from_landmarks(
                lm_list, self.normalizer, self.norm_params, hand_label
            )

            prediction = self.model.predict(input_data, verbose=0)
            pred_label = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            gesture = self.label_map_inv[pred_label]

        elif self.hand_type == "twohand" and len(hands) == 2:
            left = [h for h in hands if h["type"] == "Left"]
            right = [h for h in hands if h["type"] == "Right"]
            if left and right:
                left_data = self.normalizer.normalize_hand(np.array(left[0]["lmList"], dtype=np.float32))
                right_data = self.normalizer.normalize_hand(np.array(right[0]["lmList"], dtype=np.float32))
                combined = combine_hands_features(left_data, right_data)

                mean = np.array(self.norm_params["mean"], dtype=np.float32)
                std = np.array(self.norm_params["std"], dtype=np.float32)
                std[std < 1e-6] = 1.0
                normed_input = (combined - mean) / std
                normed_input = normed_input[np.newaxis, :, np.newaxis]

                prediction = self.model.predict(normed_input, verbose=0)
                pred_label = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                gesture = self.label_map_inv[pred_label]

        return img, gesture, confidence
