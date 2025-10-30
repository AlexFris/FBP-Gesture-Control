import numpy as np
import tensorflow as tf
import json
from PreProcessing import HandNormalizer, combine_hands_features

# ----------------------------
# Smoothing config
# ----------------------------
SMOOTH = {
    "mode": "ema",              # "ema" or "majority"
    "alpha": 0.25,              # responsiveness (lower = smoother, higher = faster)
    "min_stable_frames": 3,     # require N consistent frames before switching gesture
    "window": 20                # only used in majority mode
}

MODEL_DIR = "trained_model"
DATA_DIR = "processed_dataset"

MODELS = {
    "onehand": {
        "model_path": f"{MODEL_DIR}/gesture_cnn_onehand.h5",
        "tflite_path": f"{MODEL_DIR}/gesture_cnn_onehand.tflite",
        "norm_params": f"{DATA_DIR}/norm_params_onehand.json",
        "label_map": f"{DATA_DIR}/label_map_onehand.json"
    },
    "twohand": {
        "model_path": f"{MODEL_DIR}/gesture_cnn_twohand.h5",
        "tflite_path": f"{MODEL_DIR}/gesture_cnn_twohand.tflite",
        "norm_params": f"{DATA_DIR}/norm_params_twohand.json",
        "label_map": f"{DATA_DIR}/label_map_twohand.json"
    }
}


# ----------------------------
# Smoothing helper (integrated here instead of separate file)
# ----------------------------
class GestureSmoother:
    def __init__(self, num_classes):
        self.mode = SMOOTH["mode"]
        self.alpha = SMOOTH["alpha"]
        self.min_stable_frames = SMOOTH["min_stable_frames"]
        self.window = SMOOTH["window"]
        self.num_classes = num_classes

        self._ema = np.ones(self.num_classes, dtype=np.float32) / self.num_classes
        self._current_label = None
        self._pending_label = None
        self._pending_count = 0
        self._history = []

    def update(self, prediction):
        if self.mode == "ema":
            self._ema = (1 - self.alpha) * self._ema + self.alpha * prediction
            cand_label = int(np.argmax(self._ema))
            cand_conf  = float(self._ema[cand_label])
        else:
            label = int(np.argmax(prediction))
            self._history.append(label)
            if len(self._history) > self.window:
                self._history.pop(0)
            cand_label = max(set(self._history), key=self._history.count)
            cand_conf = self._history.count(cand_label) / len(self._history)

        if self._current_label is None:
            self._current_label = cand_label
        elif cand_label != self._current_label:
            if self._pending_label == cand_label:
                self._pending_count += 1
                if self._pending_count >= self.min_stable_frames:
                    self._current_label = cand_label
                    self._pending_label = None
                    self._pending_count = 0
            else:
                self._pending_label = cand_label
                self._pending_count = 1

        return self._current_label, float(self._ema[self._current_label]) if self.mode == "ema" else cand_conf


def load_model_and_params(hand_type="onehand"):
    cfg = MODELS[hand_type]
    if tf.io.gfile.exists(cfg["tflite_path"]):
        interpreter = tf.lite.Interpreter(model_path=cfg["tflite_path"])
        interpreter.allocate_tensors()
        is_tflite = True
        print(f"✅ Loaded TFLite model: {cfg['tflite_path']}")
    else:
        interpreter = tf.keras.models.load_model(cfg["model_path"])
        is_tflite = False
        print(f"⚠️ Loaded Keras .h5: {cfg['model_path']}")

    norm_params = json.load(open(cfg["norm_params"]))
    label_map = json.load(open(cfg["label_map"]))
    label_map_inv = {v: k for k, v in label_map.items()}

    return interpreter, norm_params, label_map_inv, is_tflite


class LiveGestureInference:
    def __init__(self, hand_type="onehand"):
        self.hand_type = hand_type
        self.model, self.norm_params, self.label_map_inv, self.is_tflite = load_model_and_params(hand_type)
        self.normalizer = HandNormalizer(rotation_invariant=True, include_z=False)

        num_classes = len(self.label_map_inv)
        self.smoother = GestureSmoother(num_classes)

        if self.is_tflite:
            self.input_details = self.model.get_input_details()
            self.output_details = self.model.get_output_details()

    def _predict(self, input_data):
        if self.is_tflite:
            self.model.set_tensor(self.input_details[0]['index'], input_data)
            self.model.invoke()
            return self.model.get_tensor(self.output_details[0]['index'])[0]
        return self.model.predict(input_data, verbose=0)[0]

    def predict_frame(self, hands, img):
        if self.hand_type == "onehand" and len(hands) == 1:
            hand = hands[0]
            input_data = self._prepare_input_for_one_hand(hand)
            prediction = self._predict(input_data)

        elif self.hand_type == "twohand" and len(hands) == 2:
            input_data = self._prepare_input_for_two_hands(hands)
            if input_data is None:
                return img, None, None
            prediction = self._predict(input_data)

        else:
            return img, None, None

        smoothed_label, smoothed_conf = self.smoother.update(prediction)
        gesture = self.label_map_inv[smoothed_label]
        return img, gesture, smoothed_conf

    def _prepare_input_for_one_hand(self, hand):
        lm_list = hand["lmList"]
        hand_label = hand["type"]
        lm = np.array(lm_list, dtype=np.float32)
        if hand_label == "Left":
            lm[:, 0] *= -1
        norm = self.normalizer.normalize_hand(lm)["flat_vector"]
        return self._normalize_vector(norm)

    def _prepare_input_for_two_hands(self, hands):
        left = [h for h in hands if h["type"] == "Left"]
        right = [h for h in hands if h["type"] == "Right"]
        if not left or not right:
            return None
        left_data = self.normalizer.normalize_hand(np.array(left[0]["lmList"], dtype=np.float32))
        right_data = self.normalizer.normalize_hand(np.array(right[0]["lmList"], dtype=np.float32))
        return self._normalize_vector(combine_hands_features(left_data, right_data))

    def _normalize_vector(self, vec):
        mean = np.array(self.norm_params["mean"], dtype=np.float32)
        std = np.array(self.norm_params["std"], dtype=np.float32)
        std[std < 1e-6] = 1.0
        return ((vec - mean) / std)[np.newaxis, :, np.newaxis]
