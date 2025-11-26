import cv2
import csv
import time
import copy
from datetime import datetime
from pathlib import Path

import numpy as np

from BodyDetector import HandDetector
from PreProcessing import HandNormalizer, combine_hands_features
from Utilities import parse_camera_args, setup_camera, read_frame, release_camera


RAW_DYNAMIC_DIR = Path("processed_dataset") / "dynamic" / "raw"
RAW_STATIC_DIR = Path("processed_dataset") / "static" / "raw"
CLIP_DURATION_SECONDS = 2.0


def _extract_features(hands, normalizer, is_two_hands):
    if is_two_hands:
        hands_sorted = sorted(hands, key=lambda h: h["type"] != "Left")
        norm_left = normalizer.normalize_hand(hands_sorted[0]["lmList"])["flat_vector"]
        norm_right = normalizer.normalize_hand(hands_sorted[1]["lmList"])["flat_vector"]
        features = combine_hands_features(
            {"flat_vector": norm_left, "norm_landmarks": hands_sorted[0]["lmList"]},
            {"flat_vector": norm_right, "norm_landmarks": hands_sorted[1]["lmList"]},
        )
    else:
        norm = normalizer.normalize_hand(hands[0]["lmList"])["flat_vector"]
        features = norm
    return features


def _save_dynamic_clip(gesture_label, is_two_hands, frames, timestamps):
    clip_length = len(frames)
    if clip_length == 0:
        print("Clip contained no valid frames; skipping save.")
        return

    RAW_DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    clip_dir = RAW_DYNAMIC_DIR / gesture_label
    clip_dir.mkdir(parents=True, exist_ok=True)

    file_stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    hand_mode = "twohand" if is_two_hands else "onehand"
    file_path = clip_dir / f"{gesture_label}_{hand_mode}_{file_stem}.npz"

    np.savez(
        file_path,
        frames=np.asarray(frames, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=np.float32),
        label=np.array(gesture_label),
        hand_mode=np.array(hand_mode),
    )
    print(f"Saved dynamic clip with {clip_length} frames to {file_path}")


def main():
    # === User setup ===
    gesture_label = input("Enter gesture label (e.g., fist, open_palm): ").strip()
    hand_mode = input("Record one-hand or two-hand gestures? (1/2): ").strip()
    normalizer = HandNormalizer(include_z=True)  # Include Z coordinates

    # Validate hand mode
    if hand_mode not in ["1", "2"]:
        print("Invalid input. Defaulting to one-hand mode.")
        hand_mode = "1"

    is_two_hands = (hand_mode == "2")

    # === Setup camera and detector ===
    args = parse_camera_args()
    cap = setup_camera(device=args.device, width=args.width, height=args.height)
    detector = HandDetector(maxHands=2)
    pTime = 0

    # === File setup ===
    hand_suffix = "twohand" if is_two_hands else "onehand"

    static_dir = RAW_STATIC_DIR / hand_suffix
    static_dir.mkdir(parents=True, exist_ok=True)
    file_path = static_dir / f"{gesture_label}_{hand_suffix}.csv"

    # Determine feature size (21 landmarks × 3 coords × 1 or 2 hands)
    feature_dim = 21 * 3 * (2 if is_two_hands else 1)
    header = [f"f{i}" for i in range(feature_dim)] + ["timestamp", "label"]

    if not file_path.exists():
        with file_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    print(f"Recording gesture: {gesture_label} ({'two-hand' if is_two_hands else 'one-hand'})")
    print("Press 'c' to capture static sample, 's' to toggle static continuous recording,")
    print("'r' to capture a dynamic clip (~2s), 'q' or 'ESC' to quit.")

    sample_count = 0
    dynamic_clip_count = 0
    continuous_mode = False
    frame_delay = 0.1  # seconds between continuous captures
    recording_dynamic_clip = False
    dynamic_frames = []
    dynamic_timestamps = []
    clip_start_time = 0.0

    # === Main loop ===
    while True:
        success, img = read_frame(cap)
        if not success:
            print("Camera read failed.")
            break

        debug_img = copy.deepcopy(img)
        hands, _ = detector.findHands(debug_img, draw=True)

        # Display info overlay
        cv2.putText(debug_img, f"Gesture: {gesture_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Mode: {'Two-Hand' if is_two_hands else 'One-Hand'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Samples: {sample_count}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Dynamic clips: {dynamic_clip_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if recording_dynamic_clip:
            elapsed = time.time() - clip_start_time
            cv2.putText(debug_img, f"Recording clip: {elapsed:.1f}s", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Gesture Recorder", debug_img)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key in [27, ord('q')]:
            break

        # Toggle continuous recording
        elif key == ord('s'):
            continuous_mode = not continuous_mode
            print(f"{'Started' if continuous_mode else 'Stopped'} continuous recording.")
            time.sleep(0.3)

        elif key == ord('r') and not recording_dynamic_clip:
            recording_dynamic_clip = True
            dynamic_frames = []
            dynamic_timestamps = []
            clip_start_time = time.time()
            print("Started dynamic clip recording.")

        # Manual capture or continuous capture
        elif key == ord('c') or continuous_mode:
            timestamp = datetime.now().isoformat(timespec="milliseconds")

            # === Validation: one-hand or two-hand ===
            if (is_two_hands and len(hands) != 2) or (not is_two_hands and len(hands) < 1):
                print("Incorrect number of hands detected, skipping sample.")
                if continuous_mode:
                    time.sleep(0.2)
                continue

            try:
                features = _extract_features(hands, normalizer, is_two_hands)

                # Validate feature count
                if len(features) != feature_dim:
                    print(f"Invalid feature length ({len(features)}), skipping.")
                    continue

                # Write to CSV
                with file_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(features) + [timestamp, gesture_label])

                sample_count += 1
                print(f"Recorded sample #{sample_count}")
                if continuous_mode:
                    time.sleep(frame_delay)

            except Exception as e:
                print(f"Error capturing frame: {e}")
                continuous_mode = False

        if recording_dynamic_clip:
            current_time = time.time()
            elapsed = current_time - clip_start_time
            if (is_two_hands and len(hands) == 2) or (not is_two_hands and len(hands) >= 1):
                try:
                    features = _extract_features(hands, normalizer, is_two_hands)
                    if len(features) == feature_dim:
                        dynamic_frames.append(features)
                        dynamic_timestamps.append(elapsed)
                except Exception as e:
                    print(f"Error capturing dynamic frame: {e}")

            if elapsed >= CLIP_DURATION_SECONDS:
                recording_dynamic_clip = False
                _save_dynamic_clip(gesture_label, is_two_hands, dynamic_frames, dynamic_timestamps)
                dynamic_clip_count += 1
                dynamic_frames = []
                dynamic_timestamps = []
                time.sleep(0.3)

    release_camera(cap)
    print(f"Recording session ended. {sample_count} samples saved to {file_path}")
    print(f"Dynamic clips captured: {dynamic_clip_count}")


if __name__ == "__main__":
    main()
