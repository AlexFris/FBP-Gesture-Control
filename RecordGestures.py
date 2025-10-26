import cv2
import csv
import os
import time
import copy
from datetime import datetime

from HandDetector import HandDetector
from PreProcessing import HandNormalizer, combine_hands_features
from Utilities import parse_camera_args, setup_camera, read_frame, release_camera


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
    os.makedirs("dataset", exist_ok=True)
    file_name = f"dataset/{gesture_label}_{'twohand' if is_two_hands else 'onehand'}.csv"

    # Determine feature size (21 landmarks × 3 coords × 1 or 2 hands)
    feature_dim = 21 * 3 * (2 if is_two_hands else 1)
    header = [f"f{i}" for i in range(feature_dim)] + ["timestamp", "label"]

    if not os.path.exists(file_name):
        with open(file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    print(f"Recording gesture: {gesture_label} ({'two-hand' if is_two_hands else 'one-hand'})")
    print("Press 'c' to capture sample, 's' to start continuous recording, 'q' or 'ESC' to quit.")

    sample_count = 0
    continuous_mode = False
    frame_delay = 0.1  # seconds between continuous captures

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
                if is_two_hands:
                    # Sort so that Left hand is first
                    hands_sorted = sorted(hands, key=lambda h: h["type"] != "Left")
                    norm_left = normalizer.normalize_hand(hands_sorted[0]["lmList"])["flat_vector"]
                    norm_right = normalizer.normalize_hand(hands_sorted[1]["lmList"])["flat_vector"]
                    features = combine_hands_features(
                        {"flat_vector": norm_left, "norm_landmarks": hands_sorted[0]["lmList"]},
                        {"flat_vector": norm_right, "norm_landmarks": hands_sorted[1]["lmList"]}
                    )
                else:
                    norm = normalizer.normalize_hand(hands[0]["lmList"])["flat_vector"]
                    features = norm

                # Validate feature count
                if len(features) != feature_dim:
                    print(f"Invalid feature length ({len(features)}), skipping.")
                    continue

                # Write to CSV
                with open(file_name, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(list(features) + [timestamp, gesture_label])

                sample_count += 1
                print(f"Recorded sample #{sample_count}")
                if continuous_mode:
                    time.sleep(frame_delay)

            except Exception as e:
                print(f"Error capturing frame: {e}")
                continuous_mode = False

    release_camera(cap)
    print(f"Recording session ended. {sample_count} samples saved to {file_name}")


if __name__ == "__main__":
    main()
