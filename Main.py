import mediapipe as mp
import cv2
import copy
import os
 # test
from Utilities import *
from HandDetector import HandDetector
from PreProcessing import HandNormalizer, combine_hands_features
from LiveInference import LiveGestureInference


def main():
    # Camera setup
    args = parse_camera_args()
    cap = setup_camera(device=args.device, width=args.width, height=args.height)
    pTime = 0

    # Initialize detector and normalizer
    detector = HandDetector()
    normalizer = HandNormalizer(rotation_invariant=True)

    # Initialize one-hand inference (required)
    onehand_model_path = "trained_model/gesture_cnn_onehand.h5"
    if os.path.exists(onehand_model_path):
        onehand_inference = LiveGestureInference(hand_type="onehand")
    else:
        raise FileNotFoundError(f"One-hand model not found: {onehand_model_path}")

    # Initialize two-hand inference (optional)
    twohand_model_path = "trained_model/gesture_cnn_twohand.h5"
    if os.path.exists(twohand_model_path):
        twohand_inference = LiveGestureInference(hand_type="twohand")
    else:
        twohand_inference = None
        print("Two-hand model not found — skipping two-hand inference.")

    while True:
        success, img = read_frame(cap)
        if not success:
            break

        debug_img = copy.deepcopy(img)

        # One-hand inference
        if onehand_inference:
            img, gesture_one, conf_one = onehand_inference.predict_frame(debug_img)
            if gesture_one:
                cv2.putText(debug_img, f"One-hand: {gesture_one} ({conf_one:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Two-hand inference (if available)
        if twohand_inference:
            img, gesture_two, conf_two = twohand_inference.predict_frame(debug_img)
            if gesture_two:
                cv2.putText(debug_img, f"Two-hand: {gesture_two} ({conf_two:.2f})",
                            (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # FPS overlay
        pTime, fps = update_fps(pTime, debug_img, draw=True)

        # Display
        cv2.imshow("Camera Feed", debug_img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    release_camera(cap)


if __name__ == "__main__":
    main()
