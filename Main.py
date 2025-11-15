import mediapipe as mp
import cv2
import copy
import os
from Utilities import *
from HandDetector import HandDetector
from PreProcessing import HandNormalizer, combine_hands_features
from LiveInference import LiveGestureInference
from PhillipsHueWrapperModule import HueController

# ----------------------------
# CONFIG
# ----------------------------
BRIDGE_IP = "192.168.178.213"       # Replace with your bridge IP
USERNAME = "NdzQVUnpZAsG21NTQOS932ilAKYTX2UFdWwNZ4gF"       # Replace with your Hue API username
TARGET_LIGHT = "plafond"         # Name of your Hue lamp to control
GESTURE_CONF_THRESHOLD = 0.65    # Minimum confidence to trigger an action
COOLDOWN_FRAMES = 60             # Avoid rapid re-triggering (about 0.5s at 40 FPS)


def main():
    args = parse_camera_args()
    cap = setup_camera(device=args.device, width=args.width, height=args.height)
    pTime = 0

    # Initialize Hue bridge
    hue = HueController(BRIDGE_IP, USERNAME)
    hue.list_lights()

    # Initialize detector
    detector = HandDetector()

    # Initialize gesture inference models
    onehand_model_path = "trained_model/gesture_cnn_onehand.h5"
    twohand_model_path = "trained_model/gesture_cnn_twohand.h5"

    if os.path.exists(onehand_model_path):
        onehand_inference = LiveGestureInference(hand_type="onehand")
    else:
        raise FileNotFoundError(f"One-hand model not found: {onehand_model_path}")

    twohand_inference = LiveGestureInference(hand_type="twohand") if os.path.exists(twohand_model_path) else None

    # Track last gesture to prevent flicker
    last_gesture = None
    cooldown_counter = 0

    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    while True:
        success, img = read_frame(cap)
        if not success:
            break

        debug_img = copy.deepcopy(img)
        hands, debug_img = detector.findHands(debug_img, draw=True)

        if len(hands) > 0:
            hand = hands[0]
            lmList = hand["lmList"]

            # Extract coordinates
            thumb_tip = lmList[4][0:2]  # (x, y)
            index_tip = lmList[8][0:2]  # (x, y)

        gesture_one, conf_one = None, 0.0
        if onehand_inference:
            _, gesture_one, conf_one = onehand_inference.predict_frame(hands, debug_img)
            if gesture_one:
                cv2.putText(debug_img, f"{gesture_one} ({conf_one:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # --- GESTURE → ACTION MAPPING ---
        if gesture_one and conf_one > GESTURE_CONF_THRESHOLD:
            if cooldown_counter == 0:  # prevent retriggering too fast
                if gesture_one == "Open":
                    #length, _, _ = detector.findDistance(thumb_tip, index_tip, debug_img)
                    #print(length)
                    hue.turn_on(TARGET_LIGHT)
                    hue.set_brightness(TARGET_LIGHT, 125)
                    print(f"[ACTION] {gesture_one} → Turn ON {TARGET_LIGHT}")
                    cooldown_counter = COOLDOWN_FRAMES
                elif gesture_one == "Fist":
                    hue.turn_off(TARGET_LIGHT)
                    print(f"[ACTION] {gesture_one} → Turn OFF {TARGET_LIGHT}")
                    cooldown_counter = COOLDOWN_FRAMES

        # Decrease cooldown
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # FPS overlay
        pTime, fps = update_fps(pTime, debug_img, draw=True)

        cv2.imshow("Camera Feed", debug_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    release_camera(cap)


if __name__ == "__main__":
    main()
