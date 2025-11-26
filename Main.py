import mediapipe as mp
import cv2
import copy
import os
from Utilities import *
from BodyDetector import HandDetector
from PreProcessing import HandNormalizer, combine_hands_features
from LiveInference import LiveGestureInference
#from PhillipsHueWrapperModule import HueController
from KinectWrapper import KinectManager

# ----------------------------
# CONFIG
# ----------------------------
BRIDGE_IP = "192.168.178.206"       # Replace with your bridge IP
USERNAME = "NdzQVUnpZAsG21NTQOS932ilAKYTX2UFdWwNZ4gF"       # Replace with your Hue API username
TARGET_LIGHT = "plafond"         # Name of your Hue lamp to control
GESTURE_CONF_THRESHOLD = 0.65    # Minimum confidence to trigger an action
COOLDOWN_FRAMES = 60             # Avoid rapid re-triggering (about 0.5s at 40 FPS)


def main():
    # Initialize Kinect
    k = KinectManager()
    pTime = 0

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


    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    while True:
        rgb = k.get_rgb()
        if rgb is not None:
            img = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            debug_img = copy.deepcopy(img)
            hands, debug_img = detector.findHands(rgb, draw=True)

            gesture_one, conf_one = None, 0.0
            if onehand_inference:
                _, gesture_one, conf_one = onehand_inference.predict_frame(hands, debug_img)
                if gesture_one:
                    cv2.putText(debug_img, f"{gesture_one} ({conf_one:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

             # FPS overlay
                pTime, fps = update_fps(pTime, debug_img, draw=True)

                show = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

                if rgb is not None:
                    cv2.imshow("Camera Feed", rgb)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

if __name__ == "__main__":
    main()
