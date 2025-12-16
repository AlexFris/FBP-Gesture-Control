import os
from Utilities import *
from BodyDetector import HandDetector, PoseDetector
from LiveInference import LiveGestureInference
from HueWrapper import HueController
from KinectWrapper import KinectManager
from OSCWrapper import TDOSC
from SpotifyWrapper import SpotifyController
from GestureStateWrapper import GestureStabilizer, InteractionStateMachine
from ActionDispatcher import ActionDispatcher
import threading
import time

# ----------------------------
# CONFIG
# ----------------------------

# Hue
#BRIDGE_IP = os.getenv("BRIDGE_IP") # Loads the envirinment variable or replace with your bridge IP
#USERNAME = "958lVW8ABIsBp5r6fOpabdgkiZq3Kh-4PoerQ1bD"       # Replace with your Hue API username
TARGET_LIGHT = "Hue color lamp"         # Name of your Hue lamp to control

# Inference model paths
onehand_model_path = "trained_model/gesture_cnn_onehand.h5"
twohand_model_path = "trained_model/gesture_cnn_twohand.h5"


def main():
    # ----------------------------
    # Initialization
    # ----------------------------

    # Initialize Kinect
    k = KinectManager()
    pTime = 0

    # Initialize OSC connection
    osc = TDOSC(ip="192.168.1.49", port=8009, verbose=True)
    osc.ping()

    # Initialize Hue connection
    hue = HueController()

    # Spotify controller
    sp = SpotifyController()
    # Check Spotify active device
    sp.set_active_device()  # Make sure we have an active device

    # Initialize detectors
    hand_detector = HandDetector()
    pose_detector = PoseDetector()

    # Initialize GestureStateWrapper Classes
    gesture_stabilizer = GestureStabilizer()
    state_machine = InteractionStateMachine()

    spotify_thread = threading.Thread(
        target=spotify_osc_loop,
        args=(sp, osc),
        daemon=True
    )
    spotify_thread.start()

    # ActionDispatcher
    dispatcher = ActionDispatcher(
        hue=hue,
        spotify = sp,
        osc = osc,
        light_name=TARGET_LIGHT
    )


    #load gesture models
    if os.path.exists(onehand_model_path):
        onehand_inference = LiveGestureInference(hand_type="onehand")
    else:
        raise FileNotFoundError(f"One-hand model not found: {onehand_model_path}")

    if os.path.exists(twohand_model_path):
        twohand_inference = LiveGestureInference(hand_type="twohand")
    else:
        print(f"One-hand model not found: {onehand_model_path}")


    control_active = 0


    # ----------------------------
    # MAIN LOOP
    # ----------------------------
    while True:
        # ----------------------------
        # CAPTURE AND DEBUG IMG SETUP
        # ----------------------------
        raw_img = k.get_rgb()
        if raw_img is None:
            continue # Skip this frame safely

        # Canonical images
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        debug_img = raw_img.copy()  # BGR, safe to draw on

        # ----------------------------
        # DETECTION/TRACKING
        # ----------------------------
        # Run detection models
        hands, debug_img = hand_detector.findHands(debug_img, draw=True)
        pose_detector.findPose(raw_img_rgb, draw=False)
        pose_lm = pose_detector.results.pose_landmarks if pose_detector.results else None

        # Arms check
        arms = pose_detector.resolve_handedness(
            raw_img_rgb,
            pose_lm,
            hand_detector.results
        )

        # Upright state tracking
        pose_detector.update_arm_upright_states(arms)
        left_up = pose_detector.left_arm_upright
        right_up = pose_detector.right_arm_upright
        #print(left_up, right_up)

        # Draw arms
        debug_img = pose_detector.draw_arms(debug_img, arms)

        """
        # Handedness debug
        debug_img = pose_detector.debug_handedness(
            rgb,
            arms,
            hand_detector.results,
            pose_lm
        )
        
        # Arm state Debug
        cv2.putText(rgb, f"L angle: {pose_detector.debug_left_angle:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(rgb, f"L upright: {pose_detector.left_arm_upright}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(rgb, f"R angle: {pose_detector.debug_right_angle:.1f}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(rgb, f"R upright: {pose_detector.right_arm_upright}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        """

        # Choose which hand to use for one-hand inference
        preferred_type = None
        if right_up:
            preferred_type = "Right"
        elif left_up:
            preferred_type = "Left"

        hands_for_onehand = []
        if preferred_type is not None:
            hands_for_onehand = [h for h in hands if h.get("type") == preferred_type]

        # If we didn't find the preferred hand (e.g., detector mislabeled), optionally fall back:
        if preferred_type is not None and len(hands_for_onehand) == 0 and len(hands) > 0:
            # fallback: pick the first detected hand
            hands_for_onehand = [hands[0]]

        # ----------------------------
        # GESTURE CLASSIFICATION
        # ----------------------------
        gesture_one, conf_one = None, 0.0
        if onehand_inference:
            _, gesture_one, conf_one = onehand_inference.predict_frame(hands_for_onehand, debug_img)
            if gesture_one:
                cv2.putText(debug_img, f"{gesture_one} ({conf_one:.2f})",
                (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # ----------------------------
        # SPATIAL CONTROL
        # ----------------------------
        point_direction = None

        if gesture_one == "Point" and len(hands_for_onehand) == 1:
            hand = hands_for_onehand[0]
            lm = hand["lmList"]

            wrist_x = lm[0][0]
            index_tip_x = lm[8][0]

            if index_tip_x < wrist_x:
                point_direction = "LEFT"
            else:
                point_direction = "RIGHT"
        cv2.putText(debug_img, f"{point_direction}",
                    (110, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        # ----------------------------
        # Controlling
        # ----------------------------
        if left_up or right_up:
            control_active = 1
        else:
            control_active = 0
        osc.send_bool("/control/active", control_active)

        detected = 0
        if len(hands) > 0 and control_active < 1:
            cv2.putText(debug_img, "I see hands",
                        (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            detected = 1
            osc.send_bool("/control/detected", detected)
        else:
            detected = 0
            osc.send_bool("/control/detected", detected)


        predicted_gesture = gesture_one  # string or None
        confidence = conf_one  # float 0.0–1.0

        activated, deactivated, changed = gesture_stabilizer.update(predicted_gesture, confidence)

        """
        # Print event-based transitions
        if activated:
            print(f"[EVENT] Gesture ACTIVATED → {activated}")

        if deactivated:
            print(f"[EVENT] Gesture DEACTIVATED → {deactivated}")

        if changed:
            old, new = changed
            print(f"[EVENT] Gesture CHANGED → {old} → {new}")
        """

        # Always check the current stabilized gesture
        current = gesture_stabilizer.stable_gesture

        # Update control_active flag (from your arm-upright logic)
        control_actions = state_machine.update_control_active(control_active)
        dispatcher.dispatch(control_actions)

        cv2.putText(debug_img, f"ST.Gesture {current}", (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Mode: {state_machine.mode}, Submode: {state_machine.submode}", (10, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        actions = state_machine.process_gesture_event(
            activated=activated,
            deactivated=deactivated,
            changed=changed
        )

        dispatcher.dispatch(actions)

        # ----------------------------
        # DRAWING
        # ----------------------------
        HueStateColour = 0
        HueStateText = ""
        if hue.is_available():
            HueStateText = "Hue active"
            HueStateColour = (0, 255, 0) # green
        else:
            HueStateText = "Hue inactive"
            HueStateColour = (0, 0, 255) # red
        cv2.putText(
            debug_img,
            HueStateText,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HueStateColour,  # green
            2
        )

        SpotifyStateColour = 0
        SpotifyStateText = ""
        if sp.is_available():
            SpotifyStateText = "Spotify active"
            SpotifyStateColour = (0, 255, 0) # green
        else:
            SpotifyStateText = "Spotify inactive"
            SpotifyStateColour = (0, 0, 255) # red
        cv2.putText(
            debug_img,
            SpotifyStateText,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            SpotifyStateColour,  # green
            2
        )

        # FPS overlay
        pTime, fps = update_fps(pTime, debug_img, draw=True)

        #show = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

        if raw_img is not None:
            cv2.imshow("Debug Feed", debug_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

def spotify_osc_loop(spotify, osc, update_interval=0.5):
    while True:
        try:
            info = spotify.get_current_track_info()

            if info:
                osc.send_string("/spotify/title", info["title"] or "")
                osc.send_string("/spotify/artist", ", ".join(info["artists"]) if info["artists"] else "")
                osc.send_string("/spotify/album", info["album"] or "")
                osc.send_string("/spotify/image_url", info["image_url"] or "")

                osc.send_int("/spotify/duration_ms", info["duration_ms"] or 0)
                osc.send_int("/spotify/progress_ms", info["progress_ms"] or 0)
                osc.send_bool("/spotify/is_playing", info["is_playing"])

            else:
                # Explicit "nothing playing" state
                osc.send_bool("/spotify/is_playing", False)

        except Exception as e:
            print(f"[Spotify OSC] Error: {e}")

        time.sleep(update_interval)

if __name__ == "__main__":
    main()
