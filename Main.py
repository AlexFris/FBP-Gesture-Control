import os
from Utilities import *
from BodyDetector import HandDetector, PoseDetector
from LiveInference import LiveGestureInference
from HueWrapper import HueController
from KinectWrapper import KinectManager
from OSCWrapper import TDOSC
from SpotifyWrapper import SpotifyController
from GestureStateWrapper import InteractionStateMachine, PointIntentTracker, PerHandGestureTracker, compute_gesture_events, HandIdentityTracker
from ActionDispatcher import ActionDispatcher
import threading
import time

# ----------------------------
# CONFIG
# ----------------------------

# Hue
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
    pTime = 0   # Used for FPS calculation

    # Initialize OSC connection 192.168.1.49
    osc = TDOSC(ip="192.168.1.49", port=8009, verbose=True)
    osc.ping()  # Sends a test message of "1"

    # Initialize Hue connection
    hue = HueController()

    # Spotify controller
    sp = SpotifyController()
    sp.set_active_device()  # Make sure we have an active device

    spotify_thread = threading.Thread(
        target=spotify_osc_loop,
        args=(sp, osc),
        daemon=True
    )
    spotify_thread.start()  # Runs the snippet of code at the end of this file, running spotify info

    # Initialize detectors
    hand_detector = HandDetector()
    pose_detector = PoseDetector()
    hand_id_tracker = HandIdentityTracker(
        base_match_distance_px=100,
        max_missing_frames=15,
        velocity_weight=0.6
    )
    hand_inference_by_id = {}

    # Initialize GestureStateWrapper classes
    per_hand_tracker = PerHandGestureTracker(
        frames_needed_up=7,
        frames_needed_down=5,
        confidence_threshold=0.6
    )
    state_machine = InteractionStateMachine()

    prev_selected = None

    # ActionDispatcher
    dispatcher = ActionDispatcher(
        hue=hue,
        spotify = sp,
        osc = osc,
        light_name=TARGET_LIGHT
    )


    point_tracker = PointIntentTracker(
        dwell_frames=8,
        release_frames=3,
        grace_frames=20
    )

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
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)  # Now in the correct type for Mediapipe
        debug_img = raw_img.copy()  # BGR, safe to draw on with OpenCV

        # ----------------------------
        # DETECTION/TRACKING
        # ----------------------------
        # Run detection models
        hands, debug_img = hand_detector.findHands(debug_img, draw=True)
        identified_hands = hand_id_tracker.update(hands)
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

        # Checks if a hand is raised and if so communicates that state over OSC
        if left_up or right_up:
            control_active = 1
        else:
            control_active = 0
        osc.send_bool("/control/active", control_active)

        # Communicates over OSC if there is a hand in frame
        detected = 0
        if len(hands) > 0 and control_active < 1:
            detected = 1
            osc.send_bool("/control/detected", detected)
        else:
            detected = 0
            osc.send_bool("/control/detected", detected)

        # ----------------------------
        # GESTURE CLASSIFICATION (PER HAND)
        # ----------------------------
        hand_gestures = []  # list of dicts: {hand, gesture, confidence}

        for item in identified_hands:
            handID = item["id"]
            hand = item["hand"]

            hand_for_inference = hand.copy()

            # Canonicalize orientation: mirror LEFT hands
            if hand["type"] == "Left":
                hand_for_inference["lmList"] = mirror_landmarks_x(
                    hand["lmList"],
                    image_width=debug_img.shape[1]
                )

            hand_id = item["id"]

            if hand_id not in hand_inference_by_id:
                hand_inference_by_id[hand_id] = LiveGestureInference(hand_type="onehand")

            inference = hand_inference_by_id[hand_id]

            hand_for_inference = hand.copy()

            # Canonicalize orientation
            if hand["type"] == "Left":
                hand_for_inference["lmList"] = mirror_landmarks_x(
                    hand["lmList"],
                    image_width=debug_img.shape[1]
                )

            _, gesture, conf = inference.predict_frame(
                [hand_for_inference],
                debug_img
            )

            hand_gestures.append({
                "id": handID,
                "hand": hand,
                "gesture": gesture,
                "confidence": conf
            })

            # Debug overlay per hand
            if gesture:
                cx, cy = hand["center"]
                cv2.putText(
                    debug_img,
                    f"{handID} {gesture} ({conf:.2f})",
                    (cx - 40, cy - 120),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2
                )

        # ----------------------------
        # PER-HAND STABILIZATION
        # ----------------------------
        per_hand_tracker.update(hand_gestures)

        stable_gestures = []

        for g in hand_gestures:
            hid = g["id"]
            stable = per_hand_tracker.stable(hid)
            if stable is not None:
                stable_gestures.append({
                    "id": hid,
                    "hand": g["hand"],
                    "gesture": stable
                })

        # ----------------------------
        # SELECT AUTHORITATIVE GESTURE (FROM STABILIZED PER HAND)
        # ----------------------------
        selected_hand_id = None
        predicted_gesture = None

        # 1) Point always allowed, prefer right hand
        point_hands = [g for g in stable_gestures if g["gesture"] == "Point"]

        if point_hands:
            right_points = [g for g in point_hands if g["hand"]["type"] == "Right"]
            chosen = right_points[0] if right_points else point_hands[0]

            selected_hand_id = chosen["id"]
            predicted_gesture = "Point"

        # 2) Other gestures require control_active
        elif control_active:
            right_hands = [g for g in stable_gestures if g["hand"]["type"] == "Right"]
            chosen = right_hands[0] if right_hands else (stable_gestures[0] if stable_gestures else None)

            if chosen:
                selected_hand_id = chosen["id"]
                predicted_gesture = chosen["gesture"]

        # ----------------------------
        # SPATIAL CONTROL (POINT)
        # ----------------------------
        point_direction = None

        if predicted_gesture == "Point" and selected_hand_id is not None:
            h = next((g["hand"] for g in hand_gestures if g["id"] == selected_hand_id), None)
            if h:
                lm = h["lmList"]
                point_direction = "LEFT" if lm[8][0] < lm[0][0] else "RIGHT"

        # Debug text
        cv2.putText(
            debug_img,
            f"Point dir: {point_direction}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if point_direction else (120, 120, 120),
            2
        )

        mode_commit, return_idle = point_tracker.update(
            point_dir=point_direction,
            control_active=bool(control_active)
        )

        if mode_commit:
            ext_actions = state_machine.set_mode_external(mode_commit)
            dispatcher.dispatch(ext_actions)

        if return_idle:
            ext_actions = state_machine.set_idle_external()
            dispatcher.dispatch(ext_actions)

        # ----------------------------
        # Controlling
        # ----------------------------
        current_selected = (
            selected_hand_id,
            predicted_gesture
        ) if predicted_gesture else None

        activated, deactivated, changed = compute_gesture_events(
            prev_selected,
            current_selected
        )

        if activated and selected_hand_id is not None:
            hand_id_tracker.lock_track(selected_hand_id)

        if deactivated and selected_hand_id is not None:
            hand_id_tracker.unlock_track(selected_hand_id)
        if not control_active:
            hand_id_tracker.unlock_all()
        if return_idle:
            hand_id_tracker.unlock_all()

        prev_selected = current_selected

        current = predicted_gesture

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

        dispatcher.sync_state(
            mode=state_machine.mode,
            submode=state_machine.submode,
            control_active=control_active
        )

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
            osc.disconnect()    # resets ping to 0 for debugging in TD
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

def mirror_landmarks_x(lm_list, image_width):
    """
    Mirrors hand landmarks horizontally so Left hands match Right-hand orientation.
    """
    return [(image_width - x, y, z) for (x, y, z) in lm_list]


if __name__ == "__main__":
    main()
