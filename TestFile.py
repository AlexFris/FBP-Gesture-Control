from KinectWrapper import KinectManager
import cv2
import mediapipe as mp
from BodyDetector import HandDetector, PoseDetector

pose_detector = PoseDetector()
hand_detector = HandDetector()
mp_draw = mp.solutions.drawing_utils

k = KinectManager()
print("Running... Press ESC to exit.")

# ---------------------------
# Main Loop
# ---------------------------
while True:

    rgb = k.get_rgb()
    if rgb is None:
        continue
    h, w = rgb.shape[:2]

    # 1) Pose
    pose_detector.findPose(rgb, draw=False)
    pose_lm = pose_detector.results.pose_landmarks if pose_detector.results else None

    # 2) Hands
    hands, rgb = hand_detector.findHands(rgb, draw=True)

    # 3) Handedness resolved arms (crucial!)
    arms = pose_detector.resolve_handedness(
        rgb,
        pose_lm,
        hand_detector.results
    )

    # 4) Upright state tracking
    pose_detector.update_arm_upright_states(arms)

    # 5) Draw arms
    rgb = pose_detector.draw_arms(rgb, arms)

    # 6) Debug the handedness visually
    rgb = pose_detector.debug_handedness(
        rgb,
        arms,
        hand_detector.results,
        pose_lm
    )

    # 7) Draw debug text
    cv2.putText(rgb, f"L angle: {pose_detector.debug_left_angle:.1f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(rgb, f"L upright: {pose_detector.left_arm_upright}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(rgb, f"R angle: {pose_detector.debug_right_angle:.1f}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(rgb, f"R upright: {pose_detector.right_arm_upright}",
                (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display
    cv2.imshow("RGB", rgb)

    if cv2.waitKey(1) & 0xFF == 27:
        break

k.close()
cv2.destroyAllWindows()
