import cv2
import math
import mediapipe as mp
import numpy as np

######################################################
#       PoseDetector
######################################################

class PoseDetector:
    """
    Detects body landmarks (limbs, torso, etc.)
    """

    def __init__(self, static_mode=False, model_complexity=1,
                 smooth_landmarks=True, segmentation=False,
                 detectionCon=0.5, minTrackCon=0.5):

        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = segmentation
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.static_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.segmentation,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon,
        )

        self.mpDraw = mp.solutions.drawing_utils

        # ---------------------------------------------
        # UPRIGHT DETECTION CONFIGURATION (user adjustable)
        # ---------------------------------------------
        self.vertical_tolerance_deg = 25      # allowed angle from vertical
        self.frames_needed_up = 7             # frames required to confirm upright
        self.frames_needed_down = 5           # frames required to confirm NOT upright

        # Internal counters / states
        self.left_up_count = 0
        self.left_down_count = 0
        self.right_up_count = 0
        self.right_down_count = 0

        self.left_arm_upright = False
        self.right_arm_upright = False

        # Optional informational flags
        self.left_elbow_above_shoulder = False
        self.right_elbow_above_shoulder = False

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)



        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )

        return img

    def resolve_handedness(self, img, pose_landmarks, hand_results):
        """
        Robust: maps MP Hands (true-left / true-right)
        to the correct pose arm, EVEN if Pose swaps sides.
        """

        h, w = img.shape[:2]

        if pose_landmarks is None:
            return {"left": None, "right": None}

        P = pose_landmarks.landmark

        def px(lm):
            return int(lm.x * w), int(lm.y * h)

        # Pose joints
        pose_left = {  # REAL LEFT ARM
            "shoulder": px(P[12]),  # MP Right Shoulder
            "elbow": px(P[14]),  # MP Right Elbow
            "wrist": px(P[16])  # MP Right Wrist
        }
        pose_right = {  # REAL RIGHT ARM
            "shoulder": px(P[11]),  # MP Left Shoulder
            "elbow": px(P[13]),  # MP Left Elbow
            "wrist": px(P[15])  # MP Left Wrist
        }

        # Default output
        out_left = pose_left
        out_right = pose_right

        # If no hands → return raw pose
        if hand_results is None or hand_results.multi_hand_landmarks is None:
            return {"left": out_left, "right": out_right}

        # Track usage to avoid double-assignment
        pose_left_used = False
        pose_right_used = False

        for hand_type, hand_lms in zip(hand_results.multi_handedness,
                                       hand_results.multi_hand_landmarks):

            label = hand_type.classification[0].label  # "Left" / "Right"

            hw = hand_lms.landmark[0]
            hx, hy = int(hw.x * w), int(hw.y * h)

            lx, ly = pose_left["wrist"]
            rx, ry = pose_right["wrist"]

            d_left = (hx - lx) ** 2 + (hy - ly) ** 2
            d_right = (hx - rx) ** 2 + (hy - ry) ** 2

            # Choose closest matching pose wrist
            if d_left < d_right:
                matched = pose_left
                pose_left_used = True
            else:
                matched = pose_right
                pose_right_used = True

            if label == "Left":
                out_left = matched
            else:
                out_right = matched

        return {"left": out_left, "right": out_right}

    def draw_arm(self, img, arm, color=(0, 0, 255), thickness=3):
        """
        Draws the shoulder → elbow → wrist chain for a single arm.

        arm: {"shoulder":(x,y), "elbow":(x,y), "wrist":(x,y)}
        color: (B,G,R)
        """
        if arm is None:
            return img

        shoulder = arm["shoulder"]
        elbow    = arm["elbow"]
        wrist    = arm["wrist"]

        # Joints
        cv2.circle(img, shoulder, 6, color, -1)
        cv2.circle(img, elbow,    6, color, -1)
        cv2.circle(img, wrist,    6, color, -1)

        # Bones
        cv2.line(img, shoulder, elbow, color, thickness)
        cv2.line(img, elbow,    wrist, color, thickness)

        return img

    def draw_arms(self, img, arms):
        """
        Draws both left and right arms in different colors.

        arms: {
            "left":  {shoulder:(x,y), elbow:(x,y), wrist:(x,y)},
            "right": { ... }
        }
        """

        left  = arms.get("left")
        right = arms.get("right")

        # LEFT ARM = BLUE
        img = self.draw_arm(img, left,  color=(255, 0, 0))

        # RIGHT ARM = RED
        img = self.draw_arm(img, right, color=(0, 0, 255))

        return img

    def update_arm_upright_states(self, arms):
        """
        Uses handedness-resolved arms (not raw MediaPipe indices).
        Checks whether each arm is upright based on elbow→wrist angle.
        """

        # ---------------------------
        # LEFT ARM
        # ---------------------------
        left = arms.get("left")
        if left:
            elbow = left["elbow"]
            wrist = left["wrist"]

            # Only consider upright if wrist is ABOVE elbow
            if wrist[1] < elbow[1]:

                angle_left = self._arm_vertical_angle(elbow, wrist)
                self.debug_left_angle = angle_left

                if angle_left < self.vertical_tolerance_deg:
                    self.left_up_count += 1
                    self.left_down_count = 0

                    if self.left_up_count >= self.frames_needed_up:
                        self.left_arm_upright = True
                else:
                    self.left_down_count += 1
                    self.left_up_count = 0

                    if self.left_down_count >= self.frames_needed_down:
                        self.left_arm_upright = False

            else:
                # Wrist is BELOW elbow → instantly counts as NOT upright
                self.left_down_count += 1
                self.left_up_count = 0
                self.debug_left_angle = 999  # For debugging
                if self.left_down_count >= self.frames_needed_down:
                    self.left_arm_upright = False

        # ---------------------------
        # RIGHT ARM
        # ---------------------------
        right = arms.get("right")
        if right:
            elbow = right["elbow"]
            wrist = right["wrist"]

            # Only consider upright if wrist is ABOVE elbow
            if wrist[1] < elbow[1]:

                angle_right = self._arm_vertical_angle(elbow, wrist)
                self.debug_right_angle = angle_right

                if angle_right < self.vertical_tolerance_deg:
                    self.right_up_count += 1
                    self.right_down_count = 0

                    if self.right_up_count >= self.frames_needed_up:
                        self.right_arm_upright = True
                else:
                    self.right_down_count += 1
                    self.right_up_count = 0

                    if self.right_down_count >= self.frames_needed_down:
                        self.right_arm_upright = False

            else:
                # Wrist is BELOW elbow
                self.right_down_count += 1
                self.right_up_count = 0
                self.debug_right_angle = 999
                if self.right_down_count >= self.frames_needed_down:
                    self.right_arm_upright = False

    def _arm_vertical_angle(self, elbow, wrist):
        """
        Computes the deviation from vertical for the line elbow → wrist.
        Perfect vertical (wrist directly above elbow) = 0 degrees.
        """

        ex, ey = elbow
        wx, wy = wrist

        # Vector wrist - elbow
        dx = wx - ex
        dy = ey - wy  # invert Y because OpenCV origin is top-left

        # Angle from vertical
        angle = abs(math.degrees(math.atan2(dx, dy)))

        return angle

    def debug_handedness(self, img, arms, hand_results, pose_landmarks):
        """
        Draw debugging lines showing the matched hand → pose wrist.
        Also draws pose left/right arm labels for visual confirmation.
        """

        h, w = img.shape[:2]

        if pose_landmarks is None:
            return img

        P = pose_landmarks.landmark

        # Pose wrist pixel positions
        pose_left_wrist = (int(P[16].x * w), int(P[16].y * h))
        pose_right_wrist = (int(P[15].x * w), int(P[15].y * h))

        # Draw pose wrist labels
        cv2.putText(img, "POSE LEFT", (pose_left_wrist[0] - 40, pose_left_wrist[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(img, "POSE RIGHT", (pose_right_wrist[0] - 40, pose_right_wrist[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # If no hands detected, nothing more to debug
        if hand_results is None or hand_results.multi_hand_landmarks is None:
            return img

        # Draw arrows from hand → matched pose wrist
        for hand_type, hand_lms in zip(hand_results.multi_handedness,
                                       hand_results.multi_hand_landmarks):

            label = hand_type.classification[0].label  # "Left" / "Right"

            hw = hand_lms.landmark[0]  # wrist of hand detector
            hx, hy = int(hw.x * w), int(hw.y * h)

            if label == "Left":
                pw = arms["left"]["wrist"]
                color = (255, 0, 0)  # BLUE for left
            else:
                pw = arms["right"]["wrist"]
                color = (0, 0, 255)  # RED for right

            # Draw arrow
            cv2.arrowedLine(img, (hx, hy), pw, color, 3, tipLength=0.25)

            # Label the arrow
            cv2.putText(
                img,
                f"{label} hand → pose",
                (hx + 10, hy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return img


######################################################
#       HandDetector
######################################################

class HandDetector:
    """
    Detects hands using MediaPipe, extracts landmarks, bounding boxes, and provides
    utility functions like counting fingers, distances, angles, and distance to camera.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.6, minTrackCon=0.6):
        """
        Args:
            staticMode (bool): Detect hands on each frame (slower) or continuous tracking.
            maxHands (int): Maximum number of hands to detect.
            modelComplexity (int): Complexity of the hand landmark model (0 or 1).
            detectionCon (float): Minimum detection confidence threshold.
            minTrackCon (float): Minimum tracking confidence threshold.
        """
        self.staticMode = staticMode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        # MediaPipe Hands initialization
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticMode,
            max_num_hands=self.maxHands,
            model_complexity=self.modelComplexity,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    def findHands(self, img, draw=True, flipType=False):
        """
        Detect hands and return landmark and bounding box information.

        Args:
            img (ndarray): BGR image frame
            draw (bool): Draw landmarks and bounding boxes on the image
            flipType (bool): Flip left/right hand labels

        Returns:
            allHands (list): List of hands with lmList, bbox, center, type
            img (ndarray): Image with optional drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        allHands = []
        h, w, c = img.shape

        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                lmList = []
                xList, yList = [], []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    lmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                # Bounding box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + boxW // 2, bbox[1] + boxH // 2

                myHand["lmList"] = lmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # Hand type
                if flipType:
                    myHand["type"] = "Left" if handType.classification[0].label == "Right" else "Right"
                else:
                    myHand["type"] = handType.classification[0].label

                allHands.append(myHand)

                # Draw landmarks and bounding box
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[0]+boxW+20, bbox[1]+boxH+20), (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0]-30, bbox[1]-30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return allHands, img

    def fingersUp(self, myHand):
        """
        Determine which fingers are up.

        Args:
            myHand (dict): Hand dictionary from findHands

        Returns:
            list: [1 if finger is up, 0 if down] for Thumb, Index, Middle, Ring, Pinky
        """
        fingers = []
        myHandType = myHand["type"]
        lmList = myHand["lmList"]

        # Thumb
        if myHandType == "Right":
            fingers.append(1 if lmList[self.tipIds[0]][0] > lmList[self.tipIds[0]-1][0] else 0)
        else:
            fingers.append(1 if lmList[self.tipIds[0]][0] < lmList[self.tipIds[0]-1][0] else 0)

        # 4 Fingers
        for id in range(1, 5):
            fingers.append(1 if lmList[self.tipIds[id]][1] < lmList[self.tipIds[id]-2][1] else 0)

        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        """
        Find distance between two points and optionally draw.

        Args:
            p1, p2 (tuple): (x, y) coordinates
            img (ndarray, optional): Image to draw
            color (tuple): Line color
            scale (int): Circle radius

        Returns:
            length (float), info (tuple), img (ndarray)
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale//3))
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, info, img

    def findHandAngle(self, p1, p2, img=None, color=(0, 255, 0), thickness=2):
        """
        Find angle between two points in degrees and optionally draw.

        Args:
            p1, p2 (tuple): Points
            img (ndarray, optional): Image to draw
            color (tuple): Line color
            thickness (int): Line thickness

        Returns:
            angle_deg (float), img (ndarray)
        """
        x1, y1 = p1
        x2, y2 = p2
        angle_rad = math.atan2(y1 - y2, x1 - x2)
        angle_deg = math.degrees(angle_rad) % 360

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        return angle_deg, img

    def findHandDistance(self, p1, p2, hand_bbox, img=None):
        """
        Estimate distance of hand from camera based on landmark spacing.

        Args:
            p1, p2 (tuple): Two landmark points (x, y) should be lm5 and lm7
            hand_bbox (tuple): Bounding box of hand (x, y, w, h)
            img (ndarray, optional): Image to draw

        Returns:
            distanceCM (float)
        """
        # Example polynomial mapping
        x_map = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        y_map = [270, 190, 145, 115, 95, 82, 72, 64, 56, 51, 48]
        A, B, C = np.polyfit(x_map, y_map, 2)

        x1, y1 = p1
        x2, y2 = p2
        x, y, w, h = hand_bbox

        distance_px = math.hypot(x2 - x1, y2 - y1)
        distanceCM = A*distance_px**2 + B*distance_px + C

        if img is not None:
            cv2.putText(img, f"{int(distanceCM)} cm", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return distanceCM
