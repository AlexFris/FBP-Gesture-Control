import cv2
import math
import mediapipe as mp
import numpy as np

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

        # Example IDs: wrist, elbow, shoulder
        self.LeftArmIds = [16, 14, 12]

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)



        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
            )

        return img

class HandDetector:
    """
    Detects hands using MediaPipe, extracts landmarks, bounding boxes, and provides
    utility functions like counting fingers, distances, angles, and distance to camera.
    """

    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.8, minTrackCon=0.8):
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
