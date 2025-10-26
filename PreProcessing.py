import numpy as np
import math


class HandNormalizer:
    """
    Normalizes raw MediaPipe-style hand landmarks into a consistent,
    position-, scale-, and (optionally) rotation-invariant representation.

    Example usage:
        normalizer = HandNormalizer(rotation_invariant=True)
        norm_data = normalizer.normalize_ hand(landmarks)
    """

    def __init__(self, rotation_invariant=True, include_z=False):
        """
        Args:
            rotation_invariant (bool): Rotate hand to a canonical axis.
            include_z (bool): Whether to include z-coordinates in the output vector.
        """
        self.rotation_invariant = rotation_invariant
        self.include_z = include_z

    # --------------------------------------------------------------
    # Core normalization pipeline
    # --------------------------------------------------------------
    def normalize_hand(self, landmarks):
        """
        Normalize one hand's landmarks.

        Args:
            landmarks (list[tuple[float, float, float]]): 21 landmark points.

        Returns:
            dict: {
                "norm_landmarks": np.ndarray (21x3),
                "flat_vector": np.ndarray (1x63 or 1x63 with all z being 0 if include_z=False),
                "features": np.ndarray (optional geometric distances)
            }
        """
        lm = np.array(landmarks, dtype=np.float32)

        # 1. Translate wrist (landmark 0) to origin
        lm = self._translate_to_origin(lm)

        # 2. Scale normalization
        lm = self._scale(lm)

        # 3. Optional rotation normalization
        if self.rotation_invariant:
            lm = self._rotate(lm)

        # 4. Optional: zero out Z if not needed
        if not self.include_z:
            lm[:, 2] = 0.0

        # 5. Extract features
        features = self._extract_features(lm)

        return {
            "norm_landmarks": lm,
            "flat_vector": lm.flatten(),
            "features": features,
        }

    # --------------------------------------------------------------
    # Step 1 – Translation
    # --------------------------------------------------------------
    def _translate_to_origin(self, lm):
        wrist = lm[0]
        return lm - wrist

    # --------------------------------------------------------------
    # Step 2 – Scaling
    # --------------------------------------------------------------
    def _scale(self, lm):
        wrist = lm[0]
        middle_mcp = lm[9]
        scale = np.linalg.norm(middle_mcp - wrist)
        if scale < 1e-6:
            scale = 1e-6
        return lm / scale

    # --------------------------------------------------------------
    # Step 3 – Rotation normalization
    # --------------------------------------------------------------
    def _rotate(self, lm):
        wrist = lm[0]
        index_mcp = lm[5]
        x_axis = index_mcp - wrist
        angle = math.atan2(x_axis[1], x_axis[0])

        rotation_matrix = np.array([
            [math.cos(-angle), -math.sin(-angle)],
            [math.sin(-angle),  math.cos(-angle)]
        ])

        lm[:, :2] = np.dot(lm[:, :2], rotation_matrix.T)
        return lm

    # --------------------------------------------------------------
    # Step 4 – Feature extraction
    # --------------------------------------------------------------
    def _extract_features(self, lm):
        """
        Compute pairwise distances between fingertips as an example geometric descriptor.
        You can extend this with angles or velocities later.
        """
        fingertip_indices = [4, 8, 12, 16, 20]
        tips = lm[fingertip_indices, :2]
        distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(tips[i] - tips[j])
                distances.append(d)
        return np.array(distances, dtype=np.float32)


# ------------------------------------------------------------------
# Helper for combining two hands' feature vectors for training or inference
# ------------------------------------------------------------------
def combine_hands_features(left_data, right_data):
    """
    Combine two normalized hands (left and right) into one composite feature vector.
    Optionally adds inter-hand relational features like distance and orientation.

    Args:
        left_data (dict): Output of HandNormalizer.normalize_hand()
        right_data (dict): Output of HandNormalizer.normalize_hand()

    Returns:
        np.ndarray: Combined feature vector ready for model input
    """
    left_vec = left_data["flat_vector"]
    right_vec = right_data["flat_vector"]

    # Inter-hand features
    left_wrist = left_data["norm_landmarks"][0]
    right_wrist = right_data["norm_landmarks"][0]

    wrist_distance = np.linalg.norm(left_wrist[:2] - right_wrist[:2])

    # Relative angle between wrist lines (rough orientation cue)
    left_index = left_data["norm_landmarks"][5]
    right_index = right_data["norm_landmarks"][5]
    vec_left = left_index - left_wrist
    vec_right = right_index - right_wrist
    angle_left = math.atan2(vec_left[1], vec_left[0])
    angle_right = math.atan2(vec_right[1], vec_right[0])
    relative_angle = angle_right - angle_left

    combined = np.concatenate([
        left_vec,
        right_vec,
        np.array([wrist_distance, relative_angle], dtype=np.float32)
    ])
    return combined


# ------------------------------------------------------------------
# Optional utility for batch normalization
# ------------------------------------------------------------------
def normalize_all_hands(hands, normalizer):
    """
    Normalize multiple detected hands in one call.

    Args:
        hands (list[dict]): List of hand data with key 'lm' for landmarks.
        normalizer (HandNormalizer): Instance of the normalizer.

    Returns:
        list[dict]: Normalized hand data (same order as input).
    """
    return [normalizer.normalize_hand(h["lm"]) for h in hands]
