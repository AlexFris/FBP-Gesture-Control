import math
from Actions import Action, ActionType

class HandIdentityTracker:
    def __init__(
        self,
        base_match_distance_px=80,
        max_missing_frames=8,
        velocity_weight=0.6
    ):
        self.base_match_distance = base_match_distance_px
        self.max_missing_frames = max_missing_frames
        self.velocity_weight = velocity_weight

        self.next_id = 0
        self.frame_idx = 0

        # track_id -> track state
        self.tracks = {}
        # track:
        # {
        #   "center": (x, y),
        #   "velocity": (vx, vy),
        #   "hand_type": "Left"/"Right"/None,
        #   "last_seen": frame_idx,
        #   "created_at": frame_idx
        #   "Locked": False/True
        # }

    # ---------------------------------------------------------
    # MAIN UPDATE
    # ---------------------------------------------------------
    def update(self, hands):
        """
        Assign stable IDs to detected hands.

        Args:
            hands: list of hand dicts with at least:
                   hand["center"] -> (x, y)
                   hand["type"]   -> "Left"/"Right"

        Returns:
            list of dicts: { "id": int, "hand": hand_dict }
        """
        self.frame_idx += 1
        assignments = []
        used_track_ids = set()

        # -------------------------------
        # MATCH DETECTIONS TO TRACKS
        # -------------------------------
        for hand in hands:
            cx, cy = hand["center"]
            hand_type = hand.get("type")

            best_id = None
            best_score = None

            for track_id, track in self.tracks.items():
                if track_id in used_track_ids:
                    continue

                # Prefer same handedness
                if track["hand_type"] and hand_type:
                    if track["hand_type"] != hand_type:
                        continue

                # Predict position using velocity
                px, py = track["center"]
                vx, vy = track["velocity"]
                max_pred = 50  # pixels, tuneable
                pred_x = px + max(-max_pred, min(vx, max_pred))
                pred_y = py + max(-max_pred, min(vy, max_pred))

                dist = math.hypot(cx - pred_x, cy - pred_y)

                # Adaptive threshold (faster motion = larger gate)
                speed = math.hypot(vx, vy)
                adaptive_thresh = self.base_match_distance + speed * self.velocity_weight

                if dist > adaptive_thresh and not track["locked"]:
                    continue

                # Bias toward older tracks
                age = self.frame_idx - track["created_at"]
                score = dist - age * 4.0

                # STRONG LOCK-IN BIAS
                if track.get("locked", False):
                    score -= 1000

                if best_score is None or score < best_score:
                    best_id = track_id
                    best_score = score

            # -------------------------------
            # CREATE OR UPDATE TRACK
            # -------------------------------
            if best_id is None:
                # New track
                best_id = self.next_id
                self.next_id += 1

                self.tracks[best_id] = {
                    "center": (cx, cy),
                    "velocity": (0.0, 0.0),
                    "hand_type": hand_type,
                    "last_seen": self.frame_idx,
                    "created_at": self.frame_idx,
                    "locked": False,
                }
            else:
                # Update existing track
                track = self.tracks[best_id]
                px, py = track["center"]
                vx, vy = track["velocity"]

                new_vx = cx - px
                new_vy = cy - py

                # Smoothed velocity update
                track["velocity"] = (
                    0.7 * vx + 0.3 * new_vx,
                    0.7 * vy + 0.3 * new_vy,
                )
                track["center"] = (cx, cy)
                track["hand_type"] = hand_type
                track["last_seen"] = self.frame_idx

            used_track_ids.add(best_id)
            assignments.append({"id": best_id, "hand": hand})

        # -------------------------------
        # REMOVE STALE TRACKS
        # -------------------------------
        to_delete = []
        for track_id, track in self.tracks.items():
            if self.frame_idx - track["last_seen"] > self.max_missing_frames:
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.tracks[track_id]

        return assignments

    def lock_track(self, track_id):
        if track_id in self.tracks:
            self.tracks[track_id]["locked"] = True

    def unlock_track(self, track_id):
        if track_id in self.tracks:
            self.tracks[track_id]["locked"] = False

    def unlock_all(self):
        for t in self.tracks.values():
            t["locked"] = False


def compute_gesture_events(prev_gesture, curr_gesture):
    """
    Compute (activated, deactivated, changed) events from a stable gesture stream.
    This replaces the global GestureStabilizer events when you already stabilized per-hand.
    """
    activated = None
    deactivated = None
    changed = None

    if prev_gesture is None and curr_gesture is not None:
        activated = curr_gesture
    elif prev_gesture is not None and curr_gesture is None:
        deactivated = prev_gesture
    elif prev_gesture is not None and curr_gesture is not None and prev_gesture != curr_gesture:
        changed = (prev_gesture, curr_gesture)
        activated = curr_gesture  # keeps your FSM behavior consistent

    return activated, deactivated, changed

class PerHandGestureTracker:
    """
    Maintains a GestureStabilizer per physical hand ID.
    """

    def __init__(self, frames_needed_up=7, frames_needed_down=5, confidence_threshold=0.6):
        self.frames_needed_up = frames_needed_up
        self.frames_needed_down = frames_needed_down
        self.confidence_threshold = confidence_threshold

        self.stabilizers = {}  # hand_id -> GestureStabilizer

    def update(self, hand_gestures):
        """
        hand_gestures: list of dicts:
            {
              "id": int,
              "hand": handdict,
              "gesture": str | None,
              "confidence": float
            }
        """

        present_ids = set()

        for g in hand_gestures:
            hid = g["id"]
            present_ids.add(hid)

            if hid not in self.stabilizers:
                self.stabilizers[hid] = GestureStabilizer(
                    self.frames_needed_up,
                    self.frames_needed_down,
                    self.confidence_threshold
                )

            self.stabilizers[hid].update(g["gesture"], g["confidence"])

        # Advance missing frames for hands not seen this frame
        for hid, stabilizer in list(self.stabilizers.items()):
            if hid not in present_ids:
                stabilizer.update(None, None)

    def stable(self, hand_id):
        if hand_id in self.stabilizers:
            return self.stabilizers[hand_id].stable_gesture
        return None

class GestureStabilizer:
    """
    Stabilizes frame-by-frame gesture predictions.
    Produces a single stable gesture at a time with noise smoothing.
    """

    def __init__(self,
                 frames_needed_up=7,
                 frames_needed_down=5,
                 confidence_threshold=0.6):

        self.frames_needed_up = frames_needed_up
        self.frames_needed_down = frames_needed_down
        self.confidence_threshold = confidence_threshold

        # Internal stabilizer state
        self.stable_gesture = None
        self.candidate_gesture = None
        self.candidate_frames = 0
        self.missing_frames = 0

    def update(self, gesture, confidence):
        """
        Update stabilizer with frame prediction.
        gesture: string or None
        confidence: float
        Returns events: (activated, deactivated, changed)
        """

        activated = None
        deactivated = None
        changed = None

        # Treat missing or low-confidence predictions as no gesture
        if confidence is None or confidence < self.confidence_threshold:
            gesture = None

        ### CASE 1: No stable gesture yet ###
        if self.stable_gesture is None:
            # We are trying to establish a new stable gesture
            if gesture == self.candidate_gesture:
                self.candidate_frames += 1
            else:
                self.candidate_gesture = gesture
                self.candidate_frames = 1

            # Promote candidate to stable
            if (self.candidate_gesture is not None and
                    self.candidate_frames >= self.frames_needed_up):
                self.stable_gesture = self.candidate_gesture
                activated = self.stable_gesture

            return activated, deactivated, changed

        ### CASE 2: We HAVE a stable gesture ###
        if gesture == self.stable_gesture:
            # Still seeing the stable gesture
            self.missing_frames = 0
            return activated, deactivated, changed

        # Gesture changed or disappeared
        if gesture is None:
            # Count missing frames
            self.missing_frames += 1

            if self.missing_frames >= self.frames_needed_down:
                # Stable gesture ends
                deactivated = self.stable_gesture
                self.stable_gesture = None
                self.candidate_gesture = None
                self.candidate_frames = 0

            return activated, deactivated, changed

        # Candidate is a different gesture than the stable one
        if gesture == self.candidate_gesture:
            self.candidate_frames += 1
        else:
            self.candidate_gesture = gesture
            self.candidate_frames = 1

        # Promote new gesture
        if self.candidate_frames >= self.frames_needed_up:
            changed = (self.stable_gesture, self.candidate_gesture)
            self.stable_gesture = self.candidate_gesture
            self.missing_frames = 0
            activated = self.stable_gesture

        return activated, deactivated, changed

#----------------------
# STATEMACHINE
#----------------------

class InteractionStateMachine:
    """
    Controls high-level modes and actions based on stabilized gesture events.
    This is the skeleton; it contains no device-specific actions yet.
    """

    def __init__(self):
        # Global modes
        self.mode = "IDLE"     # IDLE, LIGHT, SOUND

        # Submodes (continuous control states)
        self.submode = None    # e.g., LIGHT_COLOR, LIGHT_BRIGHTNESS, SOUND_VOLUME

        # External flag (from arm detection)
        self.control_active = False

    def set_mode_external(self, mode_name: str):
        """
        External mode set (e.g., pointing) that bypasses control_active gating.
        Returns actions to update OSC state cleanly.
        """
        actions = []

        if mode_name not in ("LIGHT", "SOUND"):
            return actions

        if self.mode == mode_name:
            return actions  # no change

        # Clear any submode when switching modes
        self.submode = None

        # Exit all submodes explicitly for visuals consistency
        actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
        actions.append(Action(ActionType.EXIT_LIGHT_COLOR))
        actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

        self.mode = mode_name

        if mode_name == "LIGHT":
            actions.append(Action(ActionType.SET_MODE_LIGHT))
        else:
            actions.append(Action(ActionType.SET_MODE_SOUND))

        return actions

    def set_idle_external(self):
        """
        External return to IDLE (e.g., grace window expired) bypassing control_active gating.
        """
        actions = []

        if self.mode == "IDLE" and self.submode is None:
            return actions

        self.mode = "IDLE"
        self.submode = None

        actions.append(Action(ActionType.SET_MODE_IDLE))
        actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
        actions.append(Action(ActionType.EXIT_LIGHT_COLOR))
        actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

        return actions

    def update_control_active(self, is_active):
        actions = []

        """
        Updates the arm-raised control flag.
        If control becomes inactive → return to IDLE.
        """

        # Transition: ACTIVE → INACTIVE
        if self.control_active and not is_active:
            self.control_active = False

            # Reset FSM state
            self._reset_to_idle()

            # Emit authoritative reset actions
            actions.append(Action(ActionType.SET_MODE_IDLE))
            actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
            actions.append(Action(ActionType.EXIT_LIGHT_COLOR))
            actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

            return actions

        # Normal update
        self.control_active = is_active
        return actions

    def process_gesture_event(self, activated=None, deactivated=None, changed=None):
        actions = []

        if not self.control_active:
            return actions

        if activated:
            actions.extend(self._handle_gesture_activation(activated))

        if changed:
            old, new = changed
            actions.extend(self._handle_gesture_change(old, new))

        if deactivated:
            actions.extend(self._handle_gesture_deactivation(deactivated))

        return actions

    # -----------------------------------------------------------
    # Internal handlers
    # -----------------------------------------------------------

    def _reset_to_idle(self):
        self.mode = "IDLE"
        self.submode = None

    def _handle_gesture_activation(self, gesture):
        actions = []
        """
        When a gesture becomes stable
        """
        # ---------------------
        # MODE-SELECTING GESTURES
        # ---------------------
        if gesture == "Light":
            self.mode = "LIGHT"
            self.submode = None
            actions.append(Action(ActionType.SET_MODE_LIGHT))
            return actions

        if gesture == "Sound":
            self.mode = "SOUND"
            self.submode = None
            actions.append(Action(ActionType.SET_MODE_SOUND))
            return actions

        # ---------------------
        # EVENT-ACTIONS
        # ---------------------
        if gesture == "Open":
            if self.mode == "LIGHT":
                actions.append(Action(ActionType.LIGHT_ON))
            elif self.mode == "SOUND":
                actions.append(Action(ActionType.SOUND_PLAY))
            return actions

        if gesture == "Fist":
            if self.mode == "LIGHT":
                actions.append(Action(ActionType.LIGHT_OFF))
            elif self.mode == "SOUND":
                actions.append(Action(ActionType.SOUND_PAUSE))
            return actions

        # ---------------------
        # CONTINUOUS ACTION ENTRY
        # ---------------------
        if gesture == "Pinch" and self.mode == "LIGHT":
            self.submode = "LIGHT_COLOR"
            actions.append(Action(ActionType.ENTER_LIGHT_COLOR))
            return actions

        if gesture == "FlatHand":
            if self.mode == "LIGHT":
                self.submode = "LIGHT_BRIGHTNESS"
                actions.append(Action(ActionType.ENTER_LIGHT_BRIGHTNESS))
            elif self.mode == "SOUND":
                self.submode = "SOUND_VOLUME"
                actions.append(Action(ActionType.ENTER_SOUND_VOLUME))
            return actions

        return actions

    def _handle_gesture_change(self, old, new):
        actions = []
        """
        Handles transitions such as Flat → Pinch or Light → Sound
        """
        # ---------------------
        # Switching between LIGHT and SOUND
        # ---------------------
        if new == "Light":
            self.mode = "LIGHT"
            self.submode = None
            actions.append(Action(ActionType.SET_MODE_LIGHT))
            actions.append(Action(ActionType.SET_MODE_SOUND, value=0))
            return actions

        if new == "Sound":
            self.mode = "SOUND"
            self.submode = None
            actions.append(Action(ActionType.SET_MODE_SOUND))
            actions.append(Action(ActionType.SET_MODE_LIGHT, value=0))
            return actions

        # ---------------------
        # Continuous adjustment switching
        # ---------------------
        if old == "Pinch" and self.submode == "LIGHT_COLOR":
            self.submode = None
            actions.append(Action(ActionType.EXIT_LIGHT_COLOR))

        if old == "FlatHand":
            if self.submode == "LIGHT_BRIGHTNESS":
                self.submode = None
                actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
            elif self.submode == "SOUND_VOLUME":
                self.submode = None
                actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

        return actions

    def _handle_gesture_deactivation(self, gesture):
        actions = []

        if gesture == "Pinch" and self.submode == "LIGHT_COLOR":
            self.submode = None
            actions.append(Action(ActionType.EXIT_LIGHT_COLOR))

        if gesture == "FlatHand":
            if self.submode == "LIGHT_BRIGHTNESS":
                self.submode = None
                actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
            elif self.submode == "SOUND_VOLUME":
                self.submode = None
                actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

        return actions

class PointIntentTracker:
    def __init__(
        self,
        dwell_frames=8,         # frames needed to "commit" LEFT/RIGHT while pointing
        release_frames=3,       # frames Point must be missing to count as "released"
        grace_frames=20         # frames to wait for follow-up intent before returning to IDLE
    ):
        self.dwell_frames = dwell_frames
        self.release_frames = release_frames
        self.grace_frames = grace_frames

        self.current_dir = None           # "LEFT" / "RIGHT" / None
        self.dwell_count = 0

        self.committed_mode = None        # "LIGHT" / "SOUND" / None

        self.release_count = 0
        self.in_grace = False
        self.grace_count = 0

    def update(self, point_dir, control_active):
        """
        point_dir: "LEFT" / "RIGHT" / None (None means no Point detected)
        control_active: bool/int
        Returns: (mode_commit, return_to_idle)
          mode_commit: "LIGHT"/"SOUND"/None when commitment changes
          return_to_idle: True when grace expires with no engagement
        """

        mode_commit = None
        return_to_idle = False

        is_pointing = point_dir in ("LEFT", "RIGHT")

        # If user raises arm, consider them engaged and stop grace countdown
        if control_active:
            self.in_grace = False
            self.grace_count = 0
            self.release_count = 0

        # -----------------------------
        # POINTING ACTIVE (negotiation)
        # -----------------------------
        if is_pointing:
            # pointing resumes => stop grace/release
            self.in_grace = False
            self.grace_count = 0
            self.release_count = 0

            if point_dir == self.current_dir:
                self.dwell_count += 1
            else:
                self.current_dir = point_dir
                self.dwell_count = 1

            # Commit mode once dwell reached
            if self.dwell_count >= self.dwell_frames:
                desired_mode = "LIGHT" if point_dir == "LEFT" else "SOUND"
                if desired_mode != self.committed_mode:
                    self.committed_mode = desired_mode
                    mode_commit = desired_mode

            return mode_commit, return_to_idle

        # -----------------------------
        # NOT POINTING
        # -----------------------------
        self.current_dir = None
        self.dwell_count = 0

        # If we never committed, nothing to do
        if self.committed_mode is None:
            self.release_count = 0
            self.in_grace = False
            self.grace_count = 0
            return mode_commit, return_to_idle

        # We had a committed mode: count "release"
        self.release_count += 1

        if self.release_count >= self.release_frames:
            # Start grace window if not already engaged
            if not control_active:
                self.in_grace = True

        # If in grace, count down to IDLE unless engagement happens
        if self.in_grace and not control_active:
            self.grace_count += 1
            if self.grace_count >= self.grace_frames:
                # Return to IDLE and clear commitment
                return_to_idle = True
                self.committed_mode = None
                self.in_grace = False
                self.grace_count = 0
                self.release_count = 0

        return mode_commit, return_to_idle
