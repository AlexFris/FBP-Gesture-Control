from Actions import Action, ActionType

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




    def update_control_active(self, is_active):
        actions = []

        """
        Updates the arm-raised control flag.
        If control becomes inactive → return to IDLE.
        """

        # Transition: ACTIVE → INACTIVE
        if self.control_active and not is_active:
            self.control_active = False

            self.mode = "IDLE"
            self.submode = None

            actions.append(Action(ActionType.SET_MODE_IDLE))

            # Force-exit all submodes
            actions.append(Action(ActionType.EXIT_LIGHT_BRIGHTNESS))
            actions.append(Action(ActionType.EXIT_LIGHT_COLOR))
            actions.append(Action(ActionType.EXIT_SOUND_VOLUME))

        else:
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
