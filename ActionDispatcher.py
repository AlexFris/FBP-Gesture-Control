from Actions import ActionType

class ActionDispatcher:
    def __init__(self, hue, spotify, osc, light_name):
        self.hue = hue
        self.spotify = spotify
        self.osc = osc
        self.light_name = light_name

    def dispatch(self, actions):
        for action in actions:
            self._dispatch_one(action)

    def _dispatch_one(self, action):
        t = action.type
        v = action.value

        # ------------------
        # MODE / STATE
        # ------------------
        if t == ActionType.SET_MODE_IDLE:
            self.osc.send("/mode/light", 0)
            self.osc.send("/mode/sound", 0)

        elif t == ActionType.SET_MODE_LIGHT:
            self.osc.send("/mode/light", 1)
            self.osc.send("/mode/sound", 0)

        elif t == ActionType.SET_MODE_SOUND:
            self.osc.send("/mode/light", 0)
            self.osc.send("/mode/sound", 1)

        # ------------------
        # LIGHT
        # ------------------
        elif t == ActionType.LIGHT_ON:
            self.hue.turn_on(self.light_name)
            self.osc.send("/light/on", 1)

        elif t == ActionType.LIGHT_OFF:
            self.hue.turn_off(self.light_name)
            self.osc.send("/light/on", 0)

        # ------------------
        # SOUND
        # ------------------
        elif t == ActionType.SOUND_PLAY:
            self.spotify.play()
            self.osc.send("/sound/play", 1)

        elif t == ActionType.SOUND_PAUSE:
            self.spotify.pause()
            self.osc.send("/sound/play", 0)

        # ------------------
        # SUBMODES (VISUALS)
        # ------------------
        elif t == ActionType.ENTER_LIGHT_BRIGHTNESS:
            self.osc.send("/submode/light/brightness", 1)

        elif t == ActionType.EXIT_LIGHT_BRIGHTNESS:
            self.osc.send("/submode/light/brightness", 0)

        elif t == ActionType.ENTER_LIGHT_COLOR:
            self.osc.send("/submode/light/color", 1)

        elif t == ActionType.EXIT_LIGHT_COLOR:
            self.osc.send("/submode/light/color", 0)

        elif t == ActionType.ENTER_SOUND_VOLUME:
            self.osc.send("/submode/sound/volume", 1)

        elif t == ActionType.EXIT_SOUND_VOLUME:
            self.osc.send("/submode/sound/volume", 0)

    def sync_state(self, mode, submode, control_active):
        """
        Authoritatively sync global interaction state to OSC.
        Called once per frame.
        """

        # ------------------
        # CONTROL
        # ------------------
        self.osc.send_bool("/control/active", control_active)

        # ------------------
        # MODES
        # ------------------
        self.osc.send_bool("/mode/light", int(mode == "LIGHT"))
        self.osc.send_bool("/mode/sound", int(mode == "SOUND"))

        # ------------------
        # SUBMODES
        # ------------------
        self.osc.send_bool(
            "/submode/light/color",
            int(submode == "LIGHT_COLOR")
        )
        self.osc.send_bool(
            "/submode/light/brightness",
            int(submode == "LIGHT_BRIGHTNESS")
        )
        self.osc.send_bool(
            "/submode/sound/volume",
            int(submode == "SOUND_VOLUME")
        )
