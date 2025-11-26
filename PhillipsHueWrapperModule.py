from phue import Bridge
import time

class HueController:
    def __init__(self, bridge_ip, username):
        """
        Initialize Hue controller and connect to the bridge.

        Args:
            bridge_ip (str): Local IP address of the Hue Bridge.
            username (str): Authorized API username from /api pairing.
        """
        self.bridge_ip = bridge_ip
        self.username = username
        self.bridge = Bridge(bridge_ip, username)
        self.bridge.connect()  # Safe even if already registered
        self.lights = self.bridge.get_light_objects('name')

    # ============================================================
    # Basic Light Controls
    # ============================================================
    def turn_on(self, light_name, transition=0):
        """
        Turn a light ON.

        Args:
            light_name (str)
            transition (int): Fade time in *deciseconds* (1 = 100ms).
                              Example: 10 = 1 second fade.
        """
        if light_name in self.lights:
            self.bridge.set_light(light_name, {
                "on": True,
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def turn_off(self, light_name, transition=0):
        """
        Turn a light OFF.

        Args:
            transition (int): Fade time in deciseconds.
        """
        if light_name in self.lights:
            self.bridge.set_light(light_name, {
                "on": False,
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def toggle(self, light_name, transition=0):
        """
        Toggle ON/OFF.

        Args:
            transition (int): Fade time in deciseconds.
        """
        if light_name in self.lights:
            current = self.lights[light_name].on
            self.bridge.set_light(light_name, {
                "on": not current,
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def set_brightness(self, light_name, bri, transition=0):
        """
        Set brightness (1–254).

        Args:
            bri (int): Brightness 1–254.
            transition (int): Fade time in deciseconds.
        """
        bri = max(1, min(254, int(bri)))
        if light_name in self.lights:
            self.bridge.set_light(light_name, {
                "bri": bri,
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    # ============================================================
    # White Color Temperature (Ambiance lights)
    # ============================================================
    def set_color_temperature(self, light_name, ct, transition=0):
        """
        Set white temperature.

        Args:
            ct (int): Color temperature in mireds (153–500)
                      153 = cool (blueish), 500 = warm (orange).
            transition (int): Fade time in deciseconds.
        """
        ct = max(153, min(500, int(ct)))

        if light_name in self.lights:
            self.bridge.set_light(light_name, {
                "ct": ct,
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def set_white(self, light_name, mode="neutral", transition=0):
        """
        Set white presets.

        Modes: cool, neutral, warm, very warm
        """
        mode = mode.lower()

        WHITE_MAP = {
            "cool": 153,
            "neutral": 300,
            "warm": 450,
            "very warm": 500
        }

        if mode not in WHITE_MAP:
            print(f"[WARN] Unknown white mode '{mode}'. Try: {list(WHITE_MAP.keys())}")
            return

        ct_value = WHITE_MAP[mode]
        self.set_color_temperature(light_name, ct_value, transition)
        print(f"Set '{light_name}' to {mode} white (ct={ct_value})")

    # ============================================================
    # Color Lights (Hue + Saturation)
    # ============================================================
    def set_color(self, light_name, hue, sat, transition=0):
        """
        Set color for RGB-capable bulbs.

        Args:
            hue (int): 0–65535 color wheel rotation.
            sat (int): 0–254 saturation.
            transition (int): Fade time in deciseconds.
        """
        if light_name in self.lights:
            self.bridge.set_light(light_name, {
                "hue": max(0, min(65535, int(hue))),
                "sat": max(0, min(254, int(sat))),
                "transitiontime": transition
            })
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def cycle_colors(self, light_name, delay=0.5):
        """Debug animation that jumps through hue values."""
        hues = [0, 10000, 20000, 30000, 40000, 50000, 60000]
        for h in hues:
            self.set_color(light_name, h, 254, transition=5)
            self.set_brightness(light_name, 200, transition=5)
            print(f"Set {light_name} hue={h}")
            time.sleep(delay)

    # ============================================================
    # Named Color Helper
    # ============================================================
    def set_color_by_name(self, light_name, color_name, transition=0):
        """
        Map common names to hue/saturation.

        Args:
            color_name (str): e.g. "red", "blue", "warm white"
            transition (int): fade time
        """
        color_name = color_name.lower().strip()

        COLOR_MAP = {
            "red":        (0, 254),
            "orange":     (8000, 254),
            "yellow":     (15000, 254),
            "green":      (25000, 254),
            "cyan":       (35000, 254),
            "blue":       (47000, 254),
            "purple":     (52000, 254),
            "pink":       (58000, 200),
            "white":      (0, 0),
            "warm white": (0, 30),
            "cool white": (45000, 30)
        }

        if color_name not in COLOR_MAP:
            print(f"[WARN] Unknown color '{color_name}'. Try one of: {list(COLOR_MAP.keys())}")
            return

        hue_val, sat_val = COLOR_MAP[color_name]
        self.set_color(light_name, hue_val, sat_val, transition)
        print(f"Set '{light_name}' to {color_name} (hue={hue_val}, sat={sat_val})")

    # ============================================================
    # Group Controls
    # ============================================================
    def set_group_state(self, group_name, state, value=None, transition=0):
        """
        Generic helper to control groups.

        Args:
            group_name (str)
            state (str): "on", "bri", "ct", etc.
            value: state value
            transition (int): fade time
        """
        body = {"transitiontime": transition}

        if value is not None:
            body[state] = value
        else:
            body[state] = True  # default behavior for "on"

        try:
            self.bridge.set_group(group_name, body)
        except Exception as e:
            print(f"[ERROR] Could not control group '{group_name}': {e}")

    def turn_on_group(self, group_name, transition=0):
        self.set_group_state(group_name, "on", True, transition)

    def turn_off_group(self, group_name, transition=0):
        self.set_group_state(group_name, "on", False, transition)

    def set_group_brightness(self, group_name, bri, transition=0):
        bri = max(1, min(254, int(bri)))
        self.set_group_state(group_name, "bri", bri, transition)

    # ============================================================
    # Scenes & Animation
    # ============================================================
    def run_scene(self, group_name, scene_name):
        """
        Activate a Hue Scene by name.
        """
        try:
            self.bridge.run_scene(group_name, scene_name)
            print(f"Running scene '{scene_name}' in '{group_name}'")
        except Exception as e:
            print(f"[ERROR] Could not run scene '{scene_name}' for '{group_name}': {e}")

    def animate_breath(self, light_name, min_bri=20, max_bri=254, speed=0.03):
        """
        Breathing animation (slow in/out fade).
        """
        for b in range(min_bri, max_bri, 5):
            self.set_brightness(light_name, b, transition=1)
            time.sleep(speed)

        for b in range(max_bri, min_bri, -5):
            self.set_brightness(light_name, b, transition=1)
            time.sleep(speed)

    # ============================================================
    # Query & Debug
    # ============================================================
    def list_lights(self):
        """Print all available lights."""
        print("Available lights:", list(self.lights.keys()))

    def get_light_state(self, light_name):
        """Return dict with light state."""
        if light_name in self.lights:
            return self.bridge.get_light(light_name)
        print(f"[WARN] Light '{light_name}' not found")
        return None