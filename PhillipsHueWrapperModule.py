from phue import Bridge
import time

class HueController:
    def __init__(self, bridge_ip, username):
        """
        Initialize Hue controller and connect to the bridge.

        Args:
            bridge_ip (str): Local IP address of the Hue Bridge.
            username (str): Authorized API username.
        """
        self.bridge_ip = bridge_ip
        self.username = username
        self.bridge = Bridge(bridge_ip, username)
        self.bridge.connect()  # connect once, safe if already registered
        self.lights = self.bridge.get_light_objects('name')

    # ------------------------
    # Basic Light Controls
    # ------------------------
    def turn_on(self, light_name):
        """Turn a light on by name."""
        if light_name in self.lights:
            self.lights[light_name].on = True
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def turn_off(self, light_name):
        """Turn a light off by name."""
        if light_name in self.lights:
            self.lights[light_name].on = False
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def toggle(self, light_name):
        """Toggle light on/off."""
        if light_name in self.lights:
            current = self.lights[light_name].on
            self.lights[light_name].on = not current
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def set_brightness(self, light_name, bri):
        """Set brightness (1–254)."""
        bri = max(1, min(254, int(bri)))
        if light_name in self.lights:
            self.lights[light_name].brightness = bri
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def set_color(self, light_name, hue, sat):
        """Set light colour (Hue: 0–65535, Saturation: 0–254)."""
        if light_name in self.lights:
            light = self.lights[light_name]
            light.hue = max(0, min(65535, int(hue)))
            light.saturation = max(0, min(254, int(sat)))
        else:
            print(f"[WARN] Light '{light_name}' not found")

    def cycle_colors(self, light_name, delay=0.5):
        """Cycle through basic colours to test hue values."""
        hues = [0, 10000, 20000, 30000, 40000, 50000, 60000]
        for h in hues:
            self.set_color(light_name, h, 254)
            self.set_brightness(light_name, 200)
            print(f"Set {light_name} hue={h}")
            time.sleep(delay)

    # ------------------------
    # Named Color Helper
    # ------------------------
    def set_color_by_name(self, light_name, color_name):
        """
        Set light color using common color names.
        Converts names to hue/saturation values.

        Args:
            light_name (str): Name of the light.
            color_name (str): Common color name, e.g. 'red', 'blue', 'warm white'.
        """
        color_name = color_name.lower().strip()

        # Hue-Saturation mapping (approximate)
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
        self.set_color(light_name, hue_val, sat_val)
        print(f"Set '{light_name}' to {color_name} (hue={hue_val}, sat={sat_val})")

    # ------------------------
    # Group Controls
    # ------------------------
    def set_group_state(self, group_name, state, value=None):
        """
        Generic helper for controlling a group.
        Example: set_group_state('Living Room', 'on', True)
        """
        try:
            if value is not None:
                self.bridge.set_group(group_name, state, value)
            else:
                self.bridge.set_group(group_name, state)
        except Exception as e:
            print(f"[ERROR] Could not control group '{group_name}': {e}")

    def turn_on_group(self, group_name):
        self.set_group_state(group_name, 'on', True)

    def turn_off_group(self, group_name):
        self.set_group_state(group_name, 'on', False)

    def set_group_brightness(self, group_name, bri):
        bri = max(1, min(254, int(bri)))
        self.set_group_state(group_name, 'bri', bri)

    # ------------------------
    # Query & Debug
    # ------------------------
    def list_lights(self):
        """Print all available light names."""
        print("Available lights:", list(self.lights.keys()))

    def get_light_state(self, light_name):
        """Return dict with the current state of a light."""
        if light_name in self.lights:
            return self.bridge.get_light(light_name)
        print(f"[WARN] Light '{light_name}' not found")
        return None