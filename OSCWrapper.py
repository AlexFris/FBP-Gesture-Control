import os

from pythonosc.udp_client import SimpleUDPClient

class TDOSC:

    def __init__(self, ip: str = None, port: int = None, verbose=False):
        """
        Args:
            ip (str): TouchDesigner machine IP (localhost by default)
            port (int): OSC In DAT/CHOP port (TD default often 8000)
            verbose (bool): prints messages being sent
        """
        ip = ip or os.getenv("OSC_IP")
        port = port or os.getenv("OSC_PORT")

        self.available = False
        self.verbose = verbose
        self.client = None

        if not ip or not port:
            print("[OSC] Credentials missing. Set OSC_IP and OSC_PORT.")
            return

        try:
            port = int(port)  # IMPORTANT
            self.client = SimpleUDPClient(ip, port)
            self.available = True
            print(f"[OSC] Connected to {ip}:{port}")

        except Exception as e:
            print(f"[OSC] Failed to initialize client: {e}")

    # ------------------------------------------------------------
    # Generic send function with auto type handling
    # ------------------------------------------------------------
    def send(self, address: str, value):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        """
        Send an OSC message with automatic type handling.

        Args:
            address (str): OSC path, e.g. "/spotify/title"
            value: str, int, float, bool, list
        """
        if isinstance(value, bool):
            value = int(value)

        if isinstance(value, (str, int, float)):
            self.client.send_message(address, value)
        elif isinstance(value, list):
            self.client.send_message(address, value)
        else:
            raise TypeError(f"Unsupported OSC type: {type(value)}")

        if self.verbose:
            print(f"[OSC] {address} → {value}")

    # ------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------
    def send_string(self, address: str, text: str):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        self.send(address, text)

    def send_float(self, address: str, number: float):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        self.send(address, float(number))

    def send_int(self, address: str, number: int):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        self.send(address, int(number))

    def send_bool(self, address: str, state: bool):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        self.send(address, int(state))

    def ping(self):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        """Send a test message to confirm connection."""
        self.send("/ping", 1)

    def disconnect(self):
        if not self.available:  # If no OSC channel is set up or set up properly then skip
            return
        """Send a test message to confirm connection."""
        self.send("/ping", 0)