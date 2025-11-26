import os
from typing import Optional, Dict, Any, List

import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SpotifyController:
    """
    Thin wrapper around Spotipy to control Spotify playback.

    Features:
    - Authenticate via Spotify OAuth
    - Control playback on a chosen device
      * play / pause
      * next / previous track
      * volume (0–100 %)
      * shuffle on/off
      * repeat mode: off / track / context
    - Query current track info (title, artist, album, cover art URL)
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scope: Optional[str] = None,
        cache_path: str = ".cache-spotify-controller",
    ):
        """
        Initialize SpotifyController and perform OAuth.

        Args:
            client_id (str, optional): Spotify app client ID.
                                      Defaults to env SPOTIPY_CLIENT_ID.
            client_secret (str, optional): Spotify app client secret.
                                           Defaults to env SPOTIPY_CLIENT_SECRET.
            redirect_uri (str, optional): Redirect URI set in Spotify app settings.
                                          Defaults to env SPOTIPY_REDIRECT_URI.
            scope (str, optional): OAuth scopes. If None, a sensible default is used.
            cache_path (str): Path for token cache file.
        """
        client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
        redirect_uri = redirect_uri or os.getenv("SPOTIPY_REDIRECT_URI")

        if scope is None:
            # Playback control + currently playing + user library (optional)
            scope = (
                "user-read-playback-state "
                "user-modify-playback-state "
                "user-read-currently-playing"
            )

        if not client_id or not client_secret or not redirect_uri:
            raise ValueError(
                "Spotify credentials missing. Set SPOTIPY_CLIENT_ID, "
                "SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI or pass explicitly."
            )

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            cache_path=cache_path,
            open_browser=True,
        )

        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        self.active_device_id: Optional[str] = None

    # ------------------------
    # Device handling
    # ------------------------
    def list_devices(self) -> List[Dict[str, Any]]:
        """
        Return a list of available Spotify devices.

        Each device is a dict with keys like:
        - id
        - name
        - type
        - is_active
        - volume_percent
        """
        devices = self.sp.devices().get("devices", [])
        if not devices:
            print("[INFO] No active Spotify devices found. Open Spotify on a device.")
        else:
            print("Available Spotify devices:")
            for d in devices:
                active_mark = "*" if d.get("is_active") else " "
                print(
                    f" [{active_mark}] {d.get('name')} "
                    f"({d.get('type')}) id={d.get('id')} "
                    f"vol={d.get('volume_percent')}"
                )
        return devices

    def set_active_device(self, device_name: Optional[str] = None, device_id: Optional[str] = None):
        """
        Select the active device by name or device_id.

        If both are None, auto-selects the currently active device (if any).
        """
        devices = self.sp.devices().get("devices", [])

        if not devices:
            print("[WARN] No Spotify devices available.")
            self.active_device_id = None
            return

        chosen = None

        if device_id:
            for d in devices:
                if d.get("id") == device_id:
                    chosen = d
                    break
        elif device_name:
            for d in devices:
                if d.get("name") == device_name:
                    chosen = d
                    break
        else:
            # Try to find currently active device
            for d in devices:
                if d.get("is_active"):
                    chosen = d
                    break
            # If none active, just pick the first
            if chosen is None and devices:
                chosen = devices[0]

        if chosen is None:
            print("[WARN] Could not find matching device.")
            self.active_device_id = None
        else:
            self.active_device_id = chosen.get("id")
            print(f"[INFO] Active Spotify device set to: {chosen.get('name')}")

    def _ensure_device(self):
        """
        Internal helper to ensure we have an active device.
        Tries to auto-select if not already set.
        """
        if self.active_device_id is None:
            self.set_active_device()
        if self.active_device_id is None:
            raise RuntimeError(
                "No active Spotify device. Open Spotify on a device and try again."
            )

    # ------------------------
    # Playback control
    # ------------------------
    def play(self):
        """Resume playback on the active device."""
        self._ensure_device()
        self.sp.start_playback(device_id=self.active_device_id)

    def pause(self):
        """Pause playback on the active device."""
        self._ensure_device()
        self.sp.pause_playback(device_id=self.active_device_id)

    def toggle_play_pause(self):
        """
        Toggle between play and pause based on current playback state.
        """
        current = self.sp.current_playback()
        if current and current.get("is_playing"):
            self.pause()
        else:
            self.play()

    def next_track(self):
        """Skip to the next track."""
        self._ensure_device()
        self.sp.next_track(device_id=self.active_device_id)

    def previous_track(self):
        """Skip to the previous track."""
        self._ensure_device()
        self.sp.previous_track(device_id=self.active_device_id)

    # ------------------------
    # Volume and shuffle/repeat
    # ------------------------
    def set_volume(self, volume_percent: int):
        """
        Set volume on the active device.

        Args:
            volume_percent (int): 0–100
        """
        volume_percent = max(0, min(100, int(volume_percent)))
        self._ensure_device()
        self.sp.volume(volume_percent, device_id=self.active_device_id)

    def set_shuffle(self, shuffle: bool):
        """
        Enable or disable shuffle on the active device.

        Args:
            shuffle (bool): True = shuffle on, False = off
        """
        self._ensure_device()
        self.sp.shuffle(state=shuffle, device_id=self.active_device_id)

    def set_repeat(self, mode: str):
        """
        Set repeat mode on the active device.

        Args:
            mode (str): 'off', 'track', or 'context'
        """
        mode = mode.lower()
        if mode not in ("off", "track", "context"):
            print("[WARN] Invalid repeat mode. Use 'off', 'track', or 'context'.")
            return
        self._ensure_device()
        self.sp.repeat(state=mode, device_id=self.active_device_id)

    # ------------------------
    # Current track info
    # ------------------------
    def get_current_track_info(self) -> Optional[Dict[str, Any]]:
        """
        Get a dict with info about the currently playing track.

        Returns:
            dict with keys:
              - title
              - artists (list of names)
              - album
              - is_playing (bool)
              - progress_ms
              - duration_ms
              - image_url (album art, largest available)
            or None if nothing is playing.
        """
        current = self.sp.current_playback()

        if not current or not current.get("item"):
            print("[INFO] Nothing is currently playing.")
            return None

        item = current["item"]
        title = item.get("name")
        artists = [a["name"] for a in item.get("artists", [])]
        album = item.get("album", {}).get("name")

        images = item.get("album", {}).get("images", [])
        image_url = images[0]["url"] if images else None  # usually largest first

        info = {
            "title": title,
            "artists": artists,
            "album": album,
            "is_playing": current.get("is_playing", False),
            "progress_ms": current.get("progress_ms"),
            "duration_ms": item.get("duration_ms"),
            "image_url": image_url,
        }
        return info