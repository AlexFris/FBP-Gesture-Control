# Test_SendSpotifyToTD.py
import time
from SpotifyWrapper import SpotifyController
from OSCWrapper import TDOSC


def main():
    # ----- 1. Start OSC connection -----
    osc = TDOSC(ip="192.168.68.107", port=8001, verbose=True)
    osc.ping()

    # ----- 2. Start Spotify controller -----
    sp = SpotifyController()

    # Make sure we have an active device
    sp.set_active_device()

    # ----- 3. Send updates continuously -----
    print("Sending Spotify info to TouchDesigner…")

    while True:
        info = sp.get_current_track_info()

        if info:
            # TEXT DATA
            osc.send_string("/spotify/title", info["title"])
            osc.send_string("/spotify/artist", ", ".join(info["artists"]))
            osc.send_string("/spotify/album", info["album"])
            osc.send_string("/spotify/image_url", info["image_url"] or "")

            # NUMBERS
            osc.send_int("/spotify/duration_ms", info["duration_ms"] or 0)
            osc.send_int("/spotify/progress_ms", info["progress_ms"] or 0)
            osc.send_bool("/spotify/is_playing", info["is_playing"])

        else:
            # Send "nothing playing" state
            osc.send_bool("/spotify/is_playing", False)

        time.sleep(0.5)  # update rate


if __name__ == "__main__":
    main()
