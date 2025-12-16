from SpotifyWrapper import SpotifyController
from SpotifyCoverWindow import SpotifyCoverWindow

def main():
    spc = SpotifyController()
    #spc.list_devices()           # See devices
    spc.set_active_device()      # Auto-pick active device or first device

    # Basic control
    #spc.play()
    # spc.pause()
    # spc.next_track()
    #spc.previous_track()
    # spc.set_volume(40)
    # spc.set_shuffle(True)
    # spc.set_repeat("track")  # or "off", "context"

    info = spc.get_current_track_info()
    if info:
        print("Now playing:")
        print(f"  {info['title']} – {', '.join(info['artists'])}")
        print(f"  Album: {info['album']}")
        print(f"  Cover: {info['image_url']}")

    win = SpotifyCoverWindow(spc, refresh_ms= 100)
    win.run()

if __name__ == "__main__":
    main()