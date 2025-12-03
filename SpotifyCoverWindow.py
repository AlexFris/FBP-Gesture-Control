import tkinter as tk
from PIL import Image, ImageTk
import requests
from io import BytesIO


class SpotifyCoverWindow:
    """
    GUI window that displays the current Spotify track cover art.
    - Borderless / frameless window
    - Draggable by clicking anywhere on the cover
    - Automatically refreshes and resizes art
    """

    def __init__(self, spotify_controller, refresh_ms=1000, min_width=640):
        """
        Args:
            spotify_controller (SpotifyController): your wrapper instance.
            refresh_ms (int): refresh interval in ms.
            min_width (int): minimum image width (prevents 1×1 glitch).
        """
        self.spotify = spotify_controller
        self.refresh_ms = refresh_ms
        self.min_width = min_width

        self.last_image_url = None
        self.photo = None

        # ---- Borderless Tkinter Window ----
        self.root = tk.Tk()
        self.root.overrideredirect(True)      # Remove title bar completely
        self.root.attributes("-topmost", True)  # Keep always on top (optional)

        self.root.geometry(f"{min_width}x{min_width}")
        self.root.minsize(min_width, min_width)

        # --- Draggable window offsets ---
        self._drag_x = 0
        self._drag_y = 0

        # Label to display cover art (also used for dragging)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(expand=True, fill="both")

        # Bind dragging
        self.label.bind("<Button-1>", self.start_move)
        self.label.bind("<B1-Motion>", self.on_move)

        # Optional: right-click to close
        self.label.bind("<Button-3>", lambda e: self.root.destroy())

        # Start update loop
        self.update_cover()

    # ------------------------------------------------------
    # Window dragging behavior
    # ------------------------------------------------------
    def start_move(self, event):
        """Record mouse position for dragging."""
        self._drag_x = event.x
        self._drag_y = event.y

    def on_move(self, event):
        """Move the window according to mouse drag."""
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ------------------------------------------------------
    # Helper: Download cover art
    # ------------------------------------------------------
    def download_image(self, url):
        """Download an image URL and return a PIL Image."""
        response = requests.get(url)
        img_data = BytesIO(response.content)
        return Image.open(img_data)

    # ------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------
    def update_cover(self):
        """Update the displayed cover art whenever song changes."""

        self.root.update_idletasks()
        w = self.root.winfo_width()

        # Fix: If Tk reports 1×1 early in startup, use fallback
        if w < self.min_width:
            w = self.min_width

        # Ask Spotify directly (forces refresh)
        info = self.spotify.get_current_track_info()

        if info and info.get("image_url"):
            url = info["image_url"]

            # Only download when artwork changes
            if url != self.last_image_url:
                try:
                    img = self.download_image(url)

                    # Resize while preserving aspect ratio
                    aspect = img.height / img.width
                    new_h = int(w * aspect)
                    img = img.resize((w, new_h), Image.LANCZOS)

                    # Convert to Tkinter PhotoImage
                    self.photo = ImageTk.PhotoImage(img)

                    # Update label
                    self.label.config(image=self.photo)

                    self.last_image_url = url

                except Exception as e:
                    print(f"[ERROR] failed to update cover art: {e}")

        # Schedule next update
        self.root.after(self.refresh_ms, self.update_cover)

    def run(self):
        """Start Tkinter main loop."""
        self.root.mainloop()
