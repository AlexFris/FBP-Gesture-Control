import cv2
import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime


class KinectManager:
    """High-level interface to Kinect v2 RGB, Depth, and IR streams."""

    COLOR_W, COLOR_H = 1920, 1080
    DEPTH_W, DEPTH_H = 512, 424
    IR_W, IR_H       = 512, 424

    def __init__(self):
        """Initialize Kinect and prepare streams."""
        print("[Kinect] Initializing...")

        self.kinect = PyKinectRuntime.PyKinectRuntime(
            PyKinectV2.FrameSourceTypes_Color |
            PyKinectV2.FrameSourceTypes_Depth |
            PyKinectV2.FrameSourceTypes_Infrared
        )

        print("[Kinect] Ready")

    # --------------------------------------------------------------
    # RGB FEED
    # --------------------------------------------------------------
    def get_rgb(self):
        """Returns RGB frame as a (1080x1920x3) BGR numpy array, or None."""
        if not self.kinect.has_new_color_frame():
            return None

        frame = self.kinect.get_last_color_frame()
        color = frame.reshape((self.COLOR_H, self.COLOR_W, 4)).astype(np.uint8)
        color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)
        return color

    # --------------------------------------------------------------
    # DEPTH FEED
    # --------------------------------------------------------------
    def get_depth(self, normalize=True):
        """
        Returns raw depth OR a visible depth map.

        normalize=True → returns an 8-bit grayscale depth image
        normalize=False → returns the original uint16 depth frame
        """
        if not self.kinect.has_new_depth_frame():
            return None

        depth = self.kinect.get_last_depth_frame()
        depth = depth.reshape((self.DEPTH_H, self.DEPTH_W))

        if not normalize:
            return depth  # original uint16 depth

        # Visible depth (for display)
        depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
        return depth_vis

    # --------------------------------------------------------------
    # IR FEED
    # --------------------------------------------------------------
    def get_ir(self, normalize=True):
        """
        Returns IR frame (grayscale).

        normalize=True  → 8-bit visible IR
        normalize=False → raw uint16 IR
        """
        if not self.kinect.has_new_infrared_frame():
            return None

        ir = self.kinect.get_last_infrared_frame()
        ir = ir.reshape((self.IR_H, self.IR_W))

        if not normalize:
            return ir  # original uint16 IR

        ir_vis = cv2.convertScaleAbs(ir, alpha=0.05)
        return ir_vis

    # --------------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------------
    def close(self):
        """Safely shut down Kinect."""
        if self.kinect is not None:
            print("[Kinect] Closing...")
            self.kinect.close()
            self.kinect = None
            print("[Kinect] Closed")
