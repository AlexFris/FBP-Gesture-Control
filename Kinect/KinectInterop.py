import ctypes
import numpy as np
import os

# Load DLL
dll_path = os.path.join(os.path.dirname(__file__), "KinectV2DLL.dll")
kinect = ctypes.WinDLL(dll_path)

# -----------------------------------
# Function signatures (VERY important)
# -----------------------------------

# int InitKinect();   → returns 1 on success
kinect.InitKinect.restype = ctypes.c_int

# int GetDepthAtColorPixel(int cx, int cy, unsigned short* outDepth);
kinect.GetDepthAtColorPixel.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_ushort)
]
kinect.GetDepthAtColorPixel.restype = ctypes.c_int

# void ShutdownKinect();
kinect.ShutdownKinect.restype = None


# -----------------------------------
# Python-friendly functions
# -----------------------------------

def init_kinect():
    """Start the Kinect sensor."""
    result = kinect.InitKinect()
    if result != 1:
        raise RuntimeError("Kinect initialization failed.")
    print("[KINECT] Initialized successfully")


def get_depth_at_color_pixel(cx, cy):
    """Return depth in millimeters for a given RGB pixel coordinate."""
    depth_value = ctypes.c_ushort(0)
    ok = kinect.GetDepthAtColorPixel(cx, cy, ctypes.byref(depth_value))
    if ok != 1:
        return None
    return int(depth_value.value)


def shutdown_kinect():
    """Stop Kinect sensor."""
    kinect.ShutdownKinect()
    print("[KINECT] Shutdown")
