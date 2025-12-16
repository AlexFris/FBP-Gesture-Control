import argparse
import cv2
import time

############################
# Camera utilities
############################

def setup_camera(device=0, width=1920, height=1080):
    """
    Initialize the webcam and set capture properties.

    Args:
        device (int): Camera device index (default 0).
        width (int): Frame width (default 1920).
        height (int): Frame height (default 1080).

    Returns:
        cv2.VideoCapture: OpenCV video capture object.
    """
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise ValueError(f"Cannot open camera device {device}")

    return cap

def read_frame(cap, flip_horizontal=True):
    """
    Safely read a frame from the camera and optionally flip it horizontally.

    Args:
        cap (cv2.VideoCapture): Camera capture object.
        flip_horizontal (bool): Whether to flip the frame horizontally.

    Returns:
        (success: bool, frame: ndarray)
    """
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from camera.")
        return success, None

    if flip_horizontal:
        frame = cv2.flip(frame, 1)  # 1 = horizontal flip

    return success, frame

def parse_camera_args():
    """
    Parse command-line arguments for camera configuration.
    Returns an argparse.Namespace with device, width, and height.
    """
    parser = argparse.ArgumentParser(description="Camera configuration options")
    parser.add_argument('--device', type=int, default=0, help='Camera device index (default: 0)')
    parser.add_argument('--width', type=int, default=1920, help='Capture width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Capture height (default: 1080)')
    return parser.parse_args()

def release_camera(cap):
    """
    Release the camera resource safely.

    Args:
        cap (cv2.VideoCapture): Camera object to release.
    """
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

def update_fps(prev_time, img=None, draw=True, position=(1700, 40),
               color=(255, 0, 255), scale=3, thickness=3):
    """
    Calculate FPS and optionally draw it on an image.

    Args:
        prev_time (float): Timestamp from previous frame.
        img (ndarray, optional): Image to draw FPS text on.
        draw (bool): Whether to draw FPS on the image.
        position (tuple): Position of the text.
        color (tuple): Text color.
        scale (float): Font scale.
        thickness (int): Text thickness.

    Returns:
        tuple: (current_time, fps)
    """
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0

    if draw and img is not None:
        cv2.putText(img, f"FPS: {int(fps)}", position,
                    cv2.FONT_HERSHEY_PLAIN, scale, color, thickness)

    return current_time, fps