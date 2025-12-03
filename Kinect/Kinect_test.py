from KinectInterop import init_kinect, get_depth_at_color_pixel, shutdown_kinect
import time

init_kinect()

print("Querying depth values. Move your hand in front of Kinect...")

try:
    while True:
        # Pick a pixel in the middle of the RGB image
        cx, cy = 960, 540   # 1920×1080 / 2
        d = get_depth_at_color_pixel(cx, cy)
        print("Depth at center:", d)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopping...")

shutdown_kinect()
