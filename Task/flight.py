'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import cv2
import numpy as np
from control import Control
import time
from sensor import Camera
import queue
import os

# Force OpenCV to use X11/XCB backend (avoids Wayland Qt crash)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# ── HSV color ranges for detection (H: 0-179, S: 0-255, V: 0-255) ──────────
COLOR_RANGES = {
    'red':    ([  0,  80, 80], [ 10, 255, 255],   # lower red hue
               [160,  80, 80], [179, 255, 255]),   # upper red hue (wraps)
    'blue':   ([ 90, 80, 80], [130, 255, 255], None, None),
    'green':  ([ 35, 60, 60], [ 85, 255, 255], None, None),
    'yellow': ([ 20, 80, 80], [ 35, 255, 255], None, None),
}

# BGR colors for drawing bounding boxes
DRAW_COLOR = {
    'red':    (0,   0,   255),
    'blue':   (255, 0,   0),
    'green':  (0,   255, 0),
    'yellow': (0,   200, 255),
}

MIN_CONTOUR_AREA = 500   # px² — ignore tiny blobs


class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()
        self._latest_detections = []
        self._frame_queue = queue.Queue(maxsize=2)  # main-thread display queue

    # ── frame processing (runs in camera thread) ─────────────────────────────
    def process_frame(self, frame):
        """
        Detect colored objects in the frame, draw bounding boxes,
        show a live preview window, and store detections.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        display = frame.copy()
        detections = []

        for color_name, ranges in COLOR_RANGES.items():
            lo1, hi1, lo2, hi2 = ranges

            mask = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
            if lo2 is not None:                          # red wraps around 0/180
                mask |= cv2.inRange(hsv, np.array(lo2), np.array(hi2))

            # Clean up the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                detections.append({
                    'color': color_name,
                    'bbox':  (x, y, w, h),
                    'center': (cx, cy),
                    'area':  area,
                })

                # Draw bounding box + label
                bgr = DRAW_COLOR[color_name]
                cv2.rectangle(display, (x, y), (x + w, y + h), bgr, 2)
                label = f"{color_name} ({area:.0f}px)"
                cv2.putText(display, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)
                cv2.circle(display, (cx, cy), 4, bgr, -1)

        # Frame-centre crosshair
        fh, fw = display.shape[:2]
        cv2.line(display, (fw // 2 - 15, fh // 2),
                           (fw // 2 + 15, fh // 2), (200, 200, 200), 1)
        cv2.line(display, (fw // 2, fh // 2 - 15),
                           (fw // 2, fh // 2 + 15), (200, 200, 200), 1)

        info = f"Objects: {len(detections)}"
        cv2.putText(display, info, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Push annotated frame to main thread for display (non-blocking)
        try:
            self._frame_queue.put_nowait(display)
        except queue.Full:
            pass  # drop frame if main thread is busy

        self._latest_detections = detections
        if detections:
            for d in detections:
                print(f"[DETECT] {d['color']:6s}  area={d['area']:6.0f}  "
                      f"center={d['center']}")

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_detections(self):
        """Return the most recently detected objects (thread-safe read)."""
        return list(self._latest_detections)

    def line_follow(self):
        """
        Main line following logic.
        Consider PID control for smoother movement.
        Use the control class to send movement commands based on the detected line position.
        """
        pass

    # ── main sequence ────────────────────────────────────────────────────────
    def start(self):
        """Force arm → takeoff 3 m → hover & track → land"""
        print("MAVLink connected. Starting flight sequence...")

        # 1. Set GUIDED mode
        self.control.set_mode('GUIDED')

        # 2. Force arm (bypasses pre-arm checks)
        self.control.force_arm()

        # 3. Takeoff to 3 m
        self.control.takeoff(3.0)

        # 4. Start camera now that drone is airborne
        self.camera.start_thread(self.process_frame)
        print("Camera started — tracking objects for 10 seconds...")

        # 5. Display loop runs on main thread (required by Qt/GTK)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                frame = self._frame_queue.get(timeout=0.05)
                cv2.imshow("Drone Camera", frame)
            except queue.Empty:
                pass
            cv2.waitKey(1)
        cv2.destroyAllWindows()

        # 5. Land back to ground
        self.control.land()
        print("Flight sequence complete.")

    def __del__(self):
        """Destructor to ensure threads and windows are stopped"""
        self.camera.stop_thread()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain