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
        self._latest_frame = None            # raw BGR frame for line follow
        self._frame_queue = queue.Queue(maxsize=2)  # main-thread display queue

        # PID state
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = None
        self._vy_smooth = 0.0          # smoothed lateral output

        # AprilTag detector (tag36h11 family used in world)
        self._tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        self._tag_params = cv2.aruco.DetectorParameters()
        self._tag_detector = cv2.aruco.ArucoDetector(self._tag_dict, self._tag_params)
        self._tag_land_area = 20000  # px² — trigger landing only when drone is directly over pad

    # ── frame processing (runs in camera thread) ─────────────────────────────
    def process_frame(self, frame):
        """
        Detect colored objects in the frame, draw bounding boxes,
        show a live preview window, and store detections.
        """
        self._latest_frame = frame          # store raw frame for line_follow()
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

    def _detect_apriltag(self, frame):
        """Detect AprilTag in frame. Returns (tag_id, cx, cy, area, corners) or None."""
        corners, ids, _ = self._tag_detector.detectMarkers(frame)
        if ids is None or len(ids) == 0:
            return None
        best, best_area = None, 0
        for i, tc in enumerate(corners):
            c = tc[0]
            w = np.linalg.norm(c[0] - c[1])
            h = np.linalg.norm(c[0] - c[3])
            area = w * h
            if area > best_area:
                best_area = area
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))
                best = (int(ids[i][0]), cx, cy, area, tc)
        return best

    def _pid(self, error, kp=0.0007, ki=0.000002, kd=0.0045, deadband=12):
        """PID controller — returns smoothed output given pixel error.
        deadband: errors smaller than this many pixels are treated as zero.
        """
        if abs(error) < deadband:
            error = 0  # ignore tiny wobble
        now = time.time()
        dt = (now - self._pid_last_time) if self._pid_last_time else 0.05
        self._pid_last_time = now
        self._pid_integral += error * dt
        self._pid_integral = max(-150, min(150, self._pid_integral))   # anti-windup
        derivative = (error - self._pid_last_error) / max(dt, 1e-4)
        self._pid_last_error = error
        raw = kp * error + ki * self._pid_integral + kd * derivative
        # Exponential moving average — alpha=0.25 keeps only 25% new, 75% old
        self._vy_smooth = 0.25 * raw + 0.75 * self._vy_smooth
        return self._vy_smooth

    def line_follow(self, duration=30, forward_speed=0.2, land_on_tag=True, tag_ignore_secs=0):
        """
        PID line follower using the yellow line detected by the downward camera.
        Uses the bottom 40% of the frame as region-of-interest.
        tag_ignore_secs: suppress AprilTag detection for this many seconds at start.
        Returns the detected tag ID (int) if tag triggered exit, or None if timed out.
        """
        print(f"Line following for {duration}s  (forward={forward_speed} m/s, tag_ignore={tag_ignore_secs}s)")
        fw = 640  # frame width — will update from first frame
        tag_ignore_until = time.time() + tag_ignore_secs

        deadline = time.time() + duration
        while time.time() < deadline:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.02)
                continue

            fh, fw = frame.shape[:2]
            roi = frame[int(fh * 0.6):, :]          # bottom 40 % of frame

            # Detect yellow line in ROI
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lo_y, hi_y = COLOR_RANGES['yellow'][:2]
            mask = cv2.inRange(hsv_roi, np.array(lo_y), np.array(hi_y))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))

            M = cv2.moments(mask)
            if M['m00'] > 500:                      # line found
                cx = int(M['m10'] / M['m00'])       # centroid x in ROI
                error = cx - fw // 2                # +ve = line is right → steer right
                vy = self._pid(error)
                vy = max(-0.25, min(0.25, vy))      # tighter clamp for smoother flight
                status = f"Line @ x={cx}  err={error:+d}  vy={vy:+.3f}"

                # Draw line position on latest display frame
                disp = frame.copy()
                cv2.line(disp, (cx, int(fh * 0.6)), (cx, fh), (0, 255, 255), 2)
                cv2.putText(disp, status, (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:                                   # line lost — stop moving laterally
                vy = 0.0
                status = "Line LOST"
                disp = frame.copy()
                cv2.putText(disp, status, (8, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ── AprilTag detection — scan entire frame (skipped during ignore window) ──
            if time.time() >= tag_ignore_until:
                tag = self._detect_apriltag(frame)
            else:
                tag = None  # still in ignore window

            if tag is not None:
                tid, tcx, tcy, tarea, tcorners = tag
                print(f"[TAG] AprilTag ID={tid}  area={tarea:.0f}  center=({tcx},{tcy})")
                cv2.aruco.drawDetectedMarkers(disp, [tcorners], np.array([[tid]]))
                cv2.putText(disp, f"TAG ID={tid}", (tcx - 30, tcy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if tarea >= self._tag_land_area:
                    print(f"[TAG] Tag large enough (area={tarea:.0f}) — stopping.")
                    self.control.set_velocity(0, 0, 0)
                    time.sleep(0.3)
                    if land_on_tag:
                        print("[TAG] Preparing to land on box...")
                        self._center_on_box()
                    # Push final frame before breaking
                    try:
                        self._frame_queue.put_nowait(disp)
                    except queue.Full:
                        pass
                    return tid  # return the detected tag ID

            self.control.set_velocity(vx=forward_speed, vy=vy, vz=0)

            # Push to display queue
            try:
                self._frame_queue.put_nowait(disp)
            except queue.Full:
                pass

            # Show frame on main thread
            try:
                f = self._frame_queue.get_nowait()
                cv2.imshow("Drone Camera", f)
            except queue.Empty:
                pass
            cv2.waitKey(1)

        # Stop movement when done
        self.control.set_velocity(0, 0, 0)
        print("Line following complete.")
        return None  # timed out, no tag triggered

    def _detect_box_center(self, frame):
        """
        Detect the landing-pad box (a large roughly-square gray/white object)
        by finding the biggest near-square contour in the frame.
        Returns (cx, cy, w, h) or None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Threshold to isolate the lighter box against the darker floor
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                  np.ones((15, 15), np.uint8))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                  np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 3000  # minimum area threshold
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < best_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            if aspect < 0.4:  # skip very elongated shapes
                continue
            if area > best_area:
                best_area = area
                cx = x + w // 2
                cy = y + h // 2
                best = (cx, cy, w, h)
        return best

    def _center_on_box(self, timeout=25.0):
        """
        Detect the physical landing box (gray/white rectangle), nudge toward
        its centre, then descend slowly. No re-centering — commit and land.
        """
        # ── Step 0: full stop and hover ──────────────────────────────────
        print("[BOX] Stopping to locate box...")
        self.control.set_velocity(0, 0, 0)
        time.sleep(2.0)

        # ── Step 1: detect box and nudge toward its centre ───────────────
        print("[BOX] Detecting box shape...")
        deadline = time.time() + 12.0
        nudged = False
        while time.time() < deadline:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            fh, fw = frame.shape[:2]
            box = self._detect_box_center(frame)
            disp = frame.copy()
            if box is not None:
                bcx, bcy, bw, bh = box
                ex = bcx - fw // 2
                ey = bcy - fh // 2
                # Draw box outline
                cv2.rectangle(disp, (bcx - bw // 2, bcy - bh // 2),
                              (bcx + bw // 2, bcy + bh // 2), (255, 0, 255), 2)
                cv2.circle(disp, (bcx, bcy), 6, (255, 0, 255), -1)
                cv2.putText(disp, f"BOX ex={ex:+d} ey={ey:+d} ({bw}x{bh})",
                            (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                print(f"[BOX] centre=({bcx},{bcy}) ex={ex:+d} ey={ey:+d} size={bw}x{bh}")

                # Nudge: move proportionally, cap at 0.10 m/s, for 1.5–4 s
                vy = max(-0.10, min(0.10, ex * 0.0015))
                vx = max(-0.10, min(0.10, ey * 0.0015))
                nudge_time = max(1.5, min(4.0, max(abs(ex), abs(ey)) * 0.015))
                print(f"[BOX] Nudging vx={vx:+.3f} vy={vy:+.3f} for {nudge_time:.1f}s")
                t_end = time.time() + nudge_time
                while time.time() < t_end:
                    self.control.set_velocity(vx=vx, vy=vy, vz=0)
                    time.sleep(0.1)
                self.control.set_velocity(0, 0, 0)
                time.sleep(1.0)  # settle
                nudged = True
                break
            else:
                cv2.putText(disp, "BOX: searching...", (8, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self._push_display(disp)
            time.sleep(0.1)

        if not nudged:
            print("[BOX] Could not detect box — landing at current position.")

        # ── Step 2: slow straight descent — no re-centering ──────────────
        print("[BOX] Descending slowly...")
        descent_deadline = time.time() + timeout
        while time.time() < descent_deadline:
            alt_msg = self.control.master.recv_match(
                type='GLOBAL_POSITION_INT', blocking=False)
            if alt_msg:
                alt = alt_msg.relative_alt / 1000.0
                if alt < 0.25:
                    print(f"[BOX] Altitude {alt:.2f} m — handing off to land.")
                    self.control.set_velocity(0, 0, 0)
                    break
            frame = self._latest_frame
            if frame is not None:
                disp = frame.copy()
                cv2.putText(disp, "LANDING...", (8, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self._push_display(disp)
            # Very slow descent, no lateral movement
            self.control.set_velocity(vx=0, vy=0, vz=0.06)
            time.sleep(0.05)

        self.control.set_velocity(0, 0, 0)

    def _push_display(self, disp):
        """Helper: push frame to queue and show on main thread."""
        try:
            self._frame_queue.put_nowait(disp)
        except queue.Full:
            pass
        try:
            cv2.imshow("Drone Camera", self._frame_queue.get_nowait())
        except queue.Empty:
            pass
        cv2.waitKey(1)

    def _reset_pid(self):
        """Reset PID state (call before a new line-follow phase)."""
        self._pid_integral = 0.0
        self._pid_last_error = 0.0
        self._pid_last_time = None
        self._vy_smooth = 0.0

    # ── main sequence ────────────────────────────────────────────────────────
    def start(self):
        """Force arm → takeoff → line1 → tag1: turn 90° CW → line2 → tag2: land"""
        print("MAVLink connected. Starting flight sequence...")
        tag_ids = []  # collect detected AprilTag IDs

        # 1. Set GUIDED mode
        self.control.set_mode('GUIDED')

        # 2. Force arm
        self.control.force_arm()

        # 3. Takeoff to 1.5 m — high enough for camera to see full tag box
        self.control.takeoff(1.5)

        # 4. Start camera
        self.camera.start_thread(self.process_frame)
        print("Camera started. Waiting for first frame...")
        while self._latest_frame is None:
            time.sleep(0.05)

        # 5. Phase 1 — follow line until first AprilTag is close enough
        print("=== Phase 1: following line to first AprilTag ===")
        tag1_id = self.line_follow(duration=120, forward_speed=0.25, land_on_tag=False)

        if tag1_id is not None:
            tag_ids.append(tag1_id)
            print(f"=== Phase 1 complete: detected AprilTag ID={tag1_id} ===")

            # 6. Turn 90° clockwise
            print("=== Turning 90° clockwise ===")
            self.control.turn_yaw(90)
            time.sleep(1.0)   # settle after turn

            # 7. Reset PID for fresh start on new line
            self._reset_pid()

            # 8. Phase 2 — follow next line to second AprilTag, then land
            print("=== Phase 2: following line to second AprilTag ===")
            tag2_id = self.line_follow(duration=120, forward_speed=0.25,
                                       land_on_tag=True, tag_ignore_secs=10)
            if tag2_id is not None:
                tag_ids.append(tag2_id)
                print(f"=== Phase 2 complete: detected AprilTag ID={tag2_id} ===")
        else:
            print("Timed out on phase 1 without finding tag — landing.")

        # 9. Land
        self.control.land()
        print("Flight sequence complete.")
        cv2.destroyAllWindows()

        # ── Print detected AprilTag IDs ──────────────────────────────────
        print("")
        print("=" * 50)
        print("  DETECTED APRILTAG IDs")
        print("=" * 50)
        if len(tag_ids) >= 1:
            print(f"  Tag 1 (phase 1):  ID = {tag_ids[0]}")
        if len(tag_ids) >= 2:
            print(f"  Tag 2 (phase 2):  ID = {tag_ids[1]}")
        if not tag_ids:
            print("  No tags detected.")
        print("=" * 50)

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