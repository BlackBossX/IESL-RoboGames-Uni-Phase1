import socket
import struct
import threading
import numpy as np
import cv2
import time
from pymavlink import mavutil

Airports = [1, 2]

TAKEOFF_ALTITUDE_M = 0.20
CAMERA_HOST = "localhost"
CAMERA_PORT = 5599

# Global thread-safe camera frame
current_frame = None
frame_lock = threading.Lock()
camera_running = True

def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError(f"Camera stream closed unexpectedly (received {len(data)}/{size} bytes)")
        data.extend(chunk)
    return bytes(data)

def camera_thread_loop(host: str, port: int):
    global current_frame, camera_running
    print(f"Connecting to camera stream on tcp://{host}:{port}...")

    while camera_running:
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                print("Camera connected.")
                while camera_running:
                    # 1. Header (4 bytes): width, height
                    header = recv_exact(sock, 4)
                    width, height = struct.unpack("=HH", header)
                    
                    # Ensure width and height are valid to prevent massive memory allocations
                    if width <= 0 or height <= 0 or width > 2000 or height > 2000:
                        raise ConnectionError(f"Invalid frame size received: {width}x{height}")
                        
                    # 2. Payload (width * height * 3 bytes for RGB)
                    payload_size = width * height * 3
                    payload = recv_exact(sock, payload_size)
                    
                    # 3. Convert to numpy array and reshape
                    img_array = np.frombuffer(payload, dtype=np.uint8)
                    frame_rgb = img_array.reshape((height, width, 3))
                    
                    # 4. Save converted BGR frame for OpenCV (thread-safe)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    with frame_lock:
                        current_frame = frame_bgr.copy()
                        
        except Exception as e:
            if camera_running:
                print(f"Camera stream disconnected: {e}. Retrying in 1s...")
                time.sleep(1)

def display_camera_feed():
    global current_frame, camera_running
    cv2.namedWindow("Drone Camera Feed", cv2.WINDOW_NORMAL)
    try:
        while camera_running:
            with frame_lock:
                frame = current_frame.copy() if current_frame is not None else None
                
            if frame is not None:
                # Example: You can insert your AprilTag/Image recognitions logic here
                # cv2.putText(frame, "Processing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Drone Camera Feed", frame)
                
            # Allow GUI to update, press 'q' to quit early
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("User requested camera quit.")
                break
    finally:
        cv2.destroyAllWindows()


def connect_vehicle() -> mavutil.mavfile:
    endpoints = ["udp:0.0.0.0:14550", "tcp:localhost:5760"]
    last_error = None

    for endpoint in endpoints:
        try:
            print(f"Connecting to vehicle on {endpoint}...")
            master = mavutil.mavlink_connection(endpoint)
            master.wait_heartbeat(timeout=15)
            print(f"Heartbeat from system={master.target_system}, component={master.target_component}")
            return master
        except Exception as error:
            last_error = error
            print(f"Connection failed on {endpoint}: {error}")

    raise RuntimeError(f"Unable to connect to vehicle: {last_error}")

def set_mode(master: mavutil.mavfile, mode: str) -> None:
    mode_map = master.mode_mapping()
    if mode_map is None or mode not in mode_map:
        raise RuntimeError(f"Mode '{mode}' is not supported by this vehicle")
        
    mode_id = mode_map[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )

    deadline = time.time() + 8
    while time.time() < deadline:
        heartbeat = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if heartbeat is None:
            continue
        if heartbeat.custom_mode == mode_id:
            print(f"Mode changed to {mode}")
            return

    raise TimeoutError(f"Timed out while switching to mode '{mode}'")

def wait_altitude(master: mavutil.mavfile, min_alt_m: float, timeout_s: float = 20) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is None:
            continue
        rel_alt_m = msg.relative_alt / 1000.0
        print(f"Current altitude: {rel_alt_m:.2f} m")
        if rel_alt_m >= min_alt_m:
            return

    raise TimeoutError("Timed out waiting for target altitude")

def arm_and_takeoff(master: mavutil.mavfile, target_alt_m: float) -> None:
    set_mode(master, "GUIDED")

    print("Arming motors...")
    while True:
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,
            21196,
            0,
            0,
            0,
            0,
            0,
        )
        msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg and msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            break
        print("Waiting for motors to arm...")

    print("Vehicle armed")

    print(f"Taking off to {target_alt_m:.2f} m...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        target_alt_m,
    )

    wait_altitude(master, max(0.15, target_alt_m * 0.9))
    print("Takeoff complete")

def land(master: mavutil.mavfile) -> None:
    print("Initiating slow landing...")
    
    # Stay in GUIDED mode and command small downwards velocity until close to ground
    target_descent_speed_m_s = 0.05
    
    # Send velocity commands in GUIDED mode
    deadline = time.time() + 45
    while time.time() < deadline:
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1)
        if msg is None:
            continue
        rel_alt_m = msg.relative_alt / 1000.0
        print(f"Landing altitude: {rel_alt_m:.2f} m")
        
        # If very close to the ground, switch to LAND mode to let autopilot handle touchdown detection
        if rel_alt_m <= 0.05:
            print("Near ground, switching to LAND mode for final touchdown...")
            set_mode(master, "LAND")
            # Wait for disarm or timeout
            touchdown_deadline = time.time() + 10
            while time.time() < touchdown_deadline:
                hb = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
                if hb and not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    print("Landed successfully (motors disarmed automatically).")
                    return
            print("Landing completed.")
            return
            
        # Send velocity target (NED frame: Positive Z is down)
        master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            # Type mask: ignore position and acceleration, use velocity
            int(0b110111000111),
            0, 0, 0,  # x, y, z positions
            0, 0, target_descent_speed_m_s,  # x, y, z velocity in m/s 
            0, 0, 0,  # x, y, z acceleration
            0, 0  # yaw, yaw_rate
        )

    print("Landing timeout reached; vehicle may still be descending")

def disarm_vehicle(master: mavutil.mavfile) -> None:
    print("Disarming motors...")
    # Standard disarm without forcing
    master.arducopter_disarm()
    master.motors_disarmed_wait()
    print("Vehicle disarmed.")

def main() -> None:
    global camera_running

    # 1. Start the camera receiving thread
    cam_thread = threading.Thread(target=camera_thread_loop, args=(CAMERA_HOST, CAMERA_PORT), daemon=True)
    cam_thread.start()

    # 2. Wait for the first frame to arrive before proceeding with flight
    print("Waiting for camera feed...")
    while current_frame is None:
        time.sleep(0.1)
    print("Camera feed active!")

    # 3. Start a thread for MAVLink automated flight operations so we don't block the main thread
    def flight_sequence():
        vehicle = connect_vehicle()
        try:
            arm_and_takeoff(vehicle, TAKEOFF_ALTITUDE_M)
            print("Holding altitude for 5 seconds...")
            time.sleep(5)
            land(vehicle)
            disarm_vehicle(vehicle)
        finally:
            vehicle.close()
            # Let the main thread know we are finishing
            global camera_running
            camera_running = False

    flight_thread = threading.Thread(target=flight_sequence)
    flight_thread.start()

    # 4. Display the OpenCV camera feed in the main thread (OpenCV requires GUI to run in main thread)
    display_camera_feed()
    
    # 5. Bring everything down cleanly when the camera UI exits or flight is done
    camera_running = False
    flight_thread.join()

if __name__ == "__main__":
    main()
