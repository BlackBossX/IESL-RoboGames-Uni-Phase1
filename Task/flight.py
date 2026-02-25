'''
This contains the opencv, line following stuff. This will make the decisions for the robot based on what the camera sees and 
send commands to the controller.
'''
import cv2
import numpy as np
from control import Control
import time
from sensor import Camera

class Brain:
    def __init__(self):
        self.control = Control()
        self.camera = Camera()

    def process_frame(self, frame):
        """
        Process the camera frame to detect lines and make decisions.
        use Camera class to get the frames.
        """
        pass
        

        
    
    def line_follow(self):
        """
        Main line following logic.
        Consider PID control for smoother movement.
        Use the control class to send movement commands based on the detected line position.
        """

        pass

    def start(self):
        """Force arm → takeoff 0.27 m → land"""
        print("MAVLink connected. Starting flight sequence...")

        # 1. Set GUIDED mode
        self.control.set_mode('GUIDED')

        # 2. Force arm (bypasses pre-arm checks)
        self.control.force_arm()

        # 3. Takeoff to 0.27 m
        self.control.takeoff(0.27)

        # 4. Hover briefly so the altitude is stable
        print("Hovering at 0.27 m for 3 seconds...")
        time.sleep(3)

        # 5. Land back to ground
        self.control.land()
        print("Flight sequence complete.")
    
    def __del__(self):
        """Destructor to ensure threads are stopped"""
        self.camera.stop_thread()


if __name__ == "__main__":
    brain = Brain()
    try:
        brain.start()
    except KeyboardInterrupt:
        print("Stopping brain...")
    finally:
        del brain