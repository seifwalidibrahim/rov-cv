import time
import logging
import sys
import os

# Add the streaming directory to the path so we can import the camera
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'streaming(pi)')))
from camera import StereoCamera
from scanning import AutonomousTransectScanner

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Dummy sensors since we haven't wired up the real ROV hardware yet
class DummyIMU:
    class Orientation:
        def __init__(self):
            self.roll, self.pitch, self.yaw = 0.0, 0.0, 0.0
    def read_orientation(self): return self.Orientation()

class DummyDepth:
    def read_depth(self): return 5.0

if __name__ == "__main__":
    camera_instance = None
    try:
        logging.info("Warming up standard laptop webcam (index 0) for testing...")
        
        # Using a standard 720p resolution that laptop webcams support.
        # The StereoCamera architecture will slice this down the middle, 
        # generating two 640x720 mock frames to stress-test the disk I/O queue!
        camera_instance = StereoCamera(
            camera_device_index=0, 
            hardware_width=1280, 
            hardware_height=720, 
            fps=30
        )
        
        scanner = AutonomousTransectScanner(
            target_storage_directory_path="./live_dive_data",
            stereo_camera_interface=camera_instance,
            connected_imu_sensor=DummyIMU(),
            connected_depth_sensor=DummyDepth()
        )
        
        # Start saving to disk!
        scanner.start_scanning_session()
        
        logging.info("RECORDING LIVE DATA! Let it run for 5 seconds...")
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < 5.0:
            loop_start = time.perf_counter()
            
            # The blazing fast acquisition cycle
            scanner.execute_single_acquisition_cycle()
            
            # Lock this loop to EXACTLY 30 FPS so we don't spam the queue
            while (time.perf_counter() - loop_start) < (1.0 / 30.0):
                pass
                
    finally:
        logging.info("Shutting down...")
        if 'scanner' in locals():
            scanner.stop_scanning_session()
        if camera_instance:
            camera_instance.release_stereo_camera_hardware()
        logging.info("Check the 'live_dive_data' folder for your webcam images and CSV!")