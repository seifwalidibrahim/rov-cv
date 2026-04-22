import cv2
import time
import csv
import os
import threading
import queue
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class AutonomousTransectScanner:
    def __init__(self, target_storage_directory_path, stereo_camera_interface, connected_imu_sensor, connected_depth_sensor):
        self.target_storage_directory_path = target_storage_directory_path
        self.stereo_camera_interface = stereo_camera_interface
        self.connected_imu_sensor = connected_imu_sensor
        self.connected_depth_sensor = connected_depth_sensor
        
        self.telemetry_csv_file_path = os.path.join(self.target_storage_directory_path, "transect_telemetry_database.csv")
        
        # State Flags
        self.is_actively_scanning = False
        self.captured_frame_sequence_number = 0
        
        # High-Performance I/O Queue
        # Buffers up to 300 frames (10 seconds of 30 FPS video) in RAM if disk gets slow
        self.io_queue = queue.Queue(maxsize=300) 
        self.io_thread = None
        self.csv_file_handler = None
        self.telemetry_data_writer = None

    def start_scanning_session(self):
        """Initializes directories, opens persistent file handlers, and starts the background I/O thread."""
        if self.is_actively_scanning:
            logging.warning("Scanning session is already active.")
            return

        logging.info(f"Starting Transect Scan. Saving to: {self.target_storage_directory_path}")
        os.makedirs(self.target_storage_directory_path, exist_ok=True)
        
        # Open CSV once and keep it open
        self.csv_file_handler = open(self.telemetry_csv_file_path, mode='w', newline='')
        self.telemetry_data_writer = csv.writer(self.csv_file_handler)
        self.telemetry_data_writer.writerow([
            "epoch_timestamp", 
            "left_image_filepath", 
            "right_image_filepath", 
            "depth_in_meters", 
            "imu_roll_degrees", 
            "imu_pitch_degrees", 
            "imu_yaw_degrees"
        ])
        
        self.is_actively_scanning = True
        
        # Spin up the dedicated background disk writer
        self.io_thread = threading.Thread(target=self._async_disk_writer_loop, daemon=True, name="DiskWriterThread")
        self.io_thread.start()

    def execute_single_acquisition_cycle(self):
        """
        The blazing fast main loop call. Grabs data, throws it in the queue, and returns instantly.
        Expected execution time: < 1ms.
        """
        if not self.is_actively_scanning:
            return

        # Interface directly with our new triple-buffered camera pipeline
        left_sensor_image, right_sensor_image = self.stereo_camera_interface.retrieve_synchronized_stereo_frames()
        
        if left_sensor_image is not None and right_sensor_image is not None:
            # Capture exact timestamp of hardware retrieval
            current_epoch_timestamp = time.time()
            
            # Read telemetry
            current_depth_reading_meters = self.connected_depth_sensor.read_depth()
            current_imu_orientation = self.connected_imu_sensor.read_orientation()
            
            # Generate file paths
            left_image_destination_path = os.path.join(self.target_storage_directory_path, f"left_frame_{self.captured_frame_sequence_number:05d}.png")
            right_image_destination_path = os.path.join(self.target_storage_directory_path, f"right_frame_{self.captured_frame_sequence_number:05d}.png")
            
            # Package the data
            data_payload = {
                "sequence": self.captured_frame_sequence_number,
                "timestamp": current_epoch_timestamp,
                "left_img": left_sensor_image,
                "right_img": right_sensor_image,
                "left_path": left_image_destination_path,
                "right_path": right_image_destination_path,
                "depth": current_depth_reading_meters,
                "roll": current_imu_orientation.roll,
                "pitch": current_imu_orientation.pitch,
                "yaw": current_imu_orientation.yaw
            }
            
            # Throw it to the background thread instantly
            try:
                self.io_queue.put_nowait(data_payload)
                self.captured_frame_sequence_number += 1
            except queue.Full:
                logging.error("CRITICAL: Disk I/O is too slow! RAM Queue is full. Dropping frame to preserve live pipeline.")

    def _async_disk_writer_loop(self):
        """Background thread dedicated entirely to slow disk operations."""
        while self.is_actively_scanning or not self.io_queue.empty():
            try:
                # Wait up to 0.5s for data. Allows loop to break cleanly on shutdown.
                payload = self.io_queue.get(timeout=0.5)
                
                # 1. Write PNGs (Uses IMWRITE_PNG_COMPRESSION=1 to speed up save times at cost of slight file size increase)
                cv2.imwrite(payload["left_path"], payload["left_img"], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(payload["right_path"], payload["right_img"], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                
                # 2. Write CSV row
                telemetry_database_row = [
                    payload["timestamp"],
                    payload["left_path"],
                    payload["right_path"],
                    payload["depth"],
                    payload["roll"],
                    payload["pitch"],
                    payload["yaw"]
                ]
                self.telemetry_data_writer.writerow(telemetry_database_row)
                
                # Tell the queue this task is done
                self.io_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Disk Write Error on sequence {payload.get('sequence', 'UNKNOWN')}: {e}")

    def stop_scanning_session(self):
        """Gracefully drains the RAM queue to the SD card and closes file handlers."""
        if not self.is_actively_scanning:
            return

        logging.info(f"Stopping scan. Waiting for {self.io_queue.qsize()} remaining frames to write to disk...")
        self.is_actively_scanning = False
        
        # Block until the background thread finishes writing all pending frames
        if self.io_thread is not None:
            self.io_thread.join()
            
        if self.csv_file_handler is not None:
            self.csv_file_handler.close()
            
        logging.info("Scan successfully stopped. All data saved.")


# ==========================================
# Mock Testing Block
# ==========================================
if __name__ == "__main__":
    import numpy as np

    # 1. Dummy Sensor Classes for testing
    class DummyIMU:
        class Orientation:
            def __init__(self):
                self.roll, self.pitch, self.yaw = 0.0, 5.2, 180.1
        def read_orientation(self): return self.Orientation()

    class DummyDepth:
        def read_depth(self): return 12.5

    class DummyStereoCam:
        def retrieve_synchronized_stereo_frames(self):
            # Generate fake 640x480 frames instantly
            return np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)

    # 2. Initialize the Scanner
    test_scanner = AutonomousTransectScanner(
        target_storage_directory_path="./test_scan_data",
        stereo_camera_interface=DummyStereoCam(),
        connected_imu_sensor=DummyIMU(),
        connected_depth_sensor=DummyDepth()
    )

    try:
        test_scanner.start_scanning_session()
        
        logging.info("Simulating a 30 FPS ROV dive for 3 seconds...")
        start_time = time.perf_counter()
        frames_simulated = 0
        
        # Simulate an exact 30 FPS pipeline pushing data to the scanner
        while time.perf_counter() - start_time < 3.0:
            loop_start = time.perf_counter()
            
            test_scanner.execute_single_acquisition_cycle()
            frames_simulated += 1
            
            # Spinlock throttle to exact 33.3ms (30 FPS)
            while (time.perf_counter() - loop_start) < (1.0 / 30.0):
                pass
                
        logging.info(f"Pushed {frames_simulated} frames to the scanner queue.")
        
    finally:
        # 3. Graceful shutdown (watch it drain the queue!)
        test_scanner.stop_scanning_session()