import cv2
import time
import csv
import os

class AutonomousTransectScanner:
    def __init__(self, target_storage_directory_path, stereo_camera_interface, connected_imu_sensor, connected_depth_sensor):
        self.target_storage_directory_path = target_storage_directory_path
        self.stereo_camera_interface = stereo_camera_interface
        self.connected_imu_sensor = connected_imu_sensor
        self.connected_depth_sensor = connected_depth_sensor
        self.telemetry_csv_file_path = os.path.join(self.target_storage_directory_path, "transect_telemetry_database.csv")
        self.is_actively_scanning = False
        self.captured_frame_sequence_number = 0

    def start_scanning_session(self):
        os.makedirs(self.target_storage_directory_path, exist_ok=True)
        telemetry_file_handler = open(self.telemetry_csv_file_path, mode='w', newline='')
        telemetry_data_writer = csv.writer(telemetry_file_handler)
        telemetry_data_writer.writerow([
            "epoch_timestamp", 
            "left_image_filepath", 
            "right_image_filepath", 
            "depth_in_meters", 
            "imu_roll_degrees", 
            "imu_pitch_degrees", 
            "imu_yaw_degrees"
        ])
        telemetry_file_handler.close()
        self.is_actively_scanning = True

    def stop_scanning_session(self):
        self.is_actively_scanning = False

    def extract_individual_stereo_frames(self, combined_stereo_image_frame):
        image_height, image_width, color_channels = combined_stereo_image_frame.shape
        horizontal_midpoint_index = image_width // 2
        left_sensor_image = combined_stereo_image_frame[:, :horizontal_midpoint_index]
        right_sensor_image = combined_stereo_image_frame[:, horizontal_midpoint_index:]
        return left_sensor_image, right_sensor_image

    def execute_single_acquisition_cycle(self):
        if not self.is_actively_scanning:
            return

        frame_was_successfully_retrieved, combined_stereo_image_frame = self.stereo_camera_interface.read()
        
        if frame_was_successfully_retrieved:
            current_epoch_timestamp = time.time()
            current_depth_reading_meters = self.connected_depth_sensor.read_depth()
            current_imu_orientation = self.connected_imu_sensor.read_orientation()
            
            left_sensor_image, right_sensor_image = self.extract_individual_stereo_frames(combined_stereo_image_frame)
            
            left_image_destination_path = os.path.join(self.target_storage_directory_path, f"left_frame_{self.captured_frame_sequence_number:05d}.png")
            right_image_destination_path = os.path.join(self.target_storage_directory_path, f"right_frame_{self.captured_frame_sequence_number:05d}.png")
            
            cv2.imwrite(left_image_destination_path, left_sensor_image)
            cv2.imwrite(right_image_destination_path, right_sensor_image)
            
            telemetry_file_handler = open(self.telemetry_csv_file_path, mode='a', newline='')
            telemetry_data_writer = csv.writer(telemetry_file_handler)
            telemetry_database_row = [
                current_epoch_timestamp,
                left_image_destination_path,
                right_image_destination_path,
                current_depth_reading_meters,
                current_imu_orientation.roll,
                current_imu_orientation.pitch,
                current_imu_orientation.yaw
            ]
            telemetry_data_writer.writerow(telemetry_database_row)
            telemetry_file_handler.close()
            
            self.captured_frame_sequence_number += 1