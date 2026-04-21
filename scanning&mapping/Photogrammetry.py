import cv2
import numpy as np
import csv
import open3d as o3d
from scipy.spatial.transform import Rotation

class PhotogrammetryMathematicalEngine:
    def __init__(self, disparity_to_depth_mapping_matrix):
        self.disparity_to_depth_mapping_matrix = disparity_to_depth_mapping_matrix
        self.minimum_disparity_value = 0
        self.number_of_disparities = 160
        self.block_matching_window_size = 5
        self.stereo_block_matcher = cv2.StereoSGBM_create(
            minDisparity=self.minimum_disparity_value,
            numDisparities=self.number_of_disparities,
            blockSize=self.block_matching_window_size,
            P1=8 * 3 * self.block_matching_window_size ** 2,
            P2=32 * 3 * self.block_matching_window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def transform_local_cloud_to_global_space(self, local_point_cloud, imu_roll_degrees, imu_pitch_degrees, imu_yaw_degrees, depth_in_meters):
        rotation_transformation = Rotation.from_euler('xyz', [imu_roll_degrees, imu_pitch_degrees, imu_yaw_degrees], degrees=True)
        rotation_matrix = rotation_transformation.as_matrix()
        reshaped_point_cloud = local_point_cloud.reshape(-1, 3)
        rotated_point_cloud = np.dot(reshaped_point_cloud, rotation_matrix.T)
        translation_vector = np.array([0.0, 0.0, depth_in_meters])
        return rotated_point_cloud + translation_vector

    def process_telemetry_into_fused_point_cloud(self, telemetry_database_path, output_point_cloud_filepath):
        aggregated_global_point_cloud = []
        
        with open(telemetry_database_path, mode='r') as telemetry_file_handler:
            telemetry_data_reader = csv.DictReader(telemetry_file_handler)
            
            for current_telemetry_row in telemetry_data_reader:
                left_sensor_image = cv2.imread(current_telemetry_row["left_image_filepath"], cv2.IMREAD_GRAYSCALE)
                right_sensor_image = cv2.imread(current_telemetry_row["right_image_filepath"], cv2.IMREAD_GRAYSCALE)
                
                computed_disparity_map = self.stereo_block_matcher.compute(left_sensor_image, right_sensor_image).astype(np.float32) / 16.0
                local_point_cloud = cv2.reprojectImageTo3D(computed_disparity_map, self.disparity_to_depth_mapping_matrix)
                
                imu_roll_degrees = float(current_telemetry_row["imu_roll_degrees"])
                imu_pitch_degrees = float(current_telemetry_row["imu_pitch_degrees"])
                imu_yaw_degrees = float(current_telemetry_row["imu_yaw_degrees"])
                depth_in_meters = float(current_telemetry_row["depth_in_meters"])
                
                global_point_cloud_segment = self.transform_local_cloud_to_global_space(
                    local_point_cloud, 
                    imu_roll_degrees, 
                    imu_pitch_degrees, 
                    imu_yaw_degrees, 
                    depth_in_meters
                )
                aggregated_global_point_cloud.append(global_point_cloud_segment)

        fused_global_point_cloud = np.vstack(aggregated_global_point_cloud)
        valid_points_mask = ~np.isnan(fused_global_point_cloud).any(axis=1) & np.isfinite(fused_global_point_cloud).all(axis=1)
        clean_point_cloud_data = fused_global_point_cloud[valid_points_mask]
        
        open3d_point_cloud_object = o3d.geometry.PointCloud()
        open3d_point_cloud_object.points = o3d.utility.Vector3dVector(clean_point_cloud_data)
        o3d.io.write_point_cloud(output_point_cloud_filepath, open3d_point_cloud_object)