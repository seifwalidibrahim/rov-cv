import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def compute_local_point_cloud(left_rectified_image, right_rectified_image, disparity_to_depth_mapping_matrix):
    minimum_disparity_value = 0
    number_of_disparities = 160
    block_matching_window_size = 5
    
    stereo_block_matcher = cv2.StereoSGBM_create(
        minDisparity=minimum_disparity_value,
        numDisparities=number_of_disparities,
        blockSize=block_matching_window_size,
        P1=8 * 3 * block_matching_window_size ** 2,
        P2=32 * 3 * block_matching_window_size ** 2,
        disp12MaxDiff=1,
        
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    computed_disparity_map = stereo_block_matcher.compute(left_rectified_image, right_rectified_image).astype(np.float32) / 16.0
    
    three_dimensional_point_cloud = cv2.reprojectImageTo3D(computed_disparity_map, disparity_to_depth_mapping_matrix)
    
    return three_dimensional_point_cloud, computed_disparity_map


def transform_local_cloud_to_global_space(local_point_cloud, imu_roll_degrees, imu_pitch_degrees, imu_yaw_degrees, depth_sensor_meters):
    rotation_transformation = Rotation.from_euler('xyz', [imu_roll_degrees, imu_pitch_degrees, imu_yaw_degrees], degrees=True)
    rotation_matrix = rotation_transformation.as_matrix()
    
    reshaped_point_cloud = local_point_cloud.reshape(-1, 3)
    
    rotated_point_cloud = np.dot(reshaped_point_cloud, rotation_matrix.T)
    
    translation_vector = np.array([0.0, 0.0, depth_sensor_meters])
    global_point_cloud = rotated_point_cloud + translation_vector
    
    return global_point_cloud