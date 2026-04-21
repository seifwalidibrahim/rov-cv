import cv2
import numpy as np
import glob

def perform_stereo_calibration(calibration_image_directory, checkerboard_rows, checkerboard_columns, square_size_in_meters):
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    object_point_grid = np.zeros((checkerboard_rows * checkerboard_columns, 3), np.float32)
    object_point_grid[:, :2] = np.mgrid[0:checkerboard_rows, 0:checkerboard_columns].T.reshape(-1, 2)
    object_point_grid *= square_size_in_meters

    collected_object_points = []
    collected_left_image_points = []
    collected_right_image_points = []

    left_image_filepaths = glob.glob(f"{calibration_image_directory}/left_*.png")
    right_image_filepaths = glob.glob(f"{calibration_image_directory}/right_*.png")
    
    left_image_filepaths.sort()
    right_image_filepaths.sort()

    image_dimensions = None

    for left_filepath, right_filepath in zip(left_image_filepaths, right_image_filepaths):
        left_image_matrix = cv2.imread(left_filepath)
        right_image_matrix = cv2.imread(right_filepath)
        
        left_grayscale_matrix = cv2.cvtColor(left_image_matrix, cv2.COLOR_BGR2GRAY)
        right_grayscale_matrix = cv2.cvtColor(right_image_matrix, cv2.COLOR_BGR2GRAY)
        
        if image_dimensions is None:
            image_dimensions = left_grayscale_matrix.shape[::-1]

        left_pattern_found, left_checkerboard_corners = cv2.findChessboardCorners(left_grayscale_matrix, (checkerboard_rows, checkerboard_columns), None)
        right_pattern_found, right_checkerboard_corners = cv2.findChessboardCorners(right_grayscale_matrix, (checkerboard_rows, checkerboard_columns), None)

        if left_pattern_found and right_pattern_found:
            collected_object_points.append(object_point_grid)
            
            refined_left_corners = cv2.cornerSubPix(left_grayscale_matrix, left_checkerboard_corners, (11, 11), (-1, -1), termination_criteria)
            collected_left_image_points.append(refined_left_corners)
            
            refined_right_corners = cv2.cornerSubPix(right_grayscale_matrix, right_checkerboard_corners, (11, 11), (-1, -1), termination_criteria)
            collected_right_image_points.append(refined_right_corners)

    left_reprojection_error, left_camera_matrix, left_distortion_coefficients, left_rotation_vectors, left_translation_vectors = cv2.calibrateCamera(collected_object_points, collected_left_image_points, image_dimensions, None, None)
    right_reprojection_error, right_camera_matrix, right_distortion_coefficients, right_rotation_vectors, right_translation_vectors = cv2.calibrateCamera(collected_object_points, collected_right_image_points, image_dimensions, None, None)

    stereo_calibration_flags = cv2.CALIB_FIX_INTRINSIC
    
    stereo_reprojection_error, final_left_camera_matrix, final_left_distortion_coefficients, final_right_camera_matrix, final_right_distortion_coefficients, rotation_matrix_between_cameras, translation_vector_between_cameras, essential_matrix, fundamental_matrix = cv2.stereoCalibrate(
        collected_object_points, 
        collected_left_image_points, 
        collected_right_image_points, 
        left_camera_matrix, 
        left_distortion_coefficients, 
        right_camera_matrix, 
        right_distortion_coefficients, 
        image_dimensions, 
        criteria=termination_criteria, 
        flags=stereo_calibration_flags
    )

    return final_left_camera_matrix, final_right_camera_matrix, rotation_matrix_between_cameras, translation_vector_between_cameras