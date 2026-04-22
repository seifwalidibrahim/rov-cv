import cv2
import numpy as np
import glob
import json
import os
import time
import sys
import logging

# Dynamically import the camera module from the streaming folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'streaming(pi)')))
from camera import StereoCamera

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class StereoCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size_mm=25.0, save_dir="calibration_frames"):
        # The inner corners of the checkerboard (Width, Height)
        self.checkerboard_size = checkerboard_size
        self.square_size_mm = square_size_mm
        self.save_dir = save_dir
        
        # Sub-pixel termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0) scaled by physical square size
        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size_mm

        os.makedirs(self.save_dir, exist_ok=True)

    def live_capture_assistant(self, camera_instance):
        """Guides the user to capture perfect synchronized calibration pairs."""
        logging.info(f"Starting Live Calibration Assistant. Target: {self.checkerboard_size} inner corners.")
        logging.info("Press 'c' to CAPTURE a valid pair. Press 'q' to QUIT and move to solving.")
        
        captured_pairs = 0
        
        while True:
            left_frame, right_frame = camera_instance.retrieve_synchronized_stereo_frames()
            if left_frame is None or right_frame is None:
                continue

            display_left = left_frame.copy()
            display_right = right_frame.copy()
            
            gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Fast check for corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_left, self.checkerboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_right, self.checkerboard_size, None)

            # UI Feedback
            if ret_l and ret_r:
                cv2.drawChessboardCorners(display_left, self.checkerboard_size, corners_l, ret_l)
                cv2.drawChessboardCorners(display_right, self.checkerboard_size, corners_r, ret_r)
                status_text = "LOCK ACQUIRED! Press 'c' to Capture."
                color = (0, 255, 0)
            else:
                status_text = "SEARCHING FOR CHECKERBOARD..."
                color = (0, 0, 255)

            combined_display = cv2.hconcat([display_left, display_right])
            combined_display = cv2.resize(combined_display, (1280, 480))
            
            cv2.putText(combined_display, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(combined_display, f"Pairs Captured: {captured_pairs}/30", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            
            cv2.imshow("Stereo Calibration Assistant", combined_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and ret_l and ret_r:
                timestamp = int(time.time() * 1000)
                l_path = os.path.join(self.save_dir, f"left_{timestamp}.png")
                r_path = os.path.join(self.save_dir, f"right_{timestamp}.png")
                
                # Save the CLEAN, un-drawn frames
                cv2.imwrite(l_path, left_frame)
                cv2.imwrite(r_path, right_frame)
                captured_pairs += 1
                logging.info(f"Captured Pair {captured_pairs}! Saved to {self.save_dir}")
                
                # Flash screen white to confirm capture
                cv2.imshow("Stereo Calibration Assistant", np.ones_like(combined_display)*255)
                cv2.waitKey(200)

        cv2.destroyAllWindows()
        return captured_pairs

    def solve_and_save_matrices(self, output_json_path="stereo_calibration.json"):
        """Computes sub-pixel intrinsics and extrinsics, exporting to a clean JSON file."""
        left_images = sorted(glob.glob(os.path.join(self.save_dir, 'left_*.png')))
        right_images = sorted(glob.glob(os.path.join(self.save_dir, 'right_*.png')))

        if len(left_images) == 0 or len(left_images) != len(right_images):
            logging.error("Mismatch or missing calibration frames. Capture frames first!")
            return False

        objpoints = [] # 3d point in real world space
        imgpoints_left = [] # 2d points in image plane
        imgpoints_right = []
        
        image_shape = None
        valid_pairs = 0

        logging.info("Analyzing frames for sub-pixel accuracy...")
        for l_path, r_path in zip(left_images, right_images):
            img_l = cv2.imread(l_path)
            img_r = cv2.imread(r_path)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            if image_shape is None:
                image_shape = gray_l.shape[::-1]

            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.checkerboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.checkerboard_size, None)

            if ret_l and ret_r:
                objpoints.append(self.objp)
                
                # Refine corner locations to sub-pixel accuracy
                refined_corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                imgpoints_left.append(refined_corners_l)
                
                refined_corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                imgpoints_right.append(refined_corners_r)
                valid_pairs += 1

        logging.info(f"Successfully refined corners in {valid_pairs} pairs. Calculating Matrices...")

        # 1. Calibrate Individual Cameras (Intrinsics)
        logging.info("Solving Left Intrinsics...")
        ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_left, image_shape, None, None)
        
        logging.info("Solving Right Intrinsics...")
        ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_right, image_shape, None, None)

        # 2. Stereo Calibration (Extrinsics - Rotation and Translation between lenses)
        logging.info("Solving Stereo Extrinsics (R, T, E, F)...")
        # Fixing intrinsics yields much better extrinsics since we already solved them accurately
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            mtx_l, dist_l,
            mtx_r, dist_r,
            image_shape,
            criteria=self.criteria,
            flags=flags
        )

        logging.info(f"Calibration Complete! RMS Re-projection Error: {ret_stereo:.4f} (Lower is better)")

        # 3. Export to JSON
        calibration_data = {
            "resolution": image_shape,
            "rms_error": ret_stereo,
            "left": {
                "camera_matrix": mtx_l.tolist(),
                "distortion_coefficients": dist_l.tolist()
            },
            "right": {
                "camera_matrix": mtx_r.tolist(),
                "distortion_coefficients": dist_r.tolist()
            },
            "stereo": {
                "rotation_matrix": R.tolist(),
                "translation_vector": T.tolist(),
                "essential_matrix": E.tolist(),
                "fundamental_matrix": F.tolist()
            }
        }

        with open(output_json_path, 'w') as json_file:
            json.dump(calibration_data, json_file, indent=4)
            
        logging.info(f"Mathematical Model securely saved to: {output_json_path}")
        return True


if __name__ == "__main__":
    # IMPORTANT: Measure your printed checkerboard square in millimeters!
    # Update these numbers to match your physical paper.
    calibrator = StereoCalibrator(checkerboard_size=(9, 6), square_size_mm=25.0)
    
    print("\n--- STEREO CALIBRATION PIPELINE ---")
    print("1. Live Capture Mode (Take new photos)")
    print("2. Solve Mode (Calculate math from existing photos)")
    choice = input("Select mode (1 or 2): ")
    
    if choice == '1':
        camera_instance = None
        try:
            # Using index 0 and 1280x720 to work seamlessly with the webcam test setup
            camera_instance = StereoCamera(camera_device_index=0, hardware_width=1280, hardware_height=720, fps=30)
            calibrator.live_capture_assistant(camera_instance)
            print("\nCapture complete. Run this script again and select Mode 2 to solve.")
        finally:
            if camera_instance:
                camera_instance.release_stereo_camera_hardware()
    elif choice == '2':
        calibrator.solve_and_save_matrices()
    else:
        print("Invalid choice.")