import cv2
import threading
import time

class StereoCamera:
    def __init__(self, camera_device_index, target_combined_width_pixels, target_combined_height_pixels):
        self.stereo_video_capture_interface = cv2.VideoCapture(camera_device_index, cv2.CAP_V4L2)
        
        self.stereo_video_capture_interface.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stereo_video_capture_interface.set(cv2.CAP_PROP_FRAME_WIDTH, target_combined_width_pixels)
        self.stereo_video_capture_interface.set(cv2.CAP_PROP_FRAME_HEIGHT, target_combined_height_pixels)
        self.stereo_video_capture_interface.set(cv2.CAP_PROP_FPS, 30)
        self.stereo_video_capture_interface.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_was_successfully_retrieved, self.combined_stereo_image_frame = self.stereo_video_capture_interface.read()
        self.is_background_thread_active = True
        
        # The traffic light flag
        self.new_frame_available = True 
        
        self.background_frame_retrieval_thread = threading.Thread(target=self.continuously_update_camera_frames)
        self.background_frame_retrieval_thread.daemon = True
        self.background_frame_retrieval_thread.start()

    def continuously_update_camera_frames(self):
        while self.is_background_thread_active:
            retrieved_status, current_frame = self.stereo_video_capture_interface.read()
            if retrieved_status:
                self.frame_was_successfully_retrieved = retrieved_status
                self.combined_stereo_image_frame = current_frame
                # Raise the flag: A fresh frame from the hardware is ready
                self.new_frame_available = True 


    def retrieve_synchronized_stereo_frames(self):
        # The main thread waits here until the background thread raises the flag
        while not self.new_frame_available and self.is_background_thread_active:
            time.sleep(0.005)
            
        # Lower the flag so we don't read the same frame twice
        self.new_frame_available = False 
        
        if self.frame_was_successfully_retrieved and self.combined_stereo_image_frame is not None:
            image_height, image_width, color_channels = self.combined_stereo_image_frame.shape
            horizontal_midpoint_index = image_width // 2
            
            left_sensor_image = self.combined_stereo_image_frame[:, :horizontal_midpoint_index]
            right_sensor_image = self.combined_stereo_image_frame[:, horizontal_midpoint_index:]
            
            return left_sensor_image, right_sensor_image
            
        return None, None

    def release_stereo_camera_hardware(self):
        self.is_background_thread_active = False
        self.background_frame_retrieval_thread.join()
        self.stereo_video_capture_interface.release()