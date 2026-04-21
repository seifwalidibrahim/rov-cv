import cv2

class VideoStreamRecorder:
    def __init__(self, output_video_filename, target_frames_per_second, target_frame_width, target_frame_height):
        self.output_video_filename = output_video_filename
        self.target_frames_per_second = target_frames_per_second
        self.target_frame_width = target_frame_width
        self.target_frame_height = target_frame_height
        self.webcam_capture_device = None
        self.video_file_writer = None
        self.is_recording_active = False

    def initialize_video_components(self):
        self.webcam_capture_device = cv2.VideoCapture(0)
        video_codec_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_dimensions_tuple = (self.target_frame_width, self.target_frame_height)
        self.video_file_writer = cv2.VideoWriter(self.output_video_filename, video_codec_fourcc, self.target_frames_per_second, video_dimensions_tuple)

    def start_recording_loop(self):
        self.initialize_video_components()
        self.is_recording_active = True

        while self.is_recording_active:
            frame_was_read_successfully, current_video_frame = self.webcam_capture_device.read()

            if not frame_was_read_successfully:
                self.is_recording_active = False
                break

            self.video_file_writer.write(current_video_frame)
            cv2.imshow('Active Video Recording Window', current_video_frame)

            pressed_keyboard_key = cv2.waitKey(1) & 0xFF
            if pressed_keyboard_key == ord('q'):
                self.stop_recording()

        self.release_video_resources()

    def stop_recording(self):
        self.is_recording_active = False

    def release_video_resources(self):
        if self.webcam_capture_device is not None:
            self.webcam_capture_device.release()
        if self.video_file_writer is not None:
            self.video_file_writer.release()
        cv2.destroyAllWindows()