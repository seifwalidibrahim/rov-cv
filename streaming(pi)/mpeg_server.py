import cv2
import threading
import logging
import time
from http import server
import socketserver
import numpy as np

# Import the perfected camera module
from camera import StereoCamera

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Shared Memory & Synchronization ───────────────────────────────────────
# Independent byte buffers for each eye
latest_jpeg_left = None
latest_jpeg_right = None

# A single condition variable to synchronize ALL active network streams instantly
frame_condition = threading.Condition()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard HTTP logging to keep the Pi's terminal clean
        pass

    def do_GET(self):
        global latest_jpeg_left, latest_jpeg_right
        
        # Route the request to the correct memory buffer
        if self.path == '/stream_left.mjpg':
            stream_target = 'left'
        elif self.path == '/stream_right.mjpg':
            stream_target = 'right'
        else:
            self.send_error(404)
            self.end_headers()
            return
            
        # Standard MJPEG HTTP Headers
        self.send_response(200)
        self.send_header('Age', 0)
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
        self.end_headers()
        
        try:
            while True:
                # Wait for the hardware to tick
                with frame_condition:
                    frame_condition.wait()
                    # Grab the correct frame based on the requested URL
                    frame_data = latest_jpeg_left if stream_target == 'left' else latest_jpeg_right
                    
                if frame_data is None:
                    continue
                    
                # Blast the JPEG bytes over the Ethernet socket
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', len(frame_data))
                self.end_headers()
                self.wfile.write(frame_data)
                self.wfile.write(b'\r\n')
                
        except Exception:
            logging.info(f"Client disconnected from {stream_target} stream: {self.client_address}")
            return


class ThreadedHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Handles requests in a separate thread to prevent the network from blocking the camera."""
    daemon_threads = True


def start_network_server(port=8083):
    server_address = ('0.0.0.0', port)
    httpd = ThreadedHTTPServer(server_address, StreamingHandler)
    logging.info(f"Left Stream:  http://0.0.0.0:{port}/stream_left.mjpg")
    logging.info(f"Right Stream: http://0.0.0.0:{port}/stream_right.mjpg")
    
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return httpd


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution Loop
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    camera_instance = None
    try:
        start_network_server(port=8083)
        
        logging.info("Initializing Stereo Hardware...")
        TARGET_FPS = 30.0
        TARGET_FRAME_TIME = 1.0 / TARGET_FPS
        
        camera_instance = StereoCamera(
            camera_device_index=1,
            hardware_width=2560,
            hardware_height=720,
            fps=int(TARGET_FPS),
        )

        prev_time = time.perf_counter()
        fps_smoothed = TARGET_FPS

        logging.info("Pipeline running. Waiting for GUI connections...")

        while True:
            loop_start = time.perf_counter()
            
            # 1. Pull the flawless frames from our camera architecture
            left, right = camera_instance.retrieve_synchronized_stereo_frames()

            if left is not None and right is not None:
                # 2. Resize each eye individually
                # Scaling down slightly from native 1280x720 to 640x360 keeps the stream high-def 
                # but ensures dual-encoding doesn't throttle the Pi's CPU.
                left_disp = cv2.resize(left, (640, 360), interpolation=cv2.INTER_LINEAR)
                right_disp = cv2.resize(right, (640, 360), interpolation=cv2.INTER_LINEAR)

                # Add independent FPS overlays
                cv2.putText(left_disp, f"L-FPS: {fps_smoothed:.1f}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(right_disp, f"R-FPS: {fps_smoothed:.1f}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                # 3. Hardware-Accelerated JPEG Encoding
                ret_l, jpeg_encoded_left = cv2.imencode('.jpg', left_disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                ret_r, jpeg_encoded_right = cv2.imencode('.jpg', right_disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret_l and ret_r:
                    # 4. Atomically update the shared memory and instantly wake up all network sockets
                    with frame_condition:
                        latest_jpeg_left = jpeg_encoded_left.tobytes()
                        latest_jpeg_right = jpeg_encoded_right.tobytes()
                        frame_condition.notify_all()
            else:
                # If camera disconnects, stream warning images to both endpoints
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "CONNECTION LOST",
                            (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                ret, jpeg_encoded = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                
                if ret:
                    with frame_condition:
                        latest_jpeg_left = jpeg_encoded.tobytes()
                        latest_jpeg_right = jpeg_encoded.tobytes()
                        frame_condition.notify_all()

            # 5. Precise Software Throttle (maintains exact 30 FPS)
            time_spent = time.perf_counter() - loop_start
            time_left_to_sleep = TARGET_FRAME_TIME - time_spent
            
            if time_left_to_sleep > 0:
                if time_left_to_sleep > 0.002:
                    time.sleep(time_left_to_sleep - 0.002)
                while (time.perf_counter() - loop_start) < TARGET_FRAME_TIME:
                    pass

            now = time.perf_counter()
            elapsed = max(now - prev_time, 1e-9)
            prev_time = now
            
            fps_inst = 1.0 / elapsed
            fps_smoothed = 0.92 * fps_smoothed + 0.08 * fps_inst

    except KeyboardInterrupt:
        logging.info("Shutting down network and hardware...")
    finally:
        if camera_instance:
            camera_instance.release_stereo_camera_hardware()