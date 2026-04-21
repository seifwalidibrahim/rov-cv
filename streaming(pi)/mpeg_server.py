import cv2
import threading
import logging
import time
from http import server
import socketserver
import numpy as np

# Import the perfected camera module we just built
from camera import StereoCamera

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── Shared Memory & Synchronization ───────────────────────────────────────
# This holds the compressed JPEG bytes.
latest_jpeg_bytes = None
# This condition notifies the network threads the exact microsecond a frame is ready.
jpeg_condition = threading.Condition()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard HTTP logging so it doesn't spam the Pi's terminal
        pass

    def do_GET(self):
        global latest_jpeg_bytes
        
        if self.path == '/stream.mjpg':
            # 1. Standard MJPEG HTTP Headers
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            # Boundary defines where one frame ends and the next begins
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            
            try:
                while True:
                    # 2. Wait for the hardware to tick
                    with jpeg_condition:
                        jpeg_condition.wait()
                        frame_data = latest_jpeg_bytes
                        
                    if frame_data is None:
                        continue
                        
                    # 3. Blast the JPEG bytes over the Ethernet socket
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_data))
                    self.end_headers()
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    
            except Exception:
                # Client disconnected (GUI closed, page refreshed, or Ethernet cable pulled)
                logging.info(f"Client disconnected from stream: {self.client_address}")
                return
        else:
            self.send_error(404)
            self.end_headers()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Handles requests in a separate thread to prevent the network from blocking the camera."""
    daemon_threads = True


def start_network_server(port=8083):
    # Binding to '0.0.0.0' exposes the server to the Ethernet interface 
    # so the Windows laptop can connect via 192.168.137.65
    server_address = ('0.0.0.0', port)
    httpd = ThreadedHTTPServer(server_address, StreamingHandler)
    logging.info(f"MJPEG Stream Server actively broadcasting on http://0.0.0.0:{port}/stream.mjpg")
    
    # Run the server in a background thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    return httpd


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution Loop
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    camera_instance = None
    try:
        # Start the network server immediately on port 8083
        start_network_server(port=8083)
        
        logging.info("Initializing Stereo Hardware...")
        TARGET_FPS = 30.0
        TARGET_FRAME_TIME = 1.0 / TARGET_FPS
        
        # NOTE: On the Pi, camera_device_index might be 0 instead of 1.
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
                # 2. Stitch and resize for the GUI
                # Resizing to 1280x360 keeps the stream high-def but prevents 
                # saturating the Pi's CPU with massive JPEG encoding loads.
                display = np.concatenate([left, right], axis=1)
                display = cv2.resize(display, (1280, 360), interpolation=cv2.INTER_LINEAR)

                # Add the FPS overlay directly onto the stream so the GUI team can verify performance
                cv2.putText(display, f"Server FPS: {fps_smoothed:.1f} (LOCKED)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)

                # 3. Hardware-Accelerated JPEG Encoding
                # Quality 80 is the perfect sweet spot for MJPEG streams (low latency, high clarity)
                ret, jpeg_encoded = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                if ret:
                    # 4. Atomically update the shared memory and instantly wake up the network socket
                    with jpeg_condition:
                        latest_jpeg_bytes = jpeg_encoded.tobytes()
                        jpeg_condition.notify_all()
            else:
                # If camera disconnects, stream a warning image to the GUI
                blank = np.zeros((360, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "ROV CAMERA CONNECTION LOST",
                            (350, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                ret, jpeg_encoded = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if ret:
                    with jpeg_condition:
                        latest_jpeg_bytes = jpeg_encoded.tobytes()
                        jpeg_condition.notify_all()

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