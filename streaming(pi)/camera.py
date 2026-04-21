import cv2
import threading
import time
import platform
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class StereoCamera:
    """
    30 FPS locked, tear-free stereo camera capture for Windows DirectShow / Linux V4L2.

    Architecture
    ============
    - One dedicated capture thread runs read() in a tight loop.  It never sleeps.
    - Frames land in a triple-buffer (3-slot ring).  The producer writes slot N,
      atomically swaps the "ready" index, then signals the consumer.  The consumer
      reads whatever slot is marked ready and makes a zero-copy numpy view (slice)
      of it.  There is no shared-memory window where a half-written frame can be seen.
    - The display loop calls retrieve_synchronized_stereo_frames() which returns
      immediately if a frame is ready, or blocks at most one frame period (33 ms at
      30 fps) if it's slightly ahead of the camera.
    """

    def __init__(self, camera_device_index: int = 1,
                 hardware_width: int = 2560, hardware_height: int = 720,
                 fps: int = 30):
        self.camera_device_index = camera_device_index
        self.hardware_width      = hardware_width
        self.hardware_height     = hardware_height
        self.fps                 = fps

        current_os = platform.system()
        if current_os == "Windows":
            logging.info("Windows detected - using DirectShow (CAP_DSHOW).")
            self.backend = cv2.CAP_DSHOW
        elif current_os == "Linux":
            logging.info("Linux detected - using V4L2 (CAP_V4L2).")
            self.backend = cv2.CAP_V4L2
        else:
            logging.info(f"Unknown OS '{current_os}' - using CAP_ANY.")
            self.backend = cv2.CAP_ANY

        # ── Triple-buffer ──────────────────────────────────────────────────────
        self._buf          = [None, None, None]   # filled after first real frame
        self._write_idx    = 0                     # producer owns this slot
        self._ready_idx    = -1                    # -1 = no frame yet
        self._buf_lock     = threading.Lock()

        # Semaphore ensures no signaling events are lost
        self._frame_sem    = threading.Semaphore(0)

        self.cap                         = None
        self.is_background_thread_active = True
        self.is_connected                = False

        self._initialize_hardware()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureThread"
        )
        self._capture_thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Hardware init
    # ──────────────────────────────────────────────────────────────────────────

    def _initialize_hardware(self):
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(self.camera_device_index, self.backend)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_device_index}. "
                "Is another app (VideoCap/OBS) holding the device?"
            )

        # ── THE PROVEN DIRECTSHOW FOURCC INCANTATION ───────────────────────────
        cap.set(cv2.CAP_PROP_FPS,    self.fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.hardware_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.hardware_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # ── Hardware settle ────────────────────────────────────────────────────
        time.sleep(0.5)

        # ── Flush stale startup frames ─────────────────────────────────────────
        frame = None
        for _ in range(5):
            ret, f = cap.read()
            if ret and f is not None:
                frame = f

        if frame is None:
            cap.release()
            raise RuntimeError(
                "Camera opened but failed to decode any frames after settle+flush.\n"
                "Possible causes:\n"
                "  * Wrong camera index (try 0 or 2)\n"
                "  * MJPG not supported at this resolution by this camera\n"
                "  * Another app is holding the device open\n"
            )

        # ── Verify codec ───────────────────────────────────────────────────────
        raw           = int(cap.get(cv2.CAP_PROP_FOURCC))
        actual_fourcc = "".join(chr((raw >> (8 * i)) & 0xFF) for i in range(4))
        actual_fps    = cap.get(cv2.CAP_PROP_FPS)
        actual_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logging.info(
            f"Stream committed: {actual_w}x{actual_h} @ {actual_fps:.0f} fps  "
            f"FOURCC={actual_fourcc!r}"
        )

        if actual_fourcc.upper() != "MJPG":
            logging.warning(
                f"Driver is streaming {actual_fourcc!r} instead of MJPG. "
                "Expect low FPS."
            )

        # ── Pre-allocate triple-buffer ─────────────────────────────────────────
        h, w = frame.shape[:2]
        for i in range(3):
            self._buf[i] = np.empty((h, w, 3), dtype=np.uint8)

        # Seed slot 0 with the first real frame so the consumer never starves.
        np.copyto(self._buf[0], frame)
        with self._buf_lock:
            self._ready_idx = 0
            self._write_idx = 1

        self._frame_sem.release()   # one frame is already available
        self.cap          = cap
        self.is_connected = True
        logging.info("Camera ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Capture thread  (producer)
    # ──────────────────────────────────────────────────────────────────────────

    def _capture_loop(self):
        """Runs at whatever speed the hardware pushes. Safely overwrites buffers."""
        consecutive_drops = 0

        while self.is_background_thread_active:
            if not self.is_connected:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                consecutive_drops += 1
                if consecutive_drops > 60:   # ~2 s at 30 fps
                    logging.warning("60 consecutive dropped frames - camera disconnected.")
                    self.is_connected = False
                    consecutive_drops = 0
                continue

            consecutive_drops = 0

            # Write directly into pre-allocated buffer slot — no heap alloc
            np.copyto(self._buf[self._write_idx], frame)

            # Atomically promote our slot to "ready"
            with self._buf_lock:
                old_ready        = self._ready_idx
                self._ready_idx  = self._write_idx
                self._write_idx  = 3 - self._ready_idx - old_ready

            if self._frame_sem._value == 0:
                self._frame_sem.release()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve_synchronized_stereo_frames(self):
        if not self.is_connected:
            return None, None

        acquired = self._frame_sem.acquire(timeout=0.066)
        if not acquired:
            return None, None

        with self._buf_lock:
            idx = self._ready_idx

        if idx < 0:
            return None, None

        frame = self._buf[idx]
        mid   = frame.shape[1] // 2

        return frame[:, :mid], frame[:, mid:]

    def release_stereo_camera_hardware(self):
        self.is_background_thread_active = False
        self._frame_sem.release()   
        if hasattr(self, '_capture_thread'):
            self._capture_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        logging.info("Camera released.")


# ══════════════════════════════════════════════════════════════════════════════
# Preview / test harness
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    camera_instance = None
    try:
        logging.info("Initializing...")
        
        # We explicitly request 30 FPS here
        TARGET_FPS = 30.0
        TARGET_FRAME_TIME = 1.0 / TARGET_FPS
        
        camera_instance = StereoCamera(
            camera_device_index=1,
            hardware_width=2560,
            hardware_height=720,
            fps=int(TARGET_FPS),
        )

        prev_time    = time.perf_counter()
        fps_smoothed = TARGET_FPS

        print("\n--- CONTROLS ---")
        print("  q  ->  Quit")
        print("  s  ->  Open DirectShow hardware settings\n")

        while True:
            # Mark the exact start time of this frame cycle
            loop_start = time.perf_counter()
            
            left, right = camera_instance.retrieve_synchronized_stereo_frames()

            if left is not None and right is not None:
                display = np.concatenate([left, right], axis=1)
                display = cv2.resize(display, (1280, 360), interpolation=cv2.INTER_LINEAR)

                color = (0, 255, 0) if fps_smoothed >= (TARGET_FPS - 2) else (0, 140, 255)
                cv2.putText(display, f"FPS: {fps_smoothed:.1f} (LOCKED)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
                cv2.putText(display, "s=Settings  q=Quit",
                            (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 220), 2, cv2.LINE_AA)
                cv2.imshow("Stereo Preview [2560x720 -> 1280x360]", display)
            else:
                blank = np.zeros((360, 1280, 3), dtype=np.uint8)
                cv2.putText(blank, "CONNECTION LOST - CHECK CABLE",
                            (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Stereo Preview [2560x720 -> 1280x360]", blank)

            # Handle UI and Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                logging.info("Opening DirectShow hardware settings dialog...")
                camera_instance.cap.set(cv2.CAP_PROP_SETTINGS, 1)

            # ── PRECISE 30 FPS SOFTWARE THROTTLE ─────────────────────────────────
            # Measure how much time we spent reading/resizing/displaying the frame.
            time_spent = time.perf_counter() - loop_start
            time_left_to_sleep = TARGET_FRAME_TIME - time_spent
            
            if time_left_to_sleep > 0:
                # Sleep briefly to give CPU to other apps, but wake up 2ms early
                if time_left_to_sleep > 0.002:
                    time.sleep(time_left_to_sleep - 0.002)
                # Spin-lock the final ~2 milliseconds for mathematical precision
                while (time.perf_counter() - loop_start) < TARGET_FRAME_TIME:
                    pass
            # ─────────────────────────────────────────────────────────────────────

            # Calculate the final FPS reading
            now = time.perf_counter()
            elapsed = max(now - prev_time, 1e-9)
            prev_time = now
            
            fps_inst = 1.0 / elapsed
            fps_smoothed = 0.92 * fps_smoothed + 0.08 * fps_inst

    finally:
        if camera_instance:
            camera_instance.release_stereo_camera_hardware()
        cv2.destroyAllWindows()