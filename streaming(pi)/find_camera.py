import cv2

def scan_for_cameras():
    active_indices = []
    for index in range(5):
        capture_test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture_test.isOpened():
            active_indices.append(index)
            capture_test.release()
    return active_indices

if __name__ == "__main__":
    valid_cameras = scan_for_cameras()
    print(f"Detected camera indices: {valid_cameras}")