import cv2

print("=== Mona Linux Hardware Diagnostic ===")

# 1. Force the Linux Native Backend
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
print(f"Camera Opened: {cap.isOpened()}")
print(f"Active Backend: {cap.getBackendName()}")

# 2. Force MJPG codec FIRST (Critical for Linux)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# 3. Set the exact matched resolution we found in v4l2-ctl
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# 4. Attempt to grab a single frame
ret, frame = cap.read()

if ret:
    print(f"SUCCESS! Hardware is streaming. Frame shape: {frame.shape}")
    # Show it for 3 seconds just to prove it works
    cv2.imshow("Direct Hardware Feed", frame)
    cv2.waitKey(3000)
else:
    print("FAILED: OpenCV connected to the node, but the Linux V4L2 buffer is empty.")

cap.release()
cv2.destroyAllWindows()