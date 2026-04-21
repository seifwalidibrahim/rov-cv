import cv2
from ultralytics import YOLO

def test_live_object_counting():
    # This will automatically download the stock YOLOv8 Nano model
    stock_model = YOLO("yolov8n.pt") 
    
    # Using the same camera index from your successful test
    camera_feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    print("Press 'q' to quit.")
    
    while True:
        success, frame = camera_feed.read()
        if not success:
            break
            
        # Run YOLO inference
        results = stock_model(frame, stream=True)
        
        target_count = 0
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = stock_model.names[class_id]
                
                # Change "cell phone" to "cup" or "bottle" depending on what is on your desk
                if class_name == 'cell phone':
                    target_count += 1
                    
            # Draw the standard YOLO bounding boxes
            annotated_frame = r.plot()
            
        # Display the count on the screen
        cv2.putText(annotated_frame, f"Targets Detected: {target_count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        cv2.imshow("YOLO Pipeline Test", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera_feed.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_live_object_counting()