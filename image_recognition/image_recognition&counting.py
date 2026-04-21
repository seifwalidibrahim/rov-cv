import cv2
from ultralytics import YOLO

def count_and_visualize_invasive_crabs(captured_transect_image_path, trained_yolo_model_path):
    crab_detection_model = YOLO(trained_yolo_model_path)
    detection_results = crab_detection_model(captured_transect_image_path)
    
    invasive_crab_count = 0
    
    for individual_detection in detection_results[0].boxes:
        detected_class_id = int(individual_detection.cls[0])
        
        if crab_detection_model.names[detected_class_id] == 'invasive_european_green':
            invasive_crab_count += 1
            
    annotated_transect_image_array = detection_results[0].plot()
    
    return invasive_crab_count, annotated_transect_image_array