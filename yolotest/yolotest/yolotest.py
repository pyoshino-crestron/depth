from ultralytics import YOLO
import numpy as np
from PIL import Image
import sys
from ultralytics.data.utils import check_det_dataset
import matplotlib.pyplot as plt
import os

# load the model 
model = YOLO("yolo26n-depth.pt")
data = check_det_dataset("nyu-depth.yaml")

# function to ingest a photo and predict depth
def detect_objects_with_depth(image_path):
    # Run YOLO inference
    results = model(image_path)
    result = results[0] # Obtain detections for that image
    
    # Load the original image to access depth map
    image = Image.open(image_path)
    image_array = np.array(image)
    
    # Extract depth map (YOLO depth model returns depth in results)
    depth_map = result.depth
    
    # Process detections
    detections = []

    # check if anything detected
    if result.boxes is None:
        return None

    # Determine depth of every box
    for box in result.boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1], x2)
        y2 = min(depth_map.shape[0], y2)
            
        # Extract depth values within bounding box
        depth_region = depth_map[y1:y2, x1:x2]
            
        # Calculate average depth (ignore zero/invalid values if needed)
        avg_depth = float(np.mean(depth_region))

        # Get class name and confidence
        class_id = int(box.cls)
        class_name = model.names[class_id]
        confidence = float(box.conf)
            
        # Store detection info
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
            'avg_depth': avg_depth
        })
    
    return detections, result, depth_map

# Run script
if __name__ == "__main__":
    print(data)
    images_dir = data['val']
    for i in range(len(os.listdir(images_dir))):
        image_path = os.path.join(images_dir, os.listdir(images_dir)[i])
        out = detect_objects_with_depth(image_path)
        if out:
            detections, result, depth_map = out
            annotated_image = result.plot()

            # plt annoated image and print results
            plt.figure(figsize=(12,8))
            plt.imshow(annotated_image)
            plt.show()
            
            # print all detections found
            for detection in detections:
                print(f"Object: {detection['class']}")
                print(f"  Confidence: {detection['confidence']:.2f}")
                print(f"  Bounding Box: {detection['bbox']}")
                print(f"  Average Depth: {detection['avg_depth']:.4f}\n")

            break
        