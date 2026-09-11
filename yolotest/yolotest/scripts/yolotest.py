from ultralytics import YOLO
import numpy as np
from PIL import Image
import sys
from ultralytics.data.utils import check_det_dataset
import matplotlib.pyplot as plt
import os
import cv2
import base64
from pathlib import Path

from .utils import extract_from_box

# build a depth detector that supports depth detecton for humans
class DepthObjectDetector:
    def __init__(self, pose_model="yolo26n.pt", depth_model="yolo26n-depth.pt"):
        # load the model 
        self.depth_model = YOLO(depth_model)
        self.pose_model = YOLO(pose_model)


    # function to classify objects in Image
    def detect_objects(self, image_path):
        # Run YOLO inference
        results = self.pose_model.predict(image_path, save=False, verbose=False)
        main = results[0]
        xyxy = main.boxes.xyxy.cpu().numpy() 
        return xyxy

    # function to ingest a photo and predict depth
    def detect_depth(self, image_path):
        # Run YOLO inference
        results = self.depth_model.predict(image_path, save=False, verbose=False)
        result = results[0]
        # Extract the depth map
        depth = result.depth.data.detach().numpy()
        return depth

    # function that returns the depth of the center most pixel from a detected object
    def get_center_depth(self, depth_map, xyxy):
        return extract_from_box(depth_map, xyxy)

# Run script
if __name__ == "__main__":
    data = check_det_dataset("nyu-depth.yaml")
    detector = DepthObjectDetector()
    images_dir = data['val']

    for i in range(5):
        image_path = os.path.join(images_dir, os.listdir(images_dir)[i])
        depth_map = detector.detect_depth(image_path)
        detected_objects = detector.detect_objects(image_path)

        if len(detected_objects) == 0:
            print(f"No objects detected in {image_path}")
            continue

        for xyxy in detected_objects:
            center_depth = detector.get_center_depth(depth_map, xyxy)
            print(f"Image: {image_path}, Center depth: {center_depth}")



