from pathlib import Path
from re import I
from typing import Union

import numpy as np
from ultralytics import YOLO

class DepthObjectDetector:
    def __init__(self, pose_model: str = "yolo26n.pt", depth_model: str = "yolo26n-depth.pt") -> None:
        self.depth_model = YOLO(depth_model)
        self.pose_model = YOLO(pose_model)

    # function to detect a human in the image
    def detect_objects(self, image_path):
        # run our pose model
        result = self.pose_model.predict(image_path, save=True, verbose=True)[0]
        # check to see if something was found
        if result.boxes is None or len(result.boxes) == 0:
            return None
        classes = result.boxes.cls.cpu().numpy()
        # extract bounding box of person objects
        person = result.boxes.xyxy.cpu().numpy()[classes == 0]
        # assume 1 person in frame, return first person detectoin xyxy
        if len(person) != 1:
            print("Not the right amount of people detected")
            return None
        return person[0]

    # return the depth map of our image
    def detect_depth(self, image_path):
        results = self.depth_model.predict(image_path, save=False, verbose=False)
        return np.squeeze(results[0].depth.data.detach().cpu().numpy())

    # return the depth of a person from an image
    def return_human_depth(self, image_path):
        depth_map = self.detect_depth(image_path)
        people = self.detect_objects(image_path)
        # check if a person was detected
        if people is None:
            return -1
        return self.get_center_depth(depth_map, people)

    # extract pixels from box
    def get_center_depth(self, depth_map, xyxy) -> float:
        depth_map = np.asarray(depth_map)
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a two-dimensional matrix")
        # unpack and find center of the bounding box
        x1,y1,x2,y2 = xyxy
        center_y = int((y1+y2) // 2)
        center_x = int((x1+x2) // 2)
        print(f'cords: {depth_map}')
        print(f'type: {type(depth_map)}')
        print(f'shape: {depth_map.shape}')
        return depth_map[center_y, center_x]

