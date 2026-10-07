from pathlib import Path
from re import I
from typing import Union
import cv2
import numpy as np
from ultralytics import YOLO

class DepthObjectDetector:
    def __init__(self, pose_model: str = "yolo26n.pt", depth_model: str = "yolo26n-depth.pt") -> None:
        self.depth_model = YOLO(depth_model)
        self.pose_model = YOLO(pose_model)

    # function to detect a human in the image
    def detect_objects(self, image: np.ndarray, streaming: bool = False, run_id=None, include_keypoints: bool = False):
        # build parameter dictionary - save images when gathering no save when streaming
        if not run_id:
            kwargs = {
                "save": False,
                "verbose": False,
            }
        else:
            kwargs = {
                "save": True,
                "verbose": True,
                "project": "depth-estimation",
                "name": str(run_id)
            }

        # run our pose model
        result = self.pose_model.predict(image, **kwargs)[0]
        # check to see if something was found
        if result.boxes is None or len(result.boxes) == 0:
            return None
        classes = result.boxes.cls.cpu().numpy()
        # extract bounding box of person objects
        person = result.boxes.xyxy.cpu().numpy()[classes == 0]
        if include_keypoints:
            if not streaming and len(person) != 1:
                print("Not the right amount of people detected")
                return None
            if result.keypoints is None:
                return person, None, None
            keypoints = result.keypoints.xy.cpu().numpy()[classes == 0]
            if result.keypoints.conf is None:
                confidences = None
            else:
                confidences = result.keypoints.conf.cpu().numpy()[classes == 0]
            return person, keypoints, confidences
        # if streaming return bounding boxes of all people detected
        if streaming:
            return person
        # assume 1 person in frame, return first person detectoin xyxy (data collection)
        if len(person) != 1:
            print("Not the right amount of people detected")
            return None
        # return bounding box of person
        return person[0]

    # return the depth map of our image
    def detect_depth(self, image: np.ndarray) -> list[any]:
        results = self.depth_model.predict(image, save=False, verbose=False)
        depth_data = results[0].depth.data.cpu().numpy()
        return np.squeeze(depth_data).astype(np.float32)

    # return the depth of a person from an image - this is the main function 
    def return_human_depth(self, image: np.ndarray, streaming=False, run_id=None, return_face_points=False, return_depth_map=False):
        # branch for streaming service:
        depth_map = self.detect_depth(image)
        detected = self.detect_objects(image, streaming, run_id, include_keypoints=True)
        # check if a person was detected
        if detected is None:
            if return_depth_map:
                return {}, {}, depth_map
            return None
        people, keypoints, confidences = detected
        if people is None or len(people) == 0:
            if return_depth_map:
                return {}, {}, depth_map
            return None
        # find each face center and sample its depth
        box_to_depth = {}
        box_to_face_center = {}
        for index, person in enumerate(people):
            face_center = None
            if keypoints is not None:
                person_confidences = None if confidences is None else confidences[index]
                face_center = self.get_face_center(keypoints[index], person_confidences, image.shape)
            if face_center is None:
                depth = self.get_center_depth(depth_map, person, image.shape)
            else:
                depth = self.get_depth_at_point(depth_map, face_center, image.shape)
                box_to_face_center[tuple(person)] = face_center
            box_to_depth[tuple(person)] = depth
        if return_depth_map:
            return box_to_depth, box_to_face_center, depth_map
        if return_face_points:
            return box_to_depth, box_to_face_center
        return box_to_depth

    # get the center of the face by averaging the keypoints of the face
    def get_face_center(self, keypoints, confidences, image_shape):
        image_height, image_width = image_shape[:2]
        face_points = []
        for index in (0, 1, 2, 3, 4):
            if index >= len(keypoints):
                continue
            x, y = keypoints[index]
            confidence = 1.0 if confidences is None else confidences[index]
            if not np.isfinite(confidence) or confidence < 0.2:
                continue
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                continue
            face_points.append((x, y))

        if not face_points:
            return None
        center_x, center_y = np.mean(face_points, axis=0)
        return float(center_x), float(center_y)

    # get depth at a specific point in the image using the depth map
    def get_depth_at_point(self, depth_map, point, image_shape):
        depth_map = np.asarray(depth_map)
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a two-dimensional matrix")
        image_height, image_width = image_shape[:2]
        depth_height, depth_width = depth_map.shape
        x, y = point
        depth_x = np.clip(int(x * depth_width / image_width), 0, depth_width - 1)
        depth_y = np.clip(int(y * depth_height / image_height), 0, depth_height - 1)
        return depth_map[depth_y, depth_x]

    # extract pixels from box
    def get_center_depth(self, depth_map, xyxy, image_shape=None) -> float:
        depth_map = np.asarray(depth_map)
        if depth_map.ndim != 2:
            raise ValueError("depth_map must be a two-dimensional matrix")
        # unpack and find center of the bounding box
        x1,y1,x2,y2 = xyxy
        center_y = int((y1+y2) // 2)
        center_x = int((x1+x2) // 2)
        if image_shape is not None:
            return self.get_depth_at_point(depth_map, (center_x, center_y), image_shape)
        center_x = np.clip(center_x, 0, depth_map.shape[1] - 1)
        center_y = np.clip(center_y, 0, depth_map.shape[0] - 1)
        return depth_map[center_y, center_x]

