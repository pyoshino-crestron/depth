import string
from ..camera.capture import CameraCapture
from ..detection.detector import DepthObjectDetector
from .snapshot import Snapshot
from ..detection import DepthObjectDetector
from ..camera.visca import get_current_fov
from queue import Queue
import cv2
import numpy as np
from threading import Thread
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
import socket

# function that continuosly captures snapshots and writes to a global queue
def capture_loop(camera: CameraCapture, snapshot_queue: Queue) -> np.ndarray:
    # continuosly loop taking snapshots
    while True:
        snapshot = camera.capture()
        # check if queue is full
        if snapshot_queue.full():
            snapshot_queue.get_nowait()
        snapshot_queue.put(snapshot)

# function that takes an image off of the queue and performs inference
def inference_loop(detector: DepthObjectDetector, snapshot_queue: Queue, lin_model, gb_model, fov):
    snapshot = snapshot_queue.get()
    if snapshot is None or snapshot.size == 0:
        return None
    # modify image to accept image object
    detection_result = detector.return_human_depth(
        snapshot,
        True,
        return_face_points=True,
        return_depth_map=True
    )
    if snapshot.ndim == 2:
        snapshot = cv2.cvtColor(snapshot, cv2.COLOR_GRAY2BGR)
    if not detection_result:
        return snapshot
    box_to_depth, face_centers, depth_map = detection_result

    height, width = snapshot.shape[:2]
    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    if depth_map.size:
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        snapshot = cv2.addWeighted(snapshot, 0.55, heatmap, 0.45, 0)

    # return the heatmapped frame even when no people are detected
    if not box_to_depth:
        return snapshot

    # update image for all people and annotated depths
    for index, (box, depth) in enumerate(box_to_depth.items()):
        # obtain corrected depth and create input features
        input_features = pd.DataFrame({
            "prediction": [depth],
            "fov": [fov]
        })
        hue = int(index * 180 / len(box_to_depth))
        hsv_color = np.uint8([[[hue, 255, 255]]])
        text_color = tuple(int(channel) for channel in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        # predict residual error
        lin_residual = lin_model.predict(input_features)[0]
        gb_residual = gb_model.predict(input_features)[0]
        # make calibrated predictions
        lin_corrected_depth = depth + lin_residual
        gb_corrected_depth = depth + gb_residual
        face_center = face_centers.get(box)
        if face_center is None:
            x1, y1, x2, y2 = (float(value) for value in box)
            face_center = ((x1 + x2) / 2, y1 + 0.12 * (y2 - y1))
        # annotate
        annotated_img = annotate_image(
            snapshot,
            depth,
            box,
            lin_corrected_depth,
            gb_corrected_depth,
            text_color,
            face_center
        )
    # publish image
    return annotated_img

# function to annotate depth onto image
def annotate_image(image: np.ndarray, depth: float, box, lin_corrected_depth: float, gb_corrected_depth: float, text_color, face_center=None):
    x1, y1, x2, y2 = box
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    # draw rectangle with filled background
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if face_center is not None:
        center = tuple(int(round(value)) for value in face_center)
        cv2.circle(image, center, 11, (255, 255, 255), -1)
        cv2.circle(image, center, 8, (0, 0, 0), -1)
        cv2.circle(image, center, 6, (0, 255, 0), -1)
    # Draw depth labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"depth: {depth:.2f} m", (x1, y1 - 40), font, 0.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"linear reg: {lin_corrected_depth:.2f} m", (x1, y1 - 20), font, 0.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"gradient boosting: {gb_corrected_depth:.2f} m", (x1, y1), font, 0.6, text_color, 2, cv2.LINE_AA)
    return image

# function to load pretrained model 
def load_models(model_name):
    model_dir = Path(__file__).resolve().parent.parent/"analysis" / "models"
    model = joblib.load(model_dir / f"{model_name}.joblib")
    return model

# streaming service function
def streaming_service(camera_ip: str):
    # intialize camera, queue, detector, connection, model
    camera = CameraCapture(camera_ip)
    snapshot_queue = Queue(maxsize=1)
    detector = DepthObjectDetector()
    lin_model = load_models("linear_model")
    gb_model = load_models("gb_model")
    # capture fov of the current camera to use in our model for correction
    with socket.create_connection((camera.camera, camera.port)) as connection:
        fov = get_current_fov(connection)
    # start thread to run capture and publish to our queue
    Thread(
        target=capture_loop,
        args=(camera, snapshot_queue),
        daemon=True
    ).start()
    # read from queue and do inference
    while True:
        annotated = inference_loop(detector, snapshot_queue, lin_model, gb_model, fov)
        # display
        if annotated is None or annotated.size == 0:
            continue
        cv2.imshow("Depth Estimation", annotated)
        # listen for quit input from user
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    streaming_service("10.79.59.130")