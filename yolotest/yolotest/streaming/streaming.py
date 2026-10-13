from pathlib import Path
from queue import Queue
import socket
from threading import Event, Thread

import cv2
import joblib
import numpy as np
import pandas as pd

from ..camera.capture import CameraCapture
from ..camera.visca import get_current_fov
from ..detection.detector import DepthObjectDetector

# loop to continuously capture images from the camera and put them into a queue for processing
def capture_loop(camera: CameraCapture, snapshot_queue: Queue, connection, stop_event: Event) -> None:
    while not stop_event.is_set():
        fov = get_current_fov(connection)
        snapshot = camera.capture()
        if snapshot_queue.full():
            snapshot_queue.get_nowait()
        snapshot_queue.put((snapshot, fov))

# inference loop to process images from the queue, perform depth estimation, and annotate the images
def inference_loop(
    detector: DepthObjectDetector,
    snapshot_queue: Queue,
    lin_model,
    gb_model
) -> np.ndarray | None:

    # Get the latest snapshot and fov from the queue
    snapshot, fov = snapshot_queue.get()
    if snapshot is None or snapshot.size == 0:
        return None
    # Perform depth estimation using the detector
    detection_result = detector.return_human_depth(snapshot, streaming=True)
    # No detection return same image
    if detection_result is None:
        return snapshot
    # Get the box to depth mapping, face centers, and depth map from the detection result
    box_to_depth, face_centers, depth_map = detection_result
    if depth_map is None or depth_map.size == 0:
        return snapshot
    # Prepare heatmap for overlay onto original image
    height, width = snapshot.shape[:2]
    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_LINEAR)
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_depth = cv2.normalize(
        depth_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)
    # blend heatmap with original image
    heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    snapshot = cv2.addWeighted(snapshot, 0.55, heatmap, 0.45, 0)

    if not box_to_depth:
        return snapshot
    # Loop through each detected box and its corresponding depth
    for index, (box, depth) in enumerate(box_to_depth.items()):
        x1, y1, x2, y2 = (float(value) for value in box)
        size = (x2 - x1) * (y2 - y1)
        input_features = pd.DataFrame({
            "prediction": [depth],
            "fov": [fov],
            "size": [size],
        })
        
        # Predict residuals using the linear and gradient boosting models
        hue = int(index * 180 / len(box_to_depth))
        hsv_color = np.uint8([[[hue, 255, 255]]])
        text_color = tuple(int(channel) for channel in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        lin_residual = lin_model.predict(input_features)[0]
        gb_residual = gb_model.predict(input_features)[0]
        lin_corrected_depth = depth + lin_residual
        gb_corrected_depth = depth + gb_residual
        
        # Get the face center for annotation
        face_center = face_centers.get(box)
        if face_center is None:
            face_center = ((x1 + x2) / 2, y1 + 0.12 * (y2 - y1))

        # Annotate the image with the depth information and bounding box
        annotate_image(
            snapshot,
            depth,
            box,
            lin_corrected_depth,
            gb_corrected_depth,
            text_color,
            face_center,
        )
    return snapshot

# Helper function to annotate the image with depth information, bounding boxes, and face centers
def annotate_image(
    image: np.ndarray,
    depth: float,
    box,
    lin_corrected_depth: float,
    gb_corrected_depth: float,
    text_color,
    face_center=None,
) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if face_center is not None:
        center = tuple(int(round(value)) for value in face_center)
        cv2.circle(image, center, 11, (255, 255, 255), -1)
        cv2.circle(image, center, 8, (0, 0, 0), -1)
        cv2.circle(image, center, 6, (0, 255, 0), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"depth: {depth:.2f} m", (x1, y1 - 40), font, 0.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"linear reg: {lin_corrected_depth:.2f} m", (x1, y1 - 20), font, 0.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"gradient boosting: {gb_corrected_depth:.2f} m", (x1, y1), font, 0.6, text_color, 2, cv2.LINE_AA)
    return image

# function to load models from directory
def load_models(model_name):
    model_dir = Path(__file__).resolve().parent.parent / "analysis" / "models"
    return joblib.load(model_dir / f"{model_name}.joblib")

# main function for handling streaming
def streaming_service(camera_ip: str):
    camera = CameraCapture(camera_ip)
    snapshot_queue = Queue(maxsize=1)
    detector = DepthObjectDetector()
    lin_model = load_models("linear_model")
    gb_model = load_models("gb_model")
    # create an event that is not set while streaming is running and set when we want to stop the streaming
    stop_event = Event()
    with socket.create_connection((camera.camera, camera.port)) as connection:
        capture_thread = Thread(
            target=capture_loop,
            args=(camera, snapshot_queue, connection, stop_event),
            daemon=True,
        )
        capture_thread.start()

        try:
            while True:
                annotated = inference_loop(detector, snapshot_queue, lin_model, gb_model)
                if annotated is None or annotated.size == 0:
                    continue
                cv2.imshow("Depth Estimation", annotated)
                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            stop_event.set()
            capture_thread.join()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    streaming_service("10.79.59.130")