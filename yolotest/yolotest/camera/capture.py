import socket
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from ..viscaMappings12 import FOVLOOKUPP12, P12STEP
from .visca import VISCA_PORT, change_fov

# a class to help with camera objects and capturing images and changing FOVs
class CameraCapture:
    def __init__(self, camera: str, output_dir: Path, port: int = VISCA_PORT) -> None:
        self.camera = camera
        self.output_dir = output_dir
        self.port = port

    # captures an image and writes it to our photos directory
    def capture(self) -> Path:
        # capture an image from camera via api
        response = requests.get(f"http://{self.camera}:86/onvif-http/snapshot?ch2", timeout=10)
        response.raise_for_status()
        # convert response to an image
        image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The response did not contain a valid image")
        return image

    # capture images while changing FOVs
    def capture_fovs(self, images: int, run_id: str, distance: float) -> Path:
        # create output path
        output_dir = self.output_dir / f"{datetime.now():%m%d}-{distance}-{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # open connection to camera
        with socket.create_connection((self.camera, self.port)) as connection:
            for index in range(0, len(FOVLOOKUPP12), 100):
                # set visca command for zoom
                visca = index * P12STEP
                fov = FOVLOOKUPP12[index]
                zoom = " ".join(f"0{digit}" for digit in f"{visca:04X}")
                # move camera 
                change_fov(connection, zoom)
                # capture images at set FOV x Distance
                for image_index in range(images):
                    img  = self.capture()
                    cv2.imwrite(f"{output_dir}/{fov}-{image_index}.jpg", img)
        return output_dir
