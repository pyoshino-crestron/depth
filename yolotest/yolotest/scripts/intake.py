import requests
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

from .yolotest import DepthObjectDetector


# image capture
def capture(camera):
    # Capture a single frame from the camera
	url = f"http://{camera}/onvif-http/snapshot?ch2"
	response = requests.get(url, timeout=10)
	response.raise_for_status()

	# Convert image bytes into a NumPy array and decode the image.
	image_bytes = np.frombuffer(response.content, dtype=np.uint8)
	img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError("The response did not contain a valid image")

	# Write the image to depth/yolotest/yolotest/datasets/captured_data.
	output_dir = Path(__file__).resolve().parents[1] / "datasets" / "captured_data"
	output_path = output_dir / f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
	if not cv2.imwrite(str(output_path), img):
		raise IOError(f"Could not write image to {output_path}")

	return img

# Script entry point 
if __name__ == "__main__":
	# load object detector
	detector = DepthObjectDetector()
	capture('172.30.144.182:86')
