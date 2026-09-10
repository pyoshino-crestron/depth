import requests
import numpy as np
import cv2
import socket
from datetime import datetime
from pathlib import Path

from .yolotest import DepthObjectDetector
from ..viscaMappings12 import P12STEP, FOVLOOKUP12

VISCA_PORT = 5500

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

# control FOV
def changeFOV(camera, zoom):
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.settimeout(5)
		s.connect((camera, VISCA_PORT))
		command = f'81 01 04 07 {zoom} FF'
		cmd = bytes.fromhex(command)
		s.sendall(cmd)
		# get response 
		res = s.recv(1024)
	return res

# Script entry point 
if __name__ == "__main__":
	# load object detector
	detector = DepthObjectDetector()
	# capture image
	img = capture('172.30.144.182:86')
	# get depth of the center of the detected object
	object_box = detector.detect_objects(img)
	depth_map = detector.detect_depth(img)
	predicted = detector.get_center_depth(depth_map, object_box[0])
	print(f"Predicted depth of the center of the detected object: {predicted}")
