import requests
import numpy as np
import cv2
import socket
import csv
from datetime import datetime
from pathlib import Path

from .yolotest import DepthObjectDetector
from ..viscaMappings12 import P12STEP, FOVLOOKUPP12

VISCA_PORT = 5500

# image capture
def capture(camera, output_dir=None, filename=None):
    # Capture a single frame from the camera
	url = f"http://{camera}/onvif-http/snapshot?ch2"
	response = requests.get(url, timeout=10)
	response.raise_for_status()

	# Convert image bytes into a NumPy array and decode the image.
	image_bytes = np.frombuffer(response.content, dtype=np.uint8)
	img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError("The response did not contain a valid image")

    # Write the image to the requested output directory.
	if output_dir is None:
		output_dir = Path(__file__).resolve().parents[1] / "datasets" / "captured_data"
	output_dir.mkdir(parents=True, exist_ok=True)
 if filename is None:
		filename = f"capture_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
	output_path = output_dir / filename
	if not cv2.imwrite(str(output_path), img):
		raise IOError(f"Could not write image to {output_path}")

	return img

# control FOV
def changeFOV(camera, zoom):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((camera, VISCA_PORT))
    command = f'81 01 04 07 {zoom} FF'
    cmd = bytes.fromhex(command)
    s.sendall(cmd)
    # wait until we receieve completion or error out
    responses = bytearray()
	while True:
		data = s.recv(1024)
		responses.extend(data)
		while 0xFF in responses:
			end = responses.index(0xFF)
			response = responses[:end + 1]
			responses = responses[end + 1:]
			if response == b'\x90\x51\xFF':
				print("Command completed successfully")
				s.close()
				return 0
			else:
				continue

# dataset Capture function
def captureImages(camera, true_distance, images):
	output_dir = Path(__file__).resolve().parents[1] / "datasets" / "captured_data"
	output_dir /= f"{datetime.now():%m%d}-{true_distance}"
	output_dir.mkdir(parents=True, exist_ok=True)
	for i in range(len(FOVLOOKUPP12)):
		# calculate FOV
		visca = i * P12STEP
		# convert visa dec value to hex
		hex_string = visca.to_bytes(4, byteorder='big').hex(" ").upper()
		# move to FOV and wait until completion
		changeFOV(camera, hex_string)
		# capture images
		for j in range(images):
            filename = f"capture_{j}_FOV{visca}.jpg"
			capture(camera, output_dir, filename)
			# write to CSV
			csv_file = output_dir / "capture_log.csv"
			with open(csv_file, mode='a', newline='') as file:
				writer = csv.writer(file)
				writer.writerow([filename, true_distance, visca, 'prediction'])
	

# Script entry point 
if __name__ == "__main__":
	# load object detector
	detector = DepthObjectDetector()
	# capture image
	img = capture('10.79.59.126:86')
	# get depth of the center of the detected object
	object_box = detector.detect_objects(img)
	depth_map = detector.detect_depth(img)
	predicted = detector.get_center_depth(depth_map, object_box[0])
	print(f"Predicted depth of the center of the detected object: {predicted}")
	# changeFOV
	code = changeFOV('10.79.59.126', P12STEP)
	print(code)