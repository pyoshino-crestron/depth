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
    url = f"http://{camera}:86/onvif-http/snapshot?ch2"
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
def changeFOV(conn, zoom):
    # timeout if waiting too long
    conn.settimeout(10)
    command = f'81 01 04 47 {zoom} FF'
    cmd = bytes.fromhex(command)
    conn.sendall(cmd)
    # wait until we receieve completion or error out
    responses = bytearray()
    while True:
        print("polling")
        # check for a response and handle timeout
        try:
            data = conn.recv(1024)
            if not data:
                raise ConnectionError("Connection closed by camera")
        except socket.timeout:
            raise TimeoutError("Timeout waiting for response from camera")

        responses.extend(data)
        # filter our buffer for correct response
        while 0xFF in responses:
            end = responses.index(0xFF)
            response = responses[:end + 1]
            responses = responses[end + 1:]
            if response.startswith(b'\x90\x51'):
                print("Command completed successfully")
                return 0
            else:
                continue

# dataset Capture function
def captureImages(camera, true_distance, images, runID=67):
    # establish connection
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((camera, VISCA_PORT))
    # setup output directories
    output_dir = Path(__file__).resolve().parents[1] / "datasets" / "captured_data"
    output_dir /= f"{datetime.now():%m%d}-{true_distance}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # set up csv file
    csv_file = output_dir / f"capture_log_{runID}.csv"
    if not csv_file.exists():
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "true_distance", "visca", "prediction"])
    # take pictures at various FOVs
    for i in range(0,len(FOVLOOKUPP12), 100):
        # calculate FOV
        visca = i * P12STEP
        print(visca)
        # convert visa dec value to hex
        raw_hex = f'{visca:04X}'
        hex_bytes = " ".join(f'0{digit}' for digit in raw_hex)
        # move to FOV and wait until completion
        changeFOV(s, hex_bytes)
        # capture images
        for j in range(images):
            filename = f"capture_{j}_FOV{visca}.jpg"
            capture(camera, output_dir, filename)
            # write to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([filename, true_distance, visca, 'prediction'])

# Script entry point
if __name__ == "__main__":
    # load object detector
    detector = DepthObjectDetector()
    # capture images
    captureImages('10.79.59.126', 20, 1)
