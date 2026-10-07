import sys
import csv
from pathlib import Path
import argparse
import random
from yolotest.yolotest.camera.capture import CameraCapture
from yolotest.yolotest.detection.detector import DepthObjectDetector
import cv2

# function to perform inference on an existing dataset
def inference(distance, output_dir, run_id):
    # create detector object
    detector = DepthObjectDetector()
    # build csv path and check for existence
    csv_path = Path(output_dir).parent / "logs" / f"capture_log_{run_id}.csv"
    csv_exists = csv_path.exists()
    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not csv_exists:
            writer.writerow(["filename", "true_distance", "prediction"])
        # search all images in output directory
        for image_path in sorted(output_dir.glob("*.jpg")):
            # convert to cv image
            image = cv2.imread(str(image_path))
            # make prediction, retrieve value from dictionary and write to our csv file
            depths = detector.return_human_depth(image, streaming=False, run_id=run_id)
            prediction = next(iter(depths.values()), -1) if depths is not None else -1
            writer.writerow([image_path.name, distance, prediction])

# function to perform image capture
def intake(camera_ip, distance, images):
    # establish run_id and perform full capture
    run_id = random.randint(1,10000)
    # recieve dataset parameters
    output_dir = CameraCapture(camera_ip, output).capture_fovs(images, run_id, distance)  
    return output_dir, run_id

# function to capture images, run our detector on the images, write results to csv
def main(camera_ip, distance, images, output_dir=None, run_id=None):
    # run capture if we do not have a dataset
    if not output_dir or not run_id:
        output_dir, run_id = intake(camera_ip, distance, images)
    inference(distance, output_dir, run_id)

if __name__ == "__main__":
    output = Path(__file__).resolve().parent.parent / "yolotest" / "datasets"
    main("10.79.59.130", 5.169, 3)
