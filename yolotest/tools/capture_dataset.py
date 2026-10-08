import sys
import csv
from pathlib import Path
import argparse
import random
from .utils import compute_box_size
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
            writer.writerow(["filename", "true_distance", "prediction", "size"])
        # search all images in output directory
        for image_path in sorted(output_dir.glob("*.jpg")):
            # convert to cv image
            image = cv2.imread(str(image_path))
            # make prediction, retrieve value from dictionary and write to our csv file
            result = detector.return_human_depth(image, streaming=False, run_id=run_id)
            # handle no detection casegit 
            if not result:
                print(f"No human detected in {image_path.name}")
                writer.writerow([image_path.name, distance, -1, -1])
                continue
            else:
                depths,_,_ = result
            # get depth of the first detected human in the image, if any
            prediction = next(iter(depths.values()), -1) if depths is not None else -1
            # compute box size for the first detected human in the image, if any
            size = compute_box_size(next(iter(depths.keys()), None)) if depths is not None else -1
            writer.writerow([image_path.name, distance, prediction, size])

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
