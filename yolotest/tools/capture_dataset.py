import argparse
import csv
from pathlib import Path
import argparse

from yolotest.yolotest.camera.capture import CameraCapture
from yolotest.yolotest.detection.detector import DepthObjectDetector


def main(camera_ip, distance, images, run_id, output):
    # recieve dataset parameters
    output_dir = CameraCapture(camera_ip, output).capture_fovs(images, run_id, distance)
    detector = DepthObjectDetector()
    with (output_dir / f"capture_log_{run_id}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "true_distance", "prediction"])
        for image_path in sorted(output_dir.glob("*.jpg")):
            writer.writerow([image_path.name, distance, detector.return_human_depth(image_path)])


if __name__ == "__main__":
    output = Path(__file__).resolve().parent.parent / "yolotest" / "datasets"
    main("10.79.59.126", 3.73, 3, 1, output)
