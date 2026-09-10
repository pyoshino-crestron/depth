import numpy as np
from pathlib import Path
from PIL import Image

# get the corresponding depth map for a given image. used when we have a labeled dataset of images and depth maps.
def get_ground_truth_depth(image_path):
    image_path = Path(image_path)

    depth_path = (
        image_path.parent.parent.parent
        / "depth"
        / image_path.parent.name
        / f"{image_path.stem}.png"
    )

    with Image.open(depth_path) as depth_image:
        depth_map = np.array(depth_image) // 1000 # milimeters to meters

    return depth_map

# Extract depth statistics from a bounding box.
def extract_from_box(depth_map, xyxy):
    depth_map = np.asarray(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("depth_map must be a two-dimensional matrix")

    x1, y1, x2, y2 = xyxy

    # Convert floating-point box coordinates to a pixel region and clip it.
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(depth_map.shape[1], int(np.ceil(x2)))
    y2 = min(depth_map.shape[0], int(np.ceil(y2)))

    if x1 >= x2 or y1 >= y2:
        raise ValueError("xyxy does not contain any pixels in depth_map")

    # For an even-sized region, (size - 1) // 2 selects the left/top pixel.
    center_y = y1 + (y2 - y1 - 1) // 2
    center_x = x1 + (x2 - x1 - 1) // 2
    center_depth = depth_map[center_y, center_x].item()

    return center_depth