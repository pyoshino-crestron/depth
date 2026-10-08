def compute_box_size(box):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    size = width * height
    return size