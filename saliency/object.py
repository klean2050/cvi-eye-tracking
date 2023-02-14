import numpy as np, cv2


def smap_object(img, bbox):
    """
    Generates a saliency map for the object in the provided bbox
    """
    # TODO: make it work with text prompts (CLIPS)
    height, width = img.shape[:2]
    binary_mask = np.zeros((height, width), dtype=np.uint8)

    x, y, w, h = bbox
    xmin = int(x * width)
    ymin = int(y * height)
    xmax = int((x + w) * width)
    ymax = int((y + h) * height)
    binary_mask[ymin:ymax, xmin:xmax] = 1

    smap = cv2.GaussianBlur(binary_mask, (15, 15), 0)
    return smap
