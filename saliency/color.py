import numpy as np, cv2


def smap_red(img):
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define first red spectrum
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    # define second red spectrum
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    # return an integer mask
    mask = mask1 + mask2
    return np.uint8(mask)


def smap_green(img):
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define green spectrum
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # return an integer mask
    return np.uint8(mask)


def smap_blue(img):
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define blue spectrum
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # return an integer mask
    return np.uint8(mask)


def smap_yellow(img):
    # convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # define blue spectrum
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([26, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # return an integer mask
    return np.uint8(mask)


def smap_rg_opponency(img):
    r = smap_red(img)
    g = smap_green(img)
    saliency_map = np.abs(r - g)
    return np.uint8(saliency_map)


def smap_by_opponency(img):
    b = smap_blue(img)
    y = smap_yellow(img)
    saliency_map = np.abs(b - y)
    return np.uint8(saliency_map)


def smap_color(img):
    r = smap_red(img)
    g = smap_green(img)
    b = smap_blue(img)
    y = smap_yellow(img)
    saliency_map = r + g + b + y
    return np.uint8(saliency_map)


def smap_intensity(img):
    smap = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.uint8(smap)
