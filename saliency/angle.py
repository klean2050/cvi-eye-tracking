import cv2, numpy as np


def smap_angle(image, angle):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the Gabor filter
    ksize = 55
    sigma = 3
    theta = np.pi * angle / 180
    lambd = 10
    gamma = 0.5
    psi = 0
    g_kernel = cv2.getGaborKernel(
        (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F
    )

    # Filter the input image with the Gabor filter
    filtered_image = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)

    # Apply the Canny edge detector on the filtered image
    edges = cv2.Canny(filtered_image, 100, 200)

    # Strengthen edges using gaussian blur
    out = cv2.GaussianBlur(edges, (15, 15), 0)

    # Return the saliency map
    return out


def smap_orientation(image):
    smap_0 = smap_angle(image, 0)
    smap_45 = smap_angle(image, 45)
    smap_90 = smap_angle(image, 90)
    smap_135 = smap_angle(image, 135)
    saliency_map = smap_0 + smap_45 + smap_90 + smap_135
    return np.uint8(saliency_map)
