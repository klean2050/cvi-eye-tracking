import cv2


def smap_rough(img):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, smap = saliency.computeSaliency(img)
    return (smap * 255).astype("uint8")


def smap_fine(img):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    _, smap = saliency.computeSaliency(img)
    return (smap * 255).astype("uint8")
