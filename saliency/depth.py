from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch, numpy as np

def smap_depth(img):

    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(0),
        size=img.shape[:-1],
        mode="bicubic",
        align_corners=False,
    )

    # return the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = output * 255 / np.max(output)
    return formatted.astype("uint8")
