import torch, numpy as np
import torchvision.transforms as T
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def smap_object(img, prompt):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    inputs = processor(
        text=[prompt],
        images=[img],
        padding="max_length",
        return_tensors="pt"
    )
    with torch.no_grad():
        out = model(**inputs).logits
        out = torch.sigmoid(out)

    # interpolate to original size
    smap = torch.nn.functional.interpolate(
        out.unsqueeze(0).unsqueeze(0),
        size=img.shape[:-1],
        mode="bicubic",
        align_corners=False,
    )
    smap = smap.squeeze().numpy()
    formatted = smap * 255 / np.max(smap)
    return formatted.astype("uint8")