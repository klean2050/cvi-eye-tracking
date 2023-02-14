import numpy as np, torch
from scipy.ndimage import zoom
from scipy.special import logsumexp

import sys

sys.path.append("../DeepGaze")
from deepgaze_pytorch import DeepGazeIIE


def smap_deepgaze(img):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepGazeIIE(pretrained=True).to(DEVICE)

    # load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
    # from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
    # centerbias_template = np.load('centerbias_mit1003.npy')
    centerbias_template = np.zeros((1024, 1024))
    centerbias = zoom(
        centerbias_template,
        (
            img.shape[0] / centerbias_template.shape[0],
            img.shape[1] / centerbias_template.shape[1],
        ),
        order=0,
        mode="nearest",
    )
    centerbias -= logsumexp(centerbias)

    image_tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    log_density_prediction = model(image_tensor, centerbias_tensor)
    return log_density_prediction.squeeze().detach().numpy()
