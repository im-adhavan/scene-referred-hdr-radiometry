import torch
from config import DEVICE, WEIGHT_EPS

def triangular_weight(z):
    return torch.minimum(z, 1.0 - z)


def merge_hdr_gpu(images, exposures):
    images = torch.stack(images).to(DEVICE)
    exposures = torch.tensor(exposures, dtype=torch.float32).to(DEVICE)

    radiance_sum = torch.zeros_like(images[0])
    weight_sum = torch.zeros_like(images[0])

    for i in range(len(images)):
        img = images[i]
        t = exposures[i]
        rad = img / (t + WEIGHT_EPS)
        w = triangular_weight(img)
        radiance_sum += rad * w
        weight_sum += w

    return (radiance_sum / (weight_sum + WEIGHT_EPS)).cpu().numpy()