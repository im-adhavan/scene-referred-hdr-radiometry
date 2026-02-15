import os
import torch

RAW_ROOT = r"C:\hdr_perceptual\data\raw"
HDR_ROOT = r"C:\hdr_perceptual\data\hdr_exr"
OUTPUT_DIR = os.path.join("results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_EPS = 1e-8