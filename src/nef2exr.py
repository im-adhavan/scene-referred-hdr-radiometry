import os
import numpy as np
import rawpy
import torch
import OpenEXR
import Imath
from tqdm import tqdm
from src.raw_utils import read_exposure_time
from src.hdr_merge import merge_hdr_gpu
from config import RAW_ROOT, HDR_ROOT


def save_exr(path, hdr):
    height, width, _ = hdr.shape

    header = OpenEXR.Header(width, height)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    exr = OpenEXR.OutputFile(path, header)

    R = hdr[:, :, 0].astype(np.float32).tobytes()
    G = hdr[:, :, 1].astype(np.float32).tobytes()
    B = hdr[:, :, 2].astype(np.float32).tobytes()

    exr.writePixels({'R': R, 'G': G, 'B': B})
    exr.close()


def convert_scene(scene):
    scene_path = os.path.join(RAW_ROOT, scene)
    nef_files = sorted([f for f in os.listdir(scene_path)
                        if f.lower().endswith(".nef")])

    exposures = []
    images = []

    for f in nef_files:
        path = os.path.join(scene_path, f)

        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(
                use_auto_wb=False,
                no_auto_bright=True,
                gamma=(1, 1),
                output_bps=16
            )

        img = rgb.astype(np.float32) / 65535.0
        images.append(torch.from_numpy(img))
        exposures.append(read_exposure_time(path))

    hdr = merge_hdr_gpu(images, exposures)

    os.makedirs(HDR_ROOT, exist_ok=True)
    output_path = os.path.join(HDR_ROOT, f"{scene}.exr")

    save_exr(output_path, hdr)


def convert_all():
    scenes = [d for d in os.listdir(RAW_ROOT)
              if os.path.isdir(os.path.join(RAW_ROOT, d))]

    for scene in tqdm(scenes, desc="Converting NEF → EXR"):
        convert_scene(scene)