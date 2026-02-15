import rawpy
import exifread
import numpy as np

def read_exposure_time(path):
    with open(path, 'rb') as f:
        tags = exifread.process_file(f, stop_tag="EXIF ExposureTime")

    exposure_tag = tags.get("EXIF ExposureTime")
    exposure_str = str(exposure_tag)

    if "/" in exposure_str:
        num, denom = exposure_str.split("/")
        return float(num) / float(denom)

    return float(exposure_str)


def load_raw_linear(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_auto_wb=False,
            no_auto_bright=True,
            gamma=(1,1),
            output_bps=16
        )

    return rgb.astype(np.float32) / 65535.0