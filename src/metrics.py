import numpy as np

def luminance(hdr):
    return 0.2126*hdr[:,:,0] + 0.7152*hdr[:,:,1] + 0.0722*hdr[:,:,2]


def compute_dr(hdr, low=0.1, high=99.9):
    Y = luminance(hdr)
    Y = Y[Y > 0]
    pl = np.percentile(Y, low)
    ph = np.percentile(Y, high)
    return np.log10(ph/pl)


def compute_log_spread(hdr):
    Y = luminance(hdr)
    Y = Y[Y > 0]
    return np.std(np.log10(Y))