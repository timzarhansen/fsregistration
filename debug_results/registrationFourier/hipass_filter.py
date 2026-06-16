import numpy as np


def hipass_filter(ht, wd):
    # hi-pass filter function
    # ...designed for use with Fourier-Mellin stuff
    res_ht = 1 / (ht - 1)
    res_wd = 1 / (wd - 1)

    eta = np.cos(np.pi * np.arange(-0.5, 0.5 + res_ht, res_ht))
    neta = np.cos(np.pi * np.arange(-0.5, 0.5 + res_wd, res_wd))
    X = np.outer(eta, neta)

    H = (1.0 - X) * (2.0 - X)
    return H
