import numpy as np


def weighting_function(B):
    w = np.zeros((B * 2, 1))
    
    for j in range(1, 2 * B + 1):
        result_sum = 0
        for k in range(0, B):
            result_sum = result_sum + 1 / (2 * k + 1) * np.sin((2 * j + 1) * (2 * k + 1) * np.pi / 4 / B)
        w[j - 1] = 2 / B * np.sin(np.pi * (2 * j + 1) / 4 / B) * result_sum
    
    return w
