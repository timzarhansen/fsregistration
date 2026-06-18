import numpy as np
from scipy.special import eval_jacobi


def wignerd_function(pitch, l, m1, m2):
    mu = abs(m1 - m2)
    vi = abs(m1 + m2)
    s = l - (mu + vi) / 2

    if m2 >= m1:
        zeta = 1
    else:
        zeta = (-1) ** (m2 - m1)

    d = zeta * np.sqrt(np.math.factorial(s) * np.math.factorial(s + mu + vi) / (np.math.factorial(s + mu) * np.math.factorial(s + vi))) * (np.sin(pitch / 2)) ** mu * (np.cos(pitch / 2)) ** vi * eval_jacobi(s, mu, vi, np.cos(pitch))

    return d


def calculate_score_of_rotation(yaw_to_test, flm1, flm2, B):
    c_output = 0
    roll = 0
    pitch = 0

    for l in range(1, B + 1):
        for m1 in range(1, l + 2):
            for m2 in range(1, l + 2):
                c_output += flm1[l, m1] * flm2[l, m2] * (-1) ** (m1 - m2) * np.exp(-1j * (m1 - 1) * roll) * wignerd_function(pitch, l, m1 - 1, m2 - 1) * np.exp(-1j * (m2 - 1) * yaw_to_test)
                if np.isnan(c_output):
                    print(c_output)

    print("done")
    print(yaw_to_test)
    print(c_output)

    return c_output
