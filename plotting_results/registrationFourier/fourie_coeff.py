import numpy as np
from sympy import symbols, diff, sqrt, factorial


def legendre_p(n, x):
    from sympy import legendre
    return legendre(n, x)


def ylm_of_tp(l, theta, phi):
    ylp = np.zeros((l + 1, theta.shape[0], phi.shape[0]))
    ylm = np.zeros((l, theta.shape[0], phi.shape[0]))

    for m in range(l + 1):
        x = symbols('x')
        leg = legendre_p(l, x)
        leg_diff = diff(leg, x, abs(m))
        plm = ((-1)**abs(m)) * ((1 - x**2)**(abs(m)/2)) * leg_diff
        x0 = np.cos(theta)
        plm_solution_plus = np.array([[float(plm.subs(x, cos_val)) for cos_val in x0_flat] for _ in range(theta.shape[0])])

        if m > 0:
            plm_solution_minus = plm_solution_plus * (-1)**abs(m) * factorial(l - abs(m)) / factorial(l + abs(m))

            a = (2 * l + 1) * factorial(l - m)
            b = 4 * np.pi * factorial(l + m)
            c = np.sqrt(a / b)
            ylp[m, :, :] = (-1)**m * c * plm_solution_plus.T * np.exp(-1j * m * phi)

            a = (2 * l + 1) * factorial(l + m)
            b = 4 * np.pi * factorial(l - m)
            c = np.sqrt(a / b)
            ylm[m, :, :] = (-1)**(-m) * c * plm_solution_minus.T * np.exp(1j * m * phi)
        else:
            a = (2 * l + 1) * factorial(l - m)
            b = 4 * np.pi * factorial(l + m)
            c = np.sqrt(a / b)
            ylp[m, :, :] = (-1)**m * c * plm_solution_plus.T * np.exp(-1j * m * phi)

    return ylp, ylm


def fourie_coeff(f_theta_phi, l_max, B, theta, phi):
    b = B
    weights = np.ones(B * 2)
    flm_p = np.zeros((l_max, l_max + 1))
    flm_m = np.zeros((l_max, l_max))

    for l in range(1, l_max + 1):
        ylp, ylm = ylm_of_tp(l, theta, phi)
        for m in range(l + 1):
            for j in range(2 * B):
                for k in range(2 * B):
                    if m > 0:
                        flm_p[l - 1, m] += weights[j] * f_theta_phi[j, k] * ylp[m, j, k]
                        flm_m[l - 1, m - 1] += weights[j] * f_theta_phi[j, k] * ylm[m - 1, j, k]
                    else:
                        flm_p[l - 1, m] += weights[j] * f_theta_phi[j, k] * ylp[m, j, k]

    flm_p = np.sqrt(2 * np.pi) / (2 * b) * flm_p
    flm_m = np.sqrt(2 * np.pi) / (2 * b) * flm_m

    return flm_p, flm_m
