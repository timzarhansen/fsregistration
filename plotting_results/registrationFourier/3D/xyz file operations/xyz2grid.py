import numpy as np
import os


def xyzread(filename, *varargin):
    if not isinstance(filename, str):
        raise ValueError('Input error: filename must be a string.')

    if not os.path.exists(filename):
        raise FileNotFoundError(f'Cannot find file {filename}.')

    skiprows = 0
    i = 0
    while i < len(varargin):
        if varargin[i] == 'headerlines' and i + 1 < len(varargin):
            skiprows = varargin[i + 1]
            i += 2
        else:
            i += 1

    data = np.loadtxt(filename, skiprows=skiprows)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    return x, y, z


def accumarray(subscripts, values, size=None, func=None, fill_value=0):
    if size is None or size == []:
        size = (np.max(subscripts[0]) + 1, np.max(subscripts[1]) + 1)

    result = np.full(size, fill_value)

    for i in range(len(values)):
        idx = (subscripts[0][i], subscripts[1][i])
        if func is None:
            result[idx] += values[i]
        else:
            result[idx] = func(result[idx], values[i])

    return result


def xyz2grid(*varargin):
    if len(varargin) < 1:
        raise ValueError('Not enough input arguments.')

    if isinstance(varargin[0], (list, np.ndarray)) and np.all(np.isreal(np.array(varargin[0]))):
        x = varargin[0]
        y = varargin[1]
        z = varargin[2]
    else:
        x, y, z = xyzread(*varargin)

    assert len(x) == len(y) == len(z), 'Dimensions of x, y, and z must match.'
    assert np.ndim(x) == 1, 'Inputs x, y, and z must be vectors.'

    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)

    if len(xs) == len(z):
        print('Warning: It does not seem like the xyz dataset is gridded. You may be attempting to grid scattered data, but I will try to put it into a 2D matrix anyway. Check the output spacing of X and Y.')

    Z = accumarray([yi, xi], z, [], [], np.nan)

    Z = np.flipud(Z)

    nargout = len(varargin) if len(varargin) > 3 else 1

    if nargout == 1:
        return Z
    elif nargout == 3:
        X, Y = np.meshgrid(xs, np.flipud(ys))
        return X, Y, Z
    else:
        raise ValueError('Wrong number of outputs.')
