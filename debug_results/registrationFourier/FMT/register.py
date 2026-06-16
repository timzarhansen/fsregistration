import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import io


def hipass_filter(ht, wd):
    res_ht = 1 / (ht - 1)
    res_wd = 1 / (wd - 1)

    eta = np.cos(np.pi * np.arange(-0.5, 0.5 + res_ht, res_ht))
    neta = np.cos(np.pi * np.arange(-0.5, 0.5 + res_wd, res_wd))
    X = np.outer(eta, neta)

    H = (1.0 - X) * (2.0 - X)

    return H


def register_fourier_mellin():
    I1 = io.imread('lena_cropped.bmp')
    I2 = io.imread('lena_cropped_rotated.bmp')

    plt.figure(8)
    plt.imshow(I1, cmap='gray')
    plt.title('image 1')
    plt.figure(9)
    plt.imshow(I2, cmap='gray')
    plt.title('image 2')

    SizeX = I1.shape[0]
    SizeY = I1.shape[1]

    FA = np.fft.fftshift(np.fft.fft2(I1))
    FB = np.fft.fftshift(np.fft.fft2(I2))
    plt.figure(1)
    plt.imshow(np.abs(FA))
    plt.title('Magnitude 1')
    plt.figure(2)
    plt.imshow(np.abs(FB))
    plt.title('Magnitude 2')

    IA = hipass_filter(I1.shape[0], I1.shape[1]) * np.abs(FA)
    IB = hipass_filter(I2.shape[0], I2.shape[1]) * np.abs(FB)

    plt.figure(3)
    plt.imshow(np.abs(IA))
    plt.title('Magnitude HP 1')
    plt.figure(4)
    plt.imshow(np.abs(IB))
    plt.title('Magnitude HP 2')

    L1 = transform_image(IA, SizeX, SizeY, SizeX, SizeY, 'nearest', I1.shape[0] // 2, I1.shape[1] // 2, 'valid')
    L2 = transform_image(IB, SizeX, SizeY, SizeX, SizeY, 'nearest', I2.shape[0] // 2, I2.shape[1] // 2, 'valid')

    plt.figure(5)
    plt.imshow(L1)
    plt.title('log Polar 1')
    plt.figure(6)
    plt.imshow(L2)
    plt.title('log Polar 2')

    THETA_F1 = np.fft.fft2(L1)
    THETA_F2 = np.fft.fft2(L2)

    a1 = np.angle(THETA_F1)
    a2 = np.angle(THETA_F2)

    THETA_CROSS = np.exp(1j * (a1 - a2))
    THETA_PHASE = np.real(np.fft.ifft2(THETA_CROSS))

    plt.figure(7)
    Xplot, Yplot = np.meshgrid(np.arange(1, THETA_PHASE.shape[0] + 1), np.arange(1, THETA_PHASE.shape[1] + 1))
    ax = plt.axes(projection='3d')
    ax.plot_surface(Xplot, Yplot, THETA_PHASE)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.title('result of Rotation')

    THETA_SORTED = np.sort(THETA_PHASE.flatten())

    SI = len(THETA_SORTED) - 1

    THETA_Y, THETA_X = np.where(THETA_PHASE == THETA_SORTED[SI])
    THETA_Y = THETA_Y[0]
    THETA_X = THETA_X[0]

    DPP = 360 / THETA_PHASE.shape[1]

    Theta = DPP * (THETA_Y - 1)

    R1 = ndimage.rotate(I2, -Theta, reshape=False, clip=True)
    R2 = ndimage.rotate(I2, -(Theta + 180), reshape=False, clip=True)

    R1_F2 = np.fft.fftshift(np.fft.fft2(R1))

    a1 = np.angle(FA)
    a2 = np.angle(R1_F2)

    R1_F2_CROSS = np.exp(1j * (a1 - a2))
    R1_F2_PHASE = np.real(np.fft.ifft2(R1_F2_CROSS))

    R2_F2 = np.fft.fftshift(np.fft.fft2(R2))

    a1 = np.angle(FA)
    a2 = np.angle(R2_F2)

    R2_F2_CROSS = np.exp(1j * (a1 - a2))
    R2_F2_PHASE = np.real(np.fft.ifft2(R2_F2_CROSS))

    MAX_R1_F2 = np.max(R1_F2_PHASE)
    MAX_R2_F2 = np.max(R2_F2_PHASE)

    if MAX_R1_F2 > MAX_R2_F2:
        y, x = np.where(R1_F2_PHASE == np.max(R1_F2_PHASE))
        y, x = y[0], x[0]
        R = R1
    else:
        y, x = np.where(R2_F2_PHASE == np.max(R2_F2_PHASE))
        y, x = y[0], x[0]

        if Theta < 180:
            Theta = Theta + 180
        else:
            Theta = Theta - 180

        R = R2

    Tx = x - 1
    Ty = y - 1

    if x > (I1.shape[0] // 2):
        Tx = Tx - I1.shape[0]

    if y > (I1.shape[1] // 2):
        Ty = Ty - I1.shape[1]

    input2_rectified = R
    move_ht = Ty
    move_wd = Tx

    total_height = max(I1.shape[0], abs(move_ht) + input2_rectified.shape[0])
    total_width = max(I1.shape[1], abs(move_wd) + input2_rectified.shape[1])
    combImage = np.zeros((total_height, total_width))
    registered1 = np.zeros((total_height, total_width))
    registered2 = np.zeros((total_height, total_width))

    if (move_ht >= 0) and (move_wd >= 0):
        registered1[0:I1.shape[0], 0:I1.shape[1]] = I1
        registered2[1 + move_ht:move_ht + input2_rectified.shape[0], 1 + move_wd:move_wd + input2_rectified.shape[1]] = input2_rectified
    elif (move_ht < 0) and (move_wd < 0):
        registered2[0:input2_rectified.shape[0], 0:input2_rectified.shape[1]] = input2_rectified
        registered1[1 + abs(move_ht):abs(move_ht) + I1.shape[0], 1 + abs(move_wd):abs(move_wd) + I1.shape[1]] = I1
    elif (move_ht >= 0) and (move_wd < 0):
        registered2[move_ht + 1:move_ht + input2_rectified.shape[0], 0:input2_rectified.shape[1]] = input2_rectified
        registered1[0:I1.shape[0], abs(move_wd) + 1:abs(move_wd) + I1.shape[1]] = I1
    elif (move_ht < 0) and (move_wd >= 0):
        registered1[abs(move_ht) + 1:abs(move_ht) + I1.shape[0], 0:I1.shape[1]] = I1
        registered2[0:input2_rectified.shape[0], move_wd + 1:move_wd + input2_rectified.shape[1]] = input2_rectified

    if np.sum(registered1 == 0) > np.sum(registered2 == 0):
        plant = registered1
        bleed = registered2
    else:
        plant = registered2
        bleed = registered1

    combImage = plant.copy()
    for p in range(total_height):
        for q in range(total_width):
            if combImage[p, q] == 0:
                combImage[p, q] = bleed[p, q]

    plt.figure(10)
    plt.imshow(combImage, cmap='gray', vmin=0, vmax=255)

    return Theta, Tx, Ty, R


def transform_image(A, Ar, Ac, Nrho, Ntheta, method, Center, Shape):
    theta = np.linspace(0, 2 * np.pi, Ntheta + 1)
    theta[-1] = []

    if Shape == 'full':
        corners = np.array([[1, 1], [Ar, 1], [Ar, Ac], [1, Ac]])
        d = np.max(np.sqrt(np.sum((np.tile(Center, (4, 1)) - corners) ** 2, axis=1)))
    elif Shape == 'valid':
        d = min([Ac - Center[0], Center[0] - 1, Ar - Center[1], Center[1] - 1])

    minScale = 1
    rho = np.logspace(np.log10(minScale), np.log10(d), Nrho)

    xx = np.outer(rho, np.cos(theta)) + Center[0]
    yy = np.outer(rho, np.sin(theta)) + Center[1]

    if method == 'nearest':
        r = ndimage.map_coordinates(A, [yy, xx], order=0, mode='constant', cval=0)
    elif method == 'bilinear':
        r = ndimage.map_coordinates(A, [yy, xx], order=1, mode='constant', cval=0)
    elif method == 'bicubic':
        r = ndimage.map_coordinates(A, [yy, xx], order=3, mode='constant', cval=0)
    else:
        raise ValueError(f'Unknown interpolation method: {method}')

    mask = (xx > Ac) | (xx < 1) | (yy > Ar) | (yy < 1)
    r[mask] = 0

    return r
