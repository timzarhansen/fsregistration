import numpy as np
import matplotlib.pyplot as plt


def plot_ffts2d(voxel_data, figure_number):
    plt.figure(figure_number)
    plt.subplot(1, 3, 1)

    plt.imshow(voxel_data, aspect='equal')
    plt.title('Voxel: ' + str(figure_number))
    plt.axis('image')

    # calculate 2dFFT
    fft_output = np.fft.fftshift(np.fft.fft2(voxel_data))

    plt.subplot(1, 3, 2)
    magnitude = np.abs(fft_output)
    plt.imshow(magnitude, aspect='equal')
    plt.title('Magnitude Voxel: ' + str(figure_number))
    plt.axis('image')

    imaginary_part = np.imag(fft_output)
    real_part = np.real(fft_output)
    phase = np.arctan2(imaginary_part, real_part)
    plt.subplot(1, 3, 3)
    plt.imshow(phase, aspect='equal')
    plt.title('Phase Voxel: ' + str(figure_number))
    plt.axis('image')

    return fft_output, magnitude, phase
