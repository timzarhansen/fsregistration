import numpy as np
from scipy import ndimage, signal
from skimage import measure
import warnings


def fast_2d_peak_find(d, thres=None, filt=None, edg=3, res=1, fid=None):
    if d is None:
        d = np.uint16(signal.convolve2d(np.reshape((2**14 * (np.random.rand(1, 1024*1024) > 0.99995)).astype(np.single), [1024, 1024]),
                                        np.ones((15, 15)) / (15*15), 'same') + 2**8 * np.random.rand(1024, 1024))
        return d

    if len(d.shape) > 2:
        d = np.uint16(np.mean(d, axis=2))

    if np.issubdtype(d.dtype, np.floating):
        if np.max(d) <= 1:
            d = np.uint16(d * 2**16 / np.max(d))
        else:
            d = np.uint16(d)

    if thres is None:
        thres = np.max([np.min(np.max(d, axis=0)), np.min(np.max(d, axis=1))])

    if filt is None:
        filt = np.outer(np.exp(-np.arange(4)**2), np.exp(-np.arange(4)**2))
        filt = filt / np.sum(filt)

    if np.size(thres) > 1:
        raise ValueError('threshold has to be a scalar')

    if np.any(d):
        d = ndimage.median_filter(d.astype(float), size=(3, 3))

        if np.issubdtype(d.dtype, np.uint8):
            d = d * (d > thres).astype(np.uint8)
        else:
            d = d * (d > thres).astype(np.uint16)

        if np.any(d):
            d = signal.convolve2d(d.astype(np.single), filt, mode='same')
            d = d * (d > 0.9 * thres)

            if res == 1:
                sd = d.shape
                y, x = np.where(d[edg:sd[0]-edg, edg:sd[1]-edg])

                cent = []
                cent_map = np.zeros(sd)

                x = x + edg
                y = y + edg

                for j in range(len(y)):
                    if (d[x[j], y[j]] > d[x[j]-1, y[j]-1] and
                        d[x[j], y[j]] > d[x[j]-1, y[j]] and
                        d[x[j], y[j]] > d[x[j]-1, y[j]+1] and
                        d[x[j], y[j]] > d[x[j], y[j]-1] and
                        d[x[j], y[j]] > d[x[j], y[j]+1] and
                        d[x[j], y[j]] > d[x[j]+1, y[j]-1] and
                        d[x[j], y[j]] > d[x[j]+1, y[j]] and
                        d[x[j], y[j]] > d[x[j]+1, y[j]+1]):

                        cent.extend([y[j], x[j]])
                        cent_map[x[j], y[j]] += 1

                cent = np.array(cent)

            elif res == 2:
                from skimage.measure import regionprops

                props = measure.regionprops(measure.label(d > 0), intensity_image=d)

                areas = [p.area for p in props]
                mean_area = np.mean(areas)
                std_area = np.std(areas)
                limit = mean_area + 2 * std_area

                rel_peaks_mask = np.array([a <= limit for a in areas])
                centroids = [props[i].weighted_centroid for i in range(len(props)) if rel_peaks_mask[i]]

                cent = np.array(centroids).flatten()
                cent_map = None

            if fid is not None:
                np.savetxt(fid, cent.reshape(-1, 1), fmt='%f')
        else:
            cent = np.array([])
            cent_map = np.zeros(d.shape)
    else:
        cent = np.array([])
        cent_map = np.zeros(d.shape)

    return cent, cent_map


import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate, shift
from skimage import morphology, measure, feature
from skimage.filters import threshold_local
from skimage.morphology import disk, tophat, extended_max
from matplotlib import cm
import matplotlib.colors as mcolors


def plot_all_solutions_in_registration():
    magnitude_fftw1 = np.genfromtxt("csvFiles/magnitudeFFTW1.csv", delimiter=",")
    n = int(np.sqrt(magnitude_fftw1.shape[0]))

    phase_fftw1 = np.genfromtxt("csvFiles/phaseFFTW1.csv", delimiter=",")
    voxel_data_used1 = np.genfromtxt("csvFiles/voxelDataFFTW1.csv", delimiter=",")

    magnitude_fftw2 = np.genfromtxt("csvFiles/magnitudeFFTW2.csv", delimiter=",")
    phase_fftw2 = np.genfromtxt("csvFiles/phaseFFTW2.csv", delimiter=",")
    voxel_data_used2 = np.genfromtxt("csvFiles/voxelDataFFTW2.csv", delimiter=",")

    magnitude1 = np.zeros((n, n))
    phase1 = np.zeros((n, n))
    voxel_data1 = np.zeros((n, n))
    magnitude2 = np.zeros((n, n))
    phase2 = np.zeros((n, n))
    voxel_data2 = np.zeros((n, n))

    for j in range(n):
        for i in range(n):
            magnitude1[i, j] = magnitude_fftw1[(i - 1) * n + j]
            phase1[i, j] = phase_fftw1[(i - 1) * n + j]
            voxel_data1[i, j] = voxel_data_used1[(i - 1) * n + j]
            magnitude2[i, j] = magnitude_fftw2[(i - 1) * n + j]
            phase2[i, j] = phase_fftw2[(i - 1) * n + j]
            voxel_data2[i, j] = voxel_data_used2[(i - 1) * n + j]

    magnitude1 = np.fft.fftshift(magnitude1)
    magnitude2 = np.fft.fftshift(magnitude2)

    plt.figure(11)
    plt.clf()
    plt.imshow(magnitude1, aspect='auto')

    plt.figure(22)
    plt.clf()
    plt.imshow(magnitude2, aspect='auto')

    plt.figure(1)
    plt.clf()
    plt.title('Voxel: 1')
    plt.imshow(voxel_data1, aspect='auto')

    plt.figure(2)
    plt.clf()
    plt.title('Voxel: 2')
    plt.imshow(voxel_data2, aspect='auto')

    from skimage.io import imsave
    imsave('csvFiles/testImages/image1.jpeg', voxel_data1 / voxel_data1.max(), plugin='pillow')
    imsave('csvFiles/testImages/image2.jpeg', voxel_data2 / voxel_data2.max(), plugin='pillow')

    data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")

    number_of_solutions_overall = 0
    for i in range(int(data_information[0])):
        number_of_solutions_overall += data_information[i + 1]

    cell_size = data_information[int(data_information[0]) + 1]

    plt.figure(3)
    plt.clf()
    voxel_result1_fftw = np.genfromtxt("csvFiles/resultVoxel1.csv", delimiter=",")
    n = int(np.sqrt(voxel_result1_fftw.shape[0]))

    voxel_result2_fftw = np.genfromtxt("csvFiles/resultVoxel2.csv", delimiter=",")

    voxel_result1 = np.zeros((n, n))
    voxel_result2 = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            voxel_result1[k, j] = voxel_result1_fftw[k * n - n + j]
            voxel_result2[k, j] = voxel_result2_fftw[k * n - n + j]

    plt.subplot(1, 2, 1)
    plt.imshow(voxel_result1)
    plt.title('Voxel 1: ')
    plt.axis('image')

    plt.subplot(1, 2, 2)
    from skimage.exposure import equalize_hist
    blended = (voxel_result1 + voxel_result2) / 2
    plt.imshow(blended, cmap='gray')
    plt.title('Voxel 2: ')
    plt.axis('image')

    resampled_data_for_sphere1 = np.genfromtxt("csvFiles/resampledVoxel1.csv", delimiter=",")
    resampled_data_for_sphere2 = np.genfromtxt("csvFiles/resampledVoxel2.csv", delimiter=",")
    size_of_spheres = int(np.sqrt(resampled_data_for_sphere1.shape[0]))

    resampled_data_for_sphere_result1 = np.zeros((size_of_spheres, size_of_spheres))
    resampled_data_for_sphere_result2 = np.zeros((size_of_spheres, size_of_spheres))

    for j in range(size_of_spheres):
        for i in range(size_of_spheres):
            resampled_data_for_sphere_result1[j, i] = resampled_data_for_sphere1[(i - 1) * size_of_spheres + j]
            resampled_data_for_sphere_result2[j, i] = resampled_data_for_sphere2[(i - 1) * size_of_spheres + j]

    if True:
        fig = plt.figure(4)
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, facecolors=plt.cm.jet(resampled_data_for_sphere_result1 / resampled_data_for_sphere_result1.max()), edgecolor='none')
        plt.axis('equal')

        fig = plt.figure(5)
        ax = fig.add_subplot(111, projection='3d')
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, facecolors=plt.cm.jet(resampled_data_for_sphere_result2 / resampled_data_for_sphere_result2.max()), edgecolor='none')
        plt.axis('equal')
    else:
        plt.figure(4)
        plt.imshow(resampled_data_for_sphere_result1)
        plt.axis('image')
        plt.figure(5)
        plt.imshow(resampled_data_for_sphere_result2)
        plt.axis('image')

    results = np.genfromtxt("csvFiles/resultCorrelation3D.csv", delimiter=",")
    results = results / results.max()
    result_size = int(np.round(len(results) ** (1/3)))

    a = results.reshape(result_size, result_size, result_size)

    correlation_number_matrix = a[:, :, 0]
    plt.figure(9)
    plt.imshow(correlation_number_matrix)
    plt.axis('image')

    plt.figure(9)
    every_nth_element = 5
    size_all = a.shape[0]
    b = a[0::every_nth_element, 0::every_nth_element, 0::every_nth_element]
    actual_size = int(np.ceil(size_all / every_nth_element))

    xx, yy, zz = np.meshgrid(np.arange(0, size_all, every_nth_element), np.arange(0, size_all, every_nth_element), np.arange(0, size_all, every_nth_element))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=b.flatten(), cmap='jet', s=5)
    plt.colorbar(sc)
    plt.axis('equal')
    ax.set_xlabel("z1-axis")
    ax.set_ylabel("z2-axis")
    ax.set_zlabel("y-axis")

    print("z1: " + str(31/64*360))
    print("y: " + str(39/64*360))
    print("z2: " + str(45/64*360))

    print("z1: " + str(63/64*360))
    print("y: " + str(39/64*360))
    print("z2: " + str(63/64*360))

    print("z1: " + str(31/64*360))
    print("y: " + str(39/64*360))
    print("z2: " + str(45/64*360))

    plt.figure(6)
    plt.clf()
    correlation_of_angles = np.genfromtxt("csvFiles/resultingCorrelation1D.csv", delimiter=",")
    x_for_plot = np.arange(len(correlation_of_angles))
    x_for_plot = x_for_plot / len(correlation_of_angles) * 360
    plt.plot(x_for_plot, correlation_of_angles)
    plt.ylabel("Correlation Value")
    plt.xlabel("angle in rad")

    plt.figure(10)
    results4d = np.genfromtxt("csvFiles/resultCorrelation4D.csv", delimiter=",")
    results4d = results4d / results4d.max()
    result_size4d = int(np.round(len(results4d) ** (1/3)))

    test4d = results4d.reshape(result_size4d, result_size4d, result_size4d)

    every_nth_element = 2
    test3d = test4d[0::every_nth_element, 0::every_nth_element, 0::every_nth_element]
    size_all = test4d.shape[0]
    actual_size = int(np.ceil(size_all / every_nth_element))

    xx, yy, zz = np.meshgrid(np.arange(0, size_all, every_nth_element), np.arange(0, size_all, every_nth_element), np.arange(0, size_all, every_nth_element))
    fig = plt.figure(10)
    ax = fig.add_subplot(111, projection='3d')
    h = ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(), c=test3d.flatten(), s=5, alpha=0.2)
    plt.colorbar(h)
    plt.axis('equal')
    ax.set_xlabel("z1-axis")
    ax.set_ylabel("z2-axis")
    ax.set_zlabel("y-axis")

    data_information = np.genfromtxt("csvFiles/dataForReadIn.csv", delimiter=",")
    last_number_of_points = 0

    for current_number_correlation in range(int(data_information[0])):
        correlation_matrix_shift1d = np.genfromtxt(f"csvFiles/resultingCorrelationShift_{current_number_correlation}.csv", delimiter=",")
        result_size = int(np.sqrt(len(correlation_matrix_shift1d)))
        correlation_matrix_shift2d = np.zeros((result_size, result_size))

        for j in range(result_size):
            for i in range(result_size):
                correlation_matrix_shift2d[i, j] = correlation_matrix_shift1d[(i - 1) * result_size + j]

        for current_translation_number in range(1, int(data_information[current_number_correlation + 1]) + 1):
            plt.figure(8)
            plt.clf()
            x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
            ax = plt.subplot(111, projection='3d')
            ax.plot_surface(x_plot, y_plot, correlation_matrix_shift2d, edgecolor='none')
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.hold(True)

            print("number of interest:")
            print(current_translation_number + last_number_of_points - 1)

            initial_guess_read_in = np.genfromtxt("csvFiles/initialGuess.csv", delimiter=",")
            initial_guess_read_in[0, 3] = initial_guess_read_in[0, 3] / cell_size
            initial_guess_read_in[1, 3] = initial_guess_read_in[1, 3] / cell_size
            x_of_initial_guess = -initial_guess_read_in[0, 3] + np.ceil(result_size / 2)
            y_of_initial_guess = -initial_guess_read_in[1, 3] + np.ceil(result_size / 2)

            z_for_plot = correlation_matrix_shift2d[int(round(y_of_initial_guess)), int(round(x_of_initial_guess))]
            ax.plot3D([x_of_initial_guess], [y_of_initial_guess], [z_for_plot], 'r+')

            potential_transformation_read_in = np.genfromtxt(f"csvFiles/potentialTransformation{current_translation_number - 1 + last_number_of_points}.csv", delimiter=",")
            transformation_tmpsi = potential_transformation_read_in[0:4, 0:4]
            potential_transformation_read_in[0, 3] = potential_transformation_read_in[0, 3] / cell_size
            potential_transformation_read_in[1, 3] = potential_transformation_read_in[1, 3] / cell_size
            correlation_matrix = potential_transformation_read_in[5:7, 0:2]

            eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

            transformation_tmp_pixel = potential_transformation_read_in[0:4, 0:4]
            potential_peak_height_tmp = potential_transformation_read_in[4, 0]
            x_of_interest = -potential_transformation_read_in[0, 3] + np.ceil(result_size / 2)
            y_of_interest = -potential_transformation_read_in[1, 3] + np.ceil(result_size / 2)

            ax.quiver(x_of_interest, y_of_interest, 0, eigenvalues[0] * eigenvectors[0, 0], eigenvalues[0] * eigenvectors[0, 1], 0)
            ax.quiver(x_of_interest, y_of_interest, 0, eigenvalues[1] * eigenvectors[1, 0], eigenvalues[1] * eigenvectors[1, 1], 0)
            z_for_plot = correlation_matrix_shift2d[int(round(y_of_interest)), int(round(x_of_interest))]
            ax.plot3D([x_of_interest], [y_of_interest], [z_for_plot], 'r+')

            r1 = 2 * np.sqrt(eigenvalues[0])
            r2 = 2 * np.sqrt(eigenvalues[1])
            teta = np.arange(-np.pi, np.pi, 0.01)
            ellipse_x_r = r1 * np.cos(teta)
            ellipse_y_r = r2 * np.sin(teta)
            phi = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])
            R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
            r_ellipse = np.dot(np.array([ellipse_x_r, ellipse_y_r]), R)
            ax.plot3D(r_ellipse[0] + x_of_interest, r_ellipse[1] + y_of_interest, np.ones_like(teta) * z_for_plot)

            ax.view_init(90, 0)

            plt.figure(7)
            plt.clf()
            voxel_data_tmp1 = voxel_data1
            voxel_data_tmp2 = voxel_data2

            plt.subplot(1, 2, 1)
            blended_before = (voxel_data_tmp1 + voxel_data_tmp2) / 2
            plt.imshow(blended_before, cmap='gray')
            plt.title('Before Match: ')
            plt.axis('image')

            plt.subplot(1, 2, 2)
            angle_of_interest = np.arctan2(transformation_tmp_pixel[1, 0], transformation_tmp_pixel[0, 0])
            voxel_data_tmp2_shifted = ndimage.shift(voxel_data_tmp2, [-transformation_tmp_pixel[0, 3], -transformation_tmp_pixel[1, 3]])
            voxel_data_tmp2_rotated = rotate(voxel_data_tmp2_shifted, angle_of_interest * 180 / np.pi, reshape=False)

            blended_after = (voxel_data_tmp1 + voxel_data_tmp2_rotated) / 2
            plt.imshow(blended_after, cmap='gray')
            number_tmp = potential_peak_height_tmp / (10**1)
            plt.title(f'Hight of Peak: {number_tmp}')
            plt.axis('image')

            plt.figure(77)
            plt.imshow(blended_after, cmap='gray')
            plt.axis('image')

            plt.pause(0.01)
            print(transformation_tmpsi)
            print(eigenvalues)
            print(eigenvectors)
            print(correlation_matrix)
            print(angle_of_interest / np.pi * 180)
            print("test")

        last_number_of_points += current_translation_number
        print("next")

    dimension_tmp = 511
    a_val = 31
    b_val = 31
    tmp_calculation = 0
    if a_val < np.ceil(511 / 2):
        tmp_calculation = dimension_tmp * dimension_tmp * a_val
    else:
        tmp_calculation = dimension_tmp * dimension_tmp * (dimension_tmp - a_val + 1)

    if b_val < np.ceil(511 / 2):
        tmp_calculation = tmp_calculation * b_val
    else:
        tmp_calculation = tmp_calculation * (dimension_tmp - b_val + 1)

    print(tmp_calculation)
    print((256 * 256 * 511 * 511) / tmp_calculation)

    our_test_array = correlation_matrix_shift2d
    normalized_array = our_test_array / our_test_array.max()
    normalized_array = gaussian_filter(normalized_array, sigma=1)

    selem = disk(5)
    normalized_array = tophat(normalized_array, selem)
    normalized_array = extended_max(normalized_array, 10)

    p = fast_2d_peak_find(normalized_array, 0.1)

    z_for_plot = np.zeros(len(p[0::2]))
    for k in range(len(p[0::2])):
        print(k)
        z_for_plot[k] = normalized_array[p[2 * k], p[2 * k - 1]]

    plt.figure(9)
    plt.clf()
    plt.hold(True)
    ax = plt.subplot(111, projection='3d')
    ax.plot3D(p[0::2], p[1::2], z_for_plot, 'r+')
    x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
    ax.plot_surface(x_plot, y_plot, normalized_array, edgecolor='none')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    plt.figure(9)
    plt.clf()
    plt.hold(True)

    im = correlation_matrix_shift2d
    im = im / im.max()
    im1 = im
    selem = disk(np.ceil(0.02 * result_size))
    im2 = tophat(im1, selem)
    im3 = extended_max(im2, 0.03)

    props = measure.regionprops(im3.astype(int))

    x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x_plot, y_plot, im, edgecolor='none')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    plt.imshow(im2)

    for i, prop in enumerate(props):
        x = np.ceil(prop.centroid)
        tmp = np.zeros_like(im)
        tmp[prop.coords[:, 0], prop.coords[:, 1]] = 1
        tmp2 = tmp * im2

        ref_v, b_idx = tmp2.max(), tmp2.argmax()
        x2, y2 = np.unravel_index(b_idx, im.shape)

        tmp = ndimage.binary_dilation(im2 > ref_v * 0.6, structure=np.ones((3, 3)), origin=0)
        tmp = ndimage.binary_fill_holes(tmp)
        labels = measure.label(tmp)
        target_label = labels[y2, x2]
        tmp = labels == target_label

        xi, yi = np.where(tmp)
        zi = im[yi, xi]
        index_max_z = np.argmax(zi)

        ax = plt.subplot(111, projection='3d')
        ax.plot3D([yi[index_max_z]], [xi[index_max_z]], [zi[index_max_z]], 'r.')
        plt.text(y2 + 10, x2, str(i), color='white', fontsize=16)

    plt.figure(10)
    x_plot, y_plot = np.meshgrid(np.arange(1, result_size + 1), np.arange(1, result_size + 1))
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x_plot, y_plot, im2, edgecolor='none')
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")

    plt.figure(12)
    plt.imshow(im3)


if __name__ == "__main__":
    plot_all_solutions_in_registration()
