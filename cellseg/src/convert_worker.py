import math

import mahotas as mh
import numpy as np

# Dictionary
color_extremes_min = {
    "HSL_A_H": 0.0,
    "HSL_A_S": 0.0,
    "HSL_A_L": 0.0,
    "HSL_N_H": 0.0,
    "HSL_N_S": 0.0,
    "HSL_N_L": 0.0,
    "RGB_R": 0.0,
    "RGB_G": 0.0,
    "RGB_B": 0.0,
    "XYZ_X": 0.0,
    "XYZ_Y": 0.0,
    "XYZ_Z": 0.0,
    "Luv_L": 0.0,
    "Luv_u": -1782.1012792369656,
    "Luv_v": -4009.727878283173,
}


# Dictionary
color_extremes_max = {
    "HSL_A_H": 6.2797824700042595,
    "HSL_A_S": 1.0,
    "HSL_A_L": 765.0,
    "HSL_N_H": 6.279263738552135,
    "HSL_N_S": 1.0,
    "HSL_N_L": 1.0,
    "RGB_R": 255,
    "RGB_G": 255,
    "RGB_B": 255,
    "XYZ_X": 1438.659,
    "XYZ_Y": 1440.954,
    "XYZ_Z": 1570.6215,
    "Luv_L": 1294.2113628197244,
    "Luv_u": 3845.6811355835653,
    "Luv_v": 8652.78255506302,
}


def convert_grayscale_to_bin(img, threshold_value=-1, less_than=True):
    if threshold_value == -1:
        threshold_value = img.mean()

    if less_than:
        img_bin = img < threshold_value
    else:
        img_bin = img > threshold_value

    return img_bin


def convert_grayscale_to_bin_otsu(img):
    img = img.astype(np.uint8)
    threshold_value = mh.otsu(img)
    img_bin = img < threshold_value

    return img_bin


def convert_labeled_to_bin(img, background=0):
    img_bin = img != background

    return img_bin


def convert_RGB_to_grayscale(img, width, height, W=[1 / 3, 1 / 3, 1 / 3]):
    img_grayscale = np.zeros((height, width))

    # Zjistím hodnoty RGB udělám a udělám vážený průměr

    for i in range(height):
        for j in range(width):
            avg = img[i][j][0] * W[0] + img[i][j][1] * W[1] + img[i][j][2] * W[2]
            img_grayscale[i][j] = int(avg)

    return img_grayscale


def convert_RGB_to_HSL_A(img, width, height):
    # Segmentation of cytological image using color and mathematical morphology ---> article

    r, g, b = separate_layers(img)

    img_HSL = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            R = int(r[i][j])
            G = int(g[i][j])
            B = int(b[i][j])

            min_value = float(np.amin([R, G, B]))

            numerator = (R - G) + (R - B)
            denominator = 2 * (((R - G) ** 2 + (R - B) * (G - B)) ** (1 / 2))

            if R == G == B:
                img_HSL[i][j][0] = 0
            else:
                img_HSL[i][j][0] = math.acos(numerator / denominator)

            if G < B:
                img_HSL[i][j][0] = (2 * math.pi) - img_HSL[i][j][0]

            L = R + G + B

            if L == 0:
                continue

            img_HSL[i][j][2] = L
            img_HSL[i][j][1] = 1 - ((3 * min_value) / L)

    return img_HSL


def convert_RGB_to_HSL_N(img, width, height):
    # http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/

    r, g, b = separate_layers(img)

    img_HSL = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            R = r[i][j] / 255.0
            G = g[i][j] / 255.0
            B = b[i][j] / 255.0

            if R == G == B == 0:
                continue

            # layers in 3D matrix
            # H = 0
            # S = 1
            # L = 2

            RGB = [R, G, B]

            min_value = float(np.amin(RGB))
            max_value = float(np.amax(RGB))

            img_HSL[i][j][2] = (min_value + max_value) / 2

            if min_value == max_value:
                continue

            if img_HSL[i][j][2] < 0.5:
                img_HSL[i][j][1] = (max_value - min_value) / (max_value + min_value)
            else:
                img_HSL[i][j][1] = (max_value - min_value) / (2.0 - max_value - min_value)

            if max_value == R:
                img_HSL[i][j][0] = (G - B) / (max_value - min_value)
            elif max_value == G:
                img_HSL[i][j][0] = 2.0 + ((B - R) / (max_value - min_value))
            elif max_value == B:
                img_HSL[i][j][0] = 4.0 + ((R - G) / (max_value - min_value))

            if img_HSL[i][j][0] < 0:
                img_HSL[i][j][0] = img_HSL[i][j][0] + 2 * math.pi

    return img_HSL


def convert_RGB_to_XYZ(img, width, height):
    r, g, b = separate_layers(img)

    img_XYZ = np.zeros((height, width, 3))

    # Matice z článku
    R0 = np.asarray([2.76, 1.7518, 1.13])
    R1 = np.asarray([1.0, 4.5907, 0.0601])
    R2 = np.asarray([0.0, 0.565, 5.5943])

    for i in range(height):
        for j in range(width):
            vector_RGB = np.asarray([r[i][j], g[i][j], b[i][j]])

            X = sum(R0 * vector_RGB)
            Y = sum(R1 * vector_RGB)
            Z = sum(R2 * vector_RGB)

            img_XYZ[i][j][0] = X
            img_XYZ[i][j][1] = Y
            img_XYZ[i][j][2] = Z

    return img_XYZ


def convert_XYZ_to_Luv(img, width, height):
    x, y, z = separate_layers(img)

    img_Luv = np.zeros((height, width, 3))

    X0 = 1
    Y0 = 1
    Z0 = 1

    u0 = (4 * X0) / (X0 + 15 * Y0 + 3 * Z0)
    v0 = (9 * X0) / (X0 + 15 * Y0 + 3 * Z0)

    for i in range(height):
        for j in range(width):
            X = x[i][j]
            Y = y[i][j]
            Z = z[i][j]

            if X == Y == Z == 0:
                continue

            u_ = (4 * X) / (X + 15 * Y + 3 * Z)
            v_ = (9 * X) / (X + 15 * Y + 3 * Z)

            if (Y / Y0) > 0.008856:
                L = (116 * ((Y / Y0) ** (1 / 3))) - 16
            else:
                L = 903.3 * (Y / Y0)

            u = 13 * L * (u_ - u0)
            v = 13 * L * (v_ - v0)

            img_Luv[i][j][0] = L
            img_Luv[i][j][1] = u
            img_Luv[i][j][2] = v

    return img_Luv


def convert_grayscale_to_bin_by_range(img, lower, upper):
    img_bin1 = img > lower
    img_bin2 = img < upper

    img_bin = np.logical_and(img_bin1, img_bin2)

    return img_bin


def convert_2D_to_3D(img, width, height):
    img_3D = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            img_3D[i][j][0] = img[i][j]
            img_3D[i][j][1] = img[i][j]
            img_3D[i][j][2] = img[i][j]

    return img_3D


def separate_layers(img):
    L0 = img[:, :, 0]
    L1 = img[:, :, 1]
    L2 = img[:, :, 2]

    return L0, L1, L2


def convert_img_to_norm_img(img, color_system_key):
    minimum = color_extremes_min[color_system_key]
    maximum = color_extremes_max[color_system_key]

    img_norm = (img - minimum) / (maximum - minimum)

    return img_norm


def convert_to_bin_by_horizontal_threshold(img, threshold_freq, width, height, bins=100):
    freq = np.zeros(bins)
    divisor = 1 / bins
    img_labeled = np.zeros((height, width))
    img_bin = np.ones((height, width))

    for i in range(height):
        for j in range(width):
            index = int(img[i, j] / divisor) - 1
            freq[index] += 1
            img_labeled[i, j] = index

    for i in range(height):
        for j in range(width):
            index = int(img_labeled[i, j])
            if freq[index] > threshold_freq:
                img_bin[i, j] = 0

    return img_bin
