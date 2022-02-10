import numpy as np


def convert_grayscale_to_bin(img, threshold_value = None,less_than = True):

    if threshold_value is None: threshold_value = img.mean()

    if less_than:
        img_bin = img < threshold_value
    else:
        img_bin = img > threshold_value

    return img_bin


def convert_labeled_to_bin(img, background = 0):

    img_bin = img != background

    return img_bin


def convert_RGB_to_grayscale(img, width, height, W=None):

    if W is None:
        W = [1 / 3, 1 / 3, 1 / 3]

    img_grayscale = np.zeros((height, width))

    # Zjistím hodnoty RGB udělám a udělám vážený průměr

    for i in range(height):
        for j in range(width):
            avg = img[i][j][0] * W[0] + img[i][j][1] * W[1] + img[i][j][2] * W[2]
            img_grayscale[i][j] = int(avg)

    return img_grayscale


def separate_layers(img):

    L0 = img[:,:,0]
    L1 = img[:,:,1]
    L2 = img[:,:,2]

    return L0, L1, L2


if __name__ == "__main__":

    print('Hello, home!')
