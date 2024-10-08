import numpy as np


def compute_error(img_1, img_2):
    error_matrix = img_1 != img_2

    error = sum(sum(error_matrix.astype(np.uint8)))

    return error
