import numpy as np
import math


def integral(x, y):
    if len(x) < 2:
        return y[0]

    h = x[1] - x[0]

    return h * sum(y)


def compute_ratio_blue_brown(blue_number, brown_number):
    ratio = blue_number / (brown_number + blue_number)

    return ratio


def compute_ratio_mask_img(color_number, width, height):
    ratio = color_number / (width * height)

    return ratio


if __name__ == "__main__":
    print("Hello home")
