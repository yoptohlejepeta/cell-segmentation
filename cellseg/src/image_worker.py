import mahotas as mh
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

# Moje scripty
import cellseg.src.convert_worker as cw


def color_gradient(img, width, height):
    img_color_gradient = np.zeros((height, width, 3), dtype=float)

    for i in range(height):
        for j in range(width):
            for k in range(3):
                deviation = abs(float(img[i][j][k]) - float(img[(i + 1) % height][j][k]))
                deviation += abs(float(img[i][j][k]) - float(img[(i - 1) % height][j][k]))
                deviation += abs(float(img[i][j][k]) - float(img[i][(j + 1) % width][k]))
                deviation += abs(float(img[i][j][k]) - float(img[i][(j - 1) % width][k]))

                deviation += abs(
                    float(img[i][j][k]) - float(img[(i + 1) % height][(j + 1) % width][k])
                )
                deviation += abs(
                    float(img[i][j][k]) - float(img[(i + 1) % height][(j - 1) % width][k])
                )
                deviation += abs(
                    float(img[i][j][k]) - float(img[(i - 1) % height][(j + 1) % width][k])
                )
                deviation += abs(
                    float(img[i][j][k]) - float(img[(i - 1) % height][(j - 1) % width][k])
                )

                img_color_gradient[i][j][k] = deviation

    return img_color_gradient


def color_balancing(img, width, height):
    r, g, b = cw.separate_layers(img)

    mean_r = r.mean()
    mean_g = g.mean()
    mean_b = b.mean()

    mean_i = img.mean()

    img_color_balanced = np.zeros((height, width, 3))

    WR = mean_r / mean_i
    WG = mean_g / mean_i
    WB = mean_b / mean_i

    for i in range(height):
        for j in range(width):
            img_color_balanced[i][j][0] = int(img[i][j][0] * WR)
            img_color_balanced[i][j][1] = int(img[i][j][1] * WG)
            img_color_balanced[i][j][2] = int(img[i][j][2] * WB)

    return img_color_balanced


def unsharp_mask_img(
    img: np.ndarray,
    radius: int = 10,
    percent: int = 300,
    threshold: int = 3,
) -> np.ndarray:
    """Unsharp mask image.

    Args:
    ----
        img (np.ndarray): Image to be processed.
        output_path (str): Path to save the processed image.
        radius (int, optional): Radius of the filter. Defaults to 10.
        percent (int, optional): Percentage of the sharpening. Defaults to 300.
        threshold (int, optional): Threshold of the filter. Defaults to 3.

    Returns:
    -------
        np.ndarray: Processed image.

    """
    img_pil = Image.fromarray(img, "RGB")

    bmp = img_pil.filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )
    img_unsharp = np.array(bmp)

    return img_unsharp


def get_relabeled_image(img_labeled, width, height, exclude_first_index=True):
    if exclude_first_index:
        start_index = 1
        k = 1
    else:
        start_index = 0
        k = 0

    sizes = mh.labeled.labeled_size(img_labeled)

    number_of_labels = sizes.shape[0]

    label_array = np.zeros(number_of_labels)

    for i in range(start_index, number_of_labels):
        if sizes[i] != 0:
            label_array[i] = k
            k = k + 1

    for i in range(height):
        for j in range(width):
            index = img_labeled[i][j]
            img_labeled[i][j] = label_array[index]

    return img_labeled


def cell_repair(matrix_coordinates, cell_sizes, number_of_cells, width, height, n=3, m=5):
    img_repair = np.zeros((height, width))

    mask = np.ones((n, n))

    matrix_one_cell = np.zeros((height, width), dtype=bool)

    for i in range(1, number_of_cells):
        matrix_one_cell.fill(False)

        for j in range(0, (int(cell_sizes[i] * 2)), 2):
            x = matrix_coordinates[i][j]
            y = matrix_coordinates[i][j + 1]

            matrix_one_cell[y][x] = True

        # Dilatace
        for k in range(m):
            matrix_one_cell = mh.dilate(matrix_one_cell, mask)

        # Uzavření
        matrix_one_cell = mh.close_holes(matrix_one_cell)

        # Eroze
        for k in range(m):
            matrix_one_cell = mh.erode(matrix_one_cell, mask)

        # Převod opravené buňky do výsledné matice
        for k in range(height):
            for l in range(width):
                if matrix_one_cell[k][l] == True:
                    img_repair[k][l] = i

    # Odstranění pixelů co se dotýkají stěn
    for i in range(width):
        img_repair[0][i] = 0
        img_repair[height - 1][i] = 0

    for i in range(height):
        img_repair[i][0] = 0
        img_repair[i][width - 1] = 0

    return img_repair


def boundary_to_original_image(img, img_boundary, width, height, color=[255, 255, 255]):
    img_original_with_boundary = np.copy(img)

    for i in range(height):
        for j in range(width):
            if img_boundary[i][j] != 0:
                img_original_with_boundary[i][j][0] = color[0]
                img_original_with_boundary[i][j][1] = color[1]
                img_original_with_boundary[i][j][2] = color[2]

    return img_original_with_boundary


def two_boundary_types_to_original_image(
    img, first_boundary, second_boundary, width, height, cf=[255, 0, 0], cs=[0, 255, 0]
):
    img_original_with_boundary = np.copy(img)

    for i in range(height):
        for j in range(width):
            if second_boundary[i][j] != 0:
                img_original_with_boundary[i][j][0] = cs[0]
                img_original_with_boundary[i][j][1] = cs[1]
                img_original_with_boundary[i][j][2] = cs[2]
            if first_boundary[i][j] != 0:
                img_original_with_boundary[i][j][0] = cf[0]
                img_original_with_boundary[i][j][1] = cf[1]
                img_original_with_boundary[i][j][2] = cf[2]

    return img_original_with_boundary


def relabeling_sort(img_labeled, width, height):
    sizes = mh.labeled.labeled_size(img_labeled)

    n = sizes.shape[0]

    label_array = np.zeros(n)

    k = 1

    for i in range(1, n):
        if sizes[i] != 0:
            label_array[i] = k
            k = k + 1

    for i in range(height):
        for j in range(width):
            index = img_labeled[i][j]
            img_labeled[i][j] = label_array[index]

    return img_labeled


def relabeling_background(img_labeled, width, height):
    sizes = mh.labeled.labeled_size(img_labeled)
    background = np.where(sizes == np.amax(sizes))[0][0]

    for i in range(height):
        for j in range(width):
            if img_labeled[i][j] == background:
                img_labeled[i][j] = 0

            elif img_labeled[i][j] > background:
                img_labeled[i][j] = img_labeled[i][j] - 1

            else:
                img_labeled[i][j] = img_labeled[i][j]

    return img_labeled


def close_holes_remove_noise(img: np.ndarray, mask_size: int = 3, iterations: int = 5):
    img_bin = mh.close_holes(img)
    mask = np.ones((mask_size, mask_size))  # Občas se používá kernel

    for k in range(iterations):
        img_bin = mh.erode(img_bin, mask)

    for k in range(iterations):
        img_bin = mh.dilate(img_bin, mask)

    return img_bin


def remove_noise(img, mask_size=3, iterations=3):
    kernel = np.ones((mask_size, mask_size))  # Maska

    # Eroze (zmenšení)
    for i in range(iterations):
        img = mh.erode(img, kernel)

    # Dilatace (zvětšení)
    for i in range(iterations):
        img = mh.dilate(img, kernel)

    return img


def find_markers_in_bin_img(img_bin, iterations=4, mask_size=3):
    kernel = np.ones((mask_size, mask_size))

    for i in range(iterations):
        img_bin = mh.erode(img_bin, kernel)

    return img_bin


def find_nuclei_in_mask(img_grayscale, img_bin_mask, width, height):
    img_grayscale_mask = img_grayscale * img_bin_mask

    value = 0
    size = 0

    for i in range(height):
        for j in range(width):
            if img_bin_mask[i][j] == 1:
                value += img_grayscale[i][j]
                size += 1

    threshold = value / size

    img_bin_nuclei = img_grayscale < threshold

    return img_bin_nuclei, img_grayscale_mask


def flooding_cytoplasm(labeled_cytoplasm, labeled_nuclei, width, height):
    bin_cytoplasm = cw.convert_labeled_to_bin(labeled_cytoplasm)
    bin_nuclei = cw.convert_labeled_to_bin(labeled_nuclei)

    bin_cytoplasm_nuclei = cw.convert_labeled_to_bin(bin_cytoplasm + bin_nuclei)

    labeled_nuclei_old = np.copy(labeled_nuclei)
    labeled_nuclei_new = np.copy(labeled_nuclei)

    flag = True

    while flag:
        flag = False

        for i in range(height):
            for j in range(width):
                if bin_cytoplasm_nuclei[i][j] == 0:
                    continue
                if labeled_nuclei_old[i][j] != 0:
                    continue

                if i < height - 1 and j < width - 1:
                    if labeled_nuclei_old[i + 1][j + 1] != 0:
                        labeled_nuclei_new[i][j] = labeled_nuclei_old[i + 1][j + 1]
                        flag = True
                        continue

                if i > 0 and j < width - 1:
                    if labeled_nuclei_old[i - 1][j + 1] != 0:
                        labeled_nuclei_new[i][j] = labeled_nuclei_old[i - 1][j + 1]
                        flag = True
                        continue

                if i < height - 1 and j > 0:
                    if labeled_nuclei_old[i + 1][j - 1] != 0:
                        labeled_nuclei_new[i][j] = labeled_nuclei_old[i + 1][j - 1]
                        flag = True
                        continue

                if i > 0 and j > 0 and labeled_nuclei_old[i - 1][j - 1] != 0:
                    labeled_nuclei_new[i][j] = labeled_nuclei_old[i - 1][j - 1]
                    flag = True
                    continue

                if j < width - 1 and labeled_nuclei_old[i][j + 1] != 0:
                    labeled_nuclei_new[i][j] = labeled_nuclei_old[i][j + 1]
                    flag = True
                    continue

                if j > 0 and labeled_nuclei_old[i][j - 1] != 0:
                    labeled_nuclei_new[i][j] = labeled_nuclei_old[i][j - 1]
                    flag = True
                    continue

                if i < height - 1 and labeled_nuclei_old[i + 1][j] != 0:
                    labeled_nuclei_new[i][j] = labeled_nuclei_old[i + 1][j]
                    flag = True
                    continue

                if i > 0 and labeled_nuclei_old[i - 1][j] != 0:
                    labeled_nuclei_new[i][j] = labeled_nuclei_old[i - 1][j]
                    flag = True
                    continue

        labeled_nuclei_old = np.copy(labeled_nuclei_new)

    #  zde zapojuji i ty cytoplasmy kde není jádro
    #  trošičku problém s labely ale snad pohoda

    labeled_cytoplasm_nuclei = bin_cytoplasm_nuclei - labeled_nuclei_new
    bin_cytoplasm_without_nuclei = labeled_cytoplasm_nuclei > 0
    labeled_cytoplasm_without_nuclei, _ = mh.label(bin_cytoplasm_without_nuclei)

    last_label_of_nuclei = np.amax(labeled_nuclei_new)

    img_nuclei_cytoplasm = np.copy(labeled_nuclei_new)

    for i in range(height):
        for j in range(width):
            if labeled_cytoplasm_without_nuclei[i][j] > 0:
                img_nuclei_cytoplasm[i][j] = (
                    labeled_cytoplasm_without_nuclei[i][j] + last_label_of_nuclei
                )

    return img_nuclei_cytoplasm


def get_average_values_of_nuclei(
    img, coordinates_of_nuclei, nuclei_sizes, number_of_nuclei, exclude_first_index=True
):
    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    average_values = np.zeros(number_of_nuclei)

    for i in range(start_index, number_of_nuclei):
        for j in range(0, int(2 * nuclei_sizes[i]), 2):
            x = coordinates_of_nuclei[i][j]
            y = coordinates_of_nuclei[i][j + 1]

            average_values[i] = average_values[i] + img[y, x]

        average_values[i] = average_values[i] / nuclei_sizes[i]

    return average_values


def check_cytoplasm_by_average_value_of_nuclei(
    img,
    coordinates_of_cytoplasm,
    cytoplasm_sizes,
    average_values_of_nuclei,
    number_of_nuclei,
    number_of_cytoplasm,
    width,
    height,
    deviation=0.2,
    exclude_first_index=True,
):
    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    img_labeled = np.zeros((height, width))

    for i in range(start_index, number_of_cytoplasm):
        if i < number_of_nuclei:
            left = average_values_of_nuclei[i] - deviation
            right = average_values_of_nuclei[i] + deviation

        for j in range(0, int(2 * cytoplasm_sizes[i]), 2):
            x = coordinates_of_cytoplasm[i][j]
            y = coordinates_of_cytoplasm[i][j + 1]

            if i >= number_of_nuclei or left < img[y][x] < right:
                img_labeled[y][x] = i

    return img_labeled


def get_cytoplasm_only(img_nuclei_bin, img_cytoplasm_labeled):
    img_cytoplasm_nuclei_bin = cw.convert_labeled_to_bin(img_cytoplasm_labeled)

    img_only_cytoplasm = img_cytoplasm_nuclei_bin.astype(int) - img_nuclei_bin.astype(int)

    return img_cytoplasm_labeled * img_only_cytoplasm


def threshold_in_mask(img_grayscale, img_bin_mask, width, height, threshold_value=-1):
    img_grayscale_mask = img_grayscale * img_bin_mask

    if threshold_value == -1:
        value = 0
        size = 0

        for i in range(height):
            for j in range(width):
                if img_bin_mask[i][j] == 1:
                    value += img_grayscale[i][j]
                    size += 1

        threshold_value = value / size

    # print(str(threshold_value))
    img_bin = (img_grayscale < threshold_value) * img_bin_mask

    return img_bin, img_grayscale_mask


def remove_small_regions(img, min_size=200, is_bin=False):
    if is_bin:
        img, _ = mh.label(img)

    sizes = mh.labeled.labeled_size(img)

    img_without_small_regions = mh.labeled.remove_regions_where(img, sizes < min_size)

    return img_without_small_regions


def get_cytoplasm_which_have_nuclei(img, number_of_nuclei):
    mask = img < number_of_nuclei

    return img * mask


def get_remove_nuclei_from_cytoplasm(img_cytoplasm, img_nuclei, width, height):
    for i in range(height):
        for j in range(width):
            if img_nuclei[i][j] != 0:
                img_cytoplasm[i][j] = 0

    return img_cytoplasm


def reverse_3_layers_array(img):
    L0, L1, L2 = cw.separate_layers(img)

    img_new = np.copy(img)

    img_new[:, :, 0] = L2
    img_new[:, :, 1] = L1
    img_new[:, :, 2] = L0

    return img_new


def mean_in_mask(img_grayscale, img_mask):
    img_grayscale_mask = img_grayscale * img_mask

    mean = np.sum(img_grayscale_mask) / np.sum(img_mask)

    return mean


def fft_filter_circle(r, width, height):
    im = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    center_x = int(width / 2)
    center_y = int(height / 2)

    r = center_x * r

    upper_x = center_x - r
    upper_y = center_y - r

    lower_x = center_x + r
    lower_y = center_y + r

    draw.ellipse((upper_x, upper_y, lower_x, lower_y), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.rectangle((center_x, 0, width, height), fill=(0, 0, 0), outline=(0, 0, 0))

    img = np.array(im)
    r, g, b = cw.separate_layers(img)
    mask = cw.convert_labeled_to_bin(r)

    return mask


def fft_filter_rectangle(width_r, height_r, width_img, height_img):
    im = Image.new("RGB", (width_img, height_img), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    center_x = int(width_img / 2)
    center_y = int(height_img / 2)

    width_r = center_x * width_r
    height_r = center_y * height_r

    upper_x = center_x - width_r
    upper_y = center_y - height_r

    lower_x = center_x + width_r
    lower_y = center_y + height_r

    draw.rectangle((upper_x, upper_y, lower_x, lower_y), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.rectangle((0, center_y, width_img, height_img), fill=(0, 0, 0), outline=(0, 0, 0))

    img = np.array(im)
    r, g, b = cw.separate_layers(img)
    mask = cw.convert_labeled_to_bin(r)

    return mask


def fill_boundaries(img_bin):
    img_bin[ndimage.binary_fill_holes(img_bin)] = 1

    return img_bin
