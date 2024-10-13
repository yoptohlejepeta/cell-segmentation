import math

import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.morphology import convex_hull_image


def get_compactness(cell_sizes, perimeter, number_of_cells, exclude_first_index=True):
    info_compactness = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_compactness[i] = (4 * math.pi * cell_sizes[i]) / ((perimeter[i]) ** 2)

    return info_compactness


def get_rectangularity(
    major_axis_length, minor_axis_length, cell_sizes, number_of_cells, exclude_first_index=True
):
    info_rectangularity = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_rectangularity[i] = cell_sizes[i] / (major_axis_length[i] * minor_axis_length[i])

    return info_rectangularity


def get_eccentricity(
    major_axis_length, minor_axis_length, number_of_cells, exclude_first_index=True
):
    info_eccentricity = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_eccentricity[i] = minor_axis_length[i] / major_axis_length[i]

    return info_eccentricity


def get_elongation(
    bounding_box_height, bounding_box_width, number_of_cells, exclude_first_index=True
):
    info_elongation = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_elongation[i] = bounding_box_width[i] / bounding_box_height[i]

    return info_elongation


def get_roundness(convex_hull_perimeter, cell_sizes, number_of_cells, exclude_first_index=True):
    info_roundness = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_roundness[i] = (4 * math.pi * cell_sizes[i]) / (convex_hull_perimeter[i]) ** 2

    return info_roundness


def get_convexity(convex_hull_perimeter, perimeter, number_of_cells, exclude_first_index=True):
    info_convexity = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_convexity[i] = convex_hull_perimeter[i] / perimeter[i]

    return info_convexity


def get_solidity(convex_hull_sizes, cell_sizes, number_of_cells, exclude_first_index=True):
    info_solidity = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        info_solidity[i] = cell_sizes[i] / convex_hull_sizes[i]

    return info_solidity


def get_curl(major_axis_length, perimeter, cell_sizes, number_of_cells, exclude_first_index=True):
    info_curl = np.zeros(number_of_cells)
    fibre_length = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        fibre_length[i] = (perimeter[i] - (perimeter[i] ** 2 - 16 * cell_sizes[i]) ** (1 / 2)) / (4)
        info_curl[i] = major_axis_length[i] / fibre_length[i]

    return info_curl, fibre_length


def get_sphericity(
    coordinates_of_boundary_pixels,
    boundary_sizes,
    centroids,
    number_of_cells,
    exclude_first_index=True,
):
    info_sphericity = np.zeros(number_of_cells)

    r_inner = np.zeros(number_of_cells)
    r_outer = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    # Doufám, že to bude stačit
    distance_max = 2**30

    r_inner.fill(distance_max)

    for i in range(start_index, number_of_cells):
        for k in range(0, int(boundary_sizes[i] * 2), 2):
            r = (
                (centroids[i][0] - coordinates_of_boundary_pixels[i][k]) ** 2
                + (centroids[i][1] - coordinates_of_boundary_pixels[i][k + 1]) ** 2
            ) ** (1 / 2)

            r_outer[i] = max(r, r_outer[i])

            r_inner[i] = min(r, r_inner[i])

    for i in range(start_index, number_of_cells):
        info_sphericity[i] = r_inner[i] / r_outer[i]

    return info_sphericity


def get_major_axis_vector(
    coordinates_of_boundary_pixels,
    boundary_sizes,
    number_of_cells,
    width,
    height,
    exclude_first_index=True,
):
    major_axis_vector = np.zeros((number_of_cells, 2), dtype=int)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    # Taková kontrola, budu si ukládat počáteční a koncový body
    points_of_major_axis = np.zeros((number_of_cells, 4), dtype=int)

    for i in range(start_index, number_of_cells):
        distance_max = 0

        for k in range(0, int(boundary_sizes[i] * 2), 2):
            for l in range(k + 2, int(boundary_sizes[i] * 2), 2):
                x1 = coordinates_of_boundary_pixels[i][k]
                y1 = coordinates_of_boundary_pixels[i][k + 1]

                x2 = coordinates_of_boundary_pixels[i][l]
                y2 = coordinates_of_boundary_pixels[i][l + 1]

                distance = (x2 - x1) ** 2 + (y2 - y1) ** 2

                if distance > distance_max:
                    distance_max = distance
                    major_axis_vector[i][0] = x2 - x1
                    major_axis_vector[i][1] = y1 - y2

                    points_of_major_axis[i][0] = x1
                    points_of_major_axis[i][1] = y1
                    points_of_major_axis[i][2] = x2
                    points_of_major_axis[i][3] = y2

        if major_axis_vector[i][0] < 0:
            major_axis_vector[i][0] = major_axis_vector[i][0] * (-1)
            major_axis_vector[i][1] = major_axis_vector[i][1] * (-1)

    # '''
    # ----------------------------------------------------------------------- #
    img_major_axis_vector = Image.new("L", (width, height), "black")
    d_bmp = ImageDraw.Draw(img_major_axis_vector)

    for i in range(start_index, number_of_cells):
        x1 = points_of_major_axis[i][0]
        y1 = points_of_major_axis[i][1]

        x2 = points_of_major_axis[i][2]
        y2 = points_of_major_axis[i][3]

        d_bmp.line((x1, y1, x2, y2), fill="white")
    # ----------------------------------------------------------------------- #
    # '''

    return major_axis_vector, img_major_axis_vector


def get_major_axis_angle(major_axis_vector, number_of_cells, exclude_first_index=True):
    major_axis_angle = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        ratio = major_axis_vector[i][1] / major_axis_vector[i][0]

        if major_axis_vector[i][1] < 0:
            major_axis_angle[i] = 2 * math.pi + np.arctan(ratio)
        else:
            major_axis_angle[i] = np.arctan(ratio)

    return major_axis_angle


def get_major_axis_length(major_axis_vector, number_of_cells, exclude_first_index=True):
    major_axis_length = np.zeros(number_of_cells)

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        major_axis_length[i] = int(
            (major_axis_vector[i][0] ** 2 + major_axis_vector[i][1] ** 2) ** (1 / 2)
        )

    return major_axis_length


def get_coordinates_of_rotated_cells(
    coordinates_of_boundary_pixels,
    major_axis_angle,
    boundary_sizes,
    number_of_cells,
    exclude_first_index=True,
):
    rotated_coordinates = np.zeros(
        (coordinates_of_boundary_pixels.shape[0], coordinates_of_boundary_pixels.shape[1]),
        dtype=int,
    )

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        for j in range(0, int(boundary_sizes[i] * 2), 2):
            x = coordinates_of_boundary_pixels[i][j]
            y = coordinates_of_boundary_pixels[i][j + 1]

            alpha = major_axis_angle[i]

            rotated_coordinates[i][j] = int(x * math.cos(alpha) - y * math.sin(alpha))
            rotated_coordinates[i][j + 1] = int(x * math.sin(alpha) + y * math.cos(alpha))

    # '''
    # ---------------------------------------------------------------------------------------------------------------- #
    minimum = np.amin(rotated_coordinates)
    maximum = np.amax(rotated_coordinates)

    shape_of_new_img = maximum - minimum + 1
    shift = 0 - minimum

    img_rotated_cells = np.zeros((shape_of_new_img, shape_of_new_img))

    for i in range(start_index, number_of_cells):
        for j in range(0, int(boundary_sizes[i] * 2), 2):
            x = rotated_coordinates[i][j] + shift
            y = rotated_coordinates[i][j + 1] + shift

            img_rotated_cells[y][x] = i

    # ---------------------------------------------------------------------------------------------------------------- #
    # '''

    return rotated_coordinates, img_rotated_cells


def get_minor_axis_length(
    rotated_coordinates_of_boundary_pixels,
    boundary_sizes,
    number_of_cells,
    exclude_first_index=True,
):
    minor_axis_length = np.zeros(
        number_of_cells
    )  # jen  vzdálenost nic víc pro analýzu tvarů nepotřebuji

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        distance_max = 0

        for k in range(0, int(boundary_sizes[i] * 2), 2):
            for l in range(0, int(boundary_sizes[i] * 2), 2):
                if (
                    rotated_coordinates_of_boundary_pixels[i][k]
                    == rotated_coordinates_of_boundary_pixels[i][l]
                ):
                    current_distance = abs(
                        rotated_coordinates_of_boundary_pixels[i][k + 1]
                        - rotated_coordinates_of_boundary_pixels[i][l + 1]
                    )

                    if current_distance > distance_max:
                        distance_max = current_distance
                        minor_axis_length[i] = current_distance

    return minor_axis_length


def get_convex_hull_info(
    coordinates_of_pixels, cell_sizes, number_of_cells, width, height, exclude_first_index=True
):
    convex_hull_area = np.zeros((number_of_cells), dtype=int)
    convex_hull_perimeter = np.zeros((number_of_cells), dtype=int)
    img_boundary = np.zeros((height, width))

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    single_cell = np.zeros((height, width))

    for i in range(start_index, number_of_cells):
        single_cell.fill(0)

        for j in range(0, int(cell_sizes[i] * 2), 2):
            x = coordinates_of_pixels[i][j]
            y = coordinates_of_pixels[i][j + 1]

            single_cell[y][x] = 1

        img_convex_hull_area = convex_hull_image(single_cell)
        convex_hull_area[i] = sum(sum(img_convex_hull_area))

        img_convex_hull_boundary = get_boundary_4_connected(img_convex_hull_area, width, height)
        convex_hull_perimeter[i] = sum(sum(img_convex_hull_boundary))

        # ---------------------------------------------------------------------#
        for q in range(height):
            for w in range(width):
                if img_convex_hull_boundary[q][w] != 0:
                    img_boundary[q][w] = i
        # ---------------------------------------------------------------------#

    return convex_hull_area, convex_hull_perimeter, img_boundary


def get_boundary_4_connected(img_labeled, width, height):
    img_boundary = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            if img_labeled[i][j] != 0:
                value = img_labeled[i][j]

                if i < height - 1 and j < width - 1:
                    if img_labeled[i + 1][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0 and j < width - 1:
                    if img_labeled[i - 1][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i < height - 1 and j > 0:
                    if img_labeled[i + 1][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0 and j > 0:
                    if img_labeled[i - 1][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if j < width - 1:
                    if img_labeled[i][j + 1] != value:
                        img_boundary[i][j] = value
                        continue

                if j > 0:
                    if img_labeled[i][j - 1] != value:
                        img_boundary[i][j] = value
                        continue

                if i < height - 1:
                    if img_labeled[i + 1][j] != value:
                        img_boundary[i][j] = value
                        continue

                if i > 0:
                    if img_labeled[i - 1][j] != value:
                        img_boundary[i][j] = value
                        continue

    return img_boundary


def get_centroids(
    coordinates_of_boundary_pixels, boundary_sizes, number_of_cells, exclude_first_index=True
):
    centroids = np.zeros((number_of_cells, 2))

    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    for i in range(start_index, number_of_cells):
        for j in range(0, int(boundary_sizes[i] * 2), 2):
            centroids[i][0] += coordinates_of_boundary_pixels[i][j]

        for j in range(1, int(boundary_sizes[i] * 2), 2):
            centroids[i][1] += coordinates_of_boundary_pixels[i][j]

        centroids[i][0] = int(centroids[i][0] / boundary_sizes[i])
        centroids[i][1] = int(centroids[i][1] / boundary_sizes[i])

    return centroids


def get_coordinates_of_pixels(img_labeled, cell_sizes, number_of_cells, width, height):
    # Informace o potřebných rozměrech matice
    matrix_height = number_of_cells
    matrix_width = int(np.amax(cell_sizes) * 2)

    # Matice souřadnic pixelů jenotlivých buněk
    matrix_coordinates = np.zeros((matrix_height, matrix_width), dtype=int)

    # Matice pro uložení počtu souřadnic které jsem již použil
    matrix_shifts = np.zeros(number_of_cells)

    for i in range(height):
        for j in range(width):
            if img_labeled[i][j] == 0:
                continue

            cell_index = int(img_labeled[i][j])
            shift = int(matrix_shifts[cell_index])

            matrix_coordinates[cell_index][shift] = j
            matrix_coordinates[cell_index][shift + 1] = i

            matrix_shifts[cell_index] += 2

    return matrix_coordinates


def convert_2D_array_to_1D_list(array):
    list_result = []

    for i in range(array.shape[0]):
        line = f"({array[i][0]} , {array[i][1]})"
        list_result.append(line)

    return list_result


def analysis(img_labeled, width, height, output_path, descriptor_mask, exclude_first_index=True):
    cell_sizes = mh.labeled.labeled_size(img_labeled)
    cell_sizes[0] = 0

    number_of_cells = cell_sizes.shape[0]

    img_labeled_boundary = get_boundary_4_connected(img_labeled, width, height)
    boundary_sizes = mh.labeled.labeled_size(img_labeled_boundary)
    boundary_sizes[0] = 0

    coordinates_cells = get_coordinates_of_pixels(
        img_labeled, cell_sizes, number_of_cells, width, height
    )
    coordinates_boundary = get_coordinates_of_pixels(
        img_labeled_boundary, boundary_sizes, number_of_cells, width, height
    )

    major_axis_vector, img_of_vectors = get_major_axis_vector(
        coordinates_boundary, boundary_sizes, number_of_cells, width, height, exclude_first_index
    )
    major_axis_angle = get_major_axis_angle(major_axis_vector, number_of_cells, exclude_first_index)
    major_axis_length = get_major_axis_length(
        major_axis_vector, number_of_cells, exclude_first_index
    )

    coordinates_rotated_boundary, img_rotated_cells = get_coordinates_of_rotated_cells(
        coordinates_boundary, major_axis_angle, boundary_sizes, number_of_cells, exclude_first_index
    )

    minor_axis_length = get_minor_axis_length(
        coordinates_rotated_boundary, boundary_sizes, number_of_cells, exclude_first_index
    )

    convex_hull_area, convex_hull_perimeter, img_boundary = get_convex_hull_info(
        coordinates_cells, cell_sizes, number_of_cells, width, height, exclude_first_index
    )

    centroids = get_centroids(
        coordinates_boundary, boundary_sizes, number_of_cells, exclude_first_index
    )

    # Nastavení pro ukládání
    if exclude_first_index:
        start_index = 1
    else:
        start_index = 0

    id_cell = {"Cell id": np.arange(start_index, number_of_cells)}

    # DataFrame Info
    df_info = pd.DataFrame(id_cell)

    df_info["Area"] = cell_sizes[start_index:]
    df_info["Perimeter"] = boundary_sizes[start_index:]
    df_info["Convex hull area"] = convex_hull_area[start_index:]
    df_info["Convex hull perimeter"] = convex_hull_perimeter[start_index:]
    df_info["Major axis vector"] = convert_2D_array_to_1D_list(major_axis_vector)[start_index:]
    df_info["Major axis angle (rad)"] = major_axis_angle[start_index:]
    df_info["Major axis length"] = major_axis_length[start_index:]
    df_info["Minor axis length"] = minor_axis_length[start_index:]
    df_info["Centroids"] = convert_2D_array_to_1D_list(centroids)[start_index:]

    # DataFrame shape descriptors
    df_shape_des = pd.DataFrame(id_cell)

    if descriptor_mask[0] == 1:
        compactness = get_compactness(
            cell_sizes, boundary_sizes, number_of_cells, exclude_first_index
        )
        df_shape_des["Compactness"] = compactness[start_index:]

    if descriptor_mask[1] == 1:
        rectangularity = get_rectangularity(
            major_axis_length, minor_axis_length, cell_sizes, number_of_cells, exclude_first_index
        )
        df_shape_des["Rectangularity"] = rectangularity[start_index:]

    if descriptor_mask[2] == 1:
        eccentricity = get_eccentricity(
            major_axis_length, minor_axis_length, number_of_cells, exclude_first_index
        )
        df_shape_des["Eccentricity"] = eccentricity[start_index:]

    if descriptor_mask[3] == 1:
        elongation = get_elongation()
        df_shape_des["Elongation"] = elongation[start_index:]

    if descriptor_mask[4] == 1:
        roundness = get_roundness(
            convex_hull_perimeter, cell_sizes, number_of_cells, exclude_first_index
        )
        df_shape_des["Roundness"] = roundness[start_index:]

    if descriptor_mask[5] == 1:
        convexity = get_convexity(
            convex_hull_perimeter, boundary_sizes, number_of_cells, exclude_first_index
        )
        df_shape_des["Convexity"] = convexity[start_index:]

    if descriptor_mask[6] == 1:
        solidity = get_solidity(convex_hull_area, cell_sizes, number_of_cells, exclude_first_index)
        df_shape_des["Solidity"] = solidity[start_index:]

    if descriptor_mask[7] == 1:
        curl, fibre_length = get_curl(
            major_axis_length, boundary_sizes, cell_sizes, number_of_cells, exclude_first_index
        )
        df_shape_des["Curl"] = curl[start_index:]
        df_info["Fibre length"] = fibre_length[start_index:]

    if descriptor_mask[8] == 1:
        sphericity = get_sphericity(
            coordinates_boundary, boundary_sizes, centroids, number_of_cells, exclude_first_index
        )
        df_shape_des["Sphericity"] = sphericity[start_index:]

    # df_info.to_csv(f"{output_path}CSV_TXT/Information.csv", index=False, header=True)
    # df_shape_des.to_csv(f"{output_path}CSV_TXT/Shape descriptors.csv", index=False, header=True)

    plt.imsave(f"{output_path}51_Rotated_cells.jpg", img_rotated_cells, cmap="jet")
    img_of_vectors.save(f"{output_path}50_Major_axis.jpg")
