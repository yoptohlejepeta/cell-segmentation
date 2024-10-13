import os

import mahotas as mh
import numpy as np


def get_names_from_directory(base_path):
    images = []

    for entry in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, entry)):
            images.append(entry)

    return images


def color_cube(img, name, output_path):
    """Metoda vytvoří tvz. color cube
    R -> X
    G -> Y
    B -> Z
    Uloží výsledný soubor XYZ


    :param img: snímek
    :param name: název výsledného souboru
    :param output_path: cesta
    :param save_cube_img: jestli uložit cube_labels
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    e = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
    ]

    cube_shape = 256
    # cube = np.zeros((cube_shape, cube_shape, cube_shape))
    cube_labels = np.zeros((cube_shape, cube_shape, cube_shape))

    # all_pixels = width * height

    # for i in range(height):
    #     for j in range(width):
    #         x = img[i][j][0]
    #         y = img[i][j][1]
    #         z = img[i][j][2]

    #         cube[x, y, z] += 1

    # bin_cube = cube > 0
    # num = sum(sum(sum(bin_cube)))

    # cube = cube / all_pixels

    # mn = np.min(cube[np.nonzero(cube)])
    # mx = np.max(cube[np.nonzero(cube)])

    # d = mx - mn

    # fn = d / (len(e) - 1)

    # file = open(f"{output_path}CSV_TXT/{name}.xyz", "w")

    # file.write(str(num + 8) + "\n")

    # for x in range(cube_shape):
    #     for y in range(cube_shape):
    #         for z in range(cube_shape):
    #             if cube[x, y, z] != 0:
    #                 el = int((cube[x, y, z] - mn) / fn)
    #                 file.write(f"\n{e[el]}\t{x}\t{y}\t{z}")
    #                 cube_labels[x][y][z] = int(el + 1)

    # file.write("\nAu\t0\t0\t0")
    # file.write("\nAu\t0\t255\t0")
    # file.write("\nAu\t0\t0\t255")
    # file.write("\nAu\t0\t255\t255")
    # file.write("\nAu\t255\t0\t0")
    # file.write("\nAu\t255\t255\t0")
    # file.write("\nAu\t255\t0\t255")
    # file.write("\nAu\t255\t255\t255")

    # file.close()

    return cube_labels


def color_cube_2(img):
    """Metoda vytvoří tvz. color cube.

    R -> X
    G -> Y
    B -> Z
    Uloží výsledný soubor XYZ


    :param img: snímek
    :param name: název výsledného souboru
    :param output_path: cesta
    :param save_cube_img: jestli uložit cube_labels
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    e = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
    ]

    cube_shape = 256
    cube = np.zeros((cube_shape, cube_shape, cube_shape))
    cube_labels = np.zeros((cube_shape, cube_shape, cube_shape))

    all_pixels = width * height

    for i in range(height):
        for j in range(width):
            x = img[i][j][0]
            y = img[i][j][1]
            z = img[i][j][2]

            cube[x, y, z] += 1

    cube = cube / all_pixels

    mn = np.min(cube[np.nonzero(cube)])
    mx = np.max(cube[np.nonzero(cube)])

    d = mx - mn

    fn = d / (len(e) - 1)

    for x in range(cube_shape):
        for y in range(cube_shape):
            for z in range(cube_shape):
                if cube[x, y, z] != 0:
                    el = int((cube[x, y, z] - mn) / fn)
                    cube_labels[x][y][z] = int(el + 1)

    return cube_labels


def intersection(cube_1, cube_2):
    n = 256

    cube_intersection_1 = np.zeros((n, n, n))
    cube_intersection_2 = np.zeros((n, n, n))

    for x in range(n):
        for y in range(n):
            for z in range(n):
                if cube_1[x][y][z] > 0 and cube_2[x][y][z] > 0:
                    cube_intersection_2[x][y][z] = 1
                if (
                    cube_1[x][y][z] == cube_2[x][y][z]
                    and cube_2[x][y][z] != 0
                    and cube_1[x][y][z] != 0
                ):
                    cube_intersection_1[x][y][z] = cube_1[x][y][z]

    # TODO vyřešit

    """cube_intersection_1 = cube_1 == cube_2
    cube_intersection_1 = cube_intersection_1 * cube_1  # klidne *= cube_2

    a = cube_1 > 0
    b = cube_2 > 0

    cube_intersection_2 = (a.astype(np.uint8) + b.astype(np.uint8)) > 1"""

    return cube_intersection_1, cube_intersection_2


def save_xyz(cube, output_path, name):
    cube_shape = 256
    bin_cube = cube > 0
    num = sum(sum(sum(bin_cube)))

    e = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
    ]

    file = open(f"{output_path}{name}.xyz", "w")

    file.write(str(num + 8) + "\n")

    for x in range(cube_shape):
        for y in range(cube_shape):
            for z in range(cube_shape):
                if cube[x, y, z] != 0:
                    file.write(f"\n{e[int(cube[x, y, z]-1)]}\t{x}\t{y}\t{z}")

    file.write("\nAu\t0\t0\t0")
    file.write("\nAu\t0\t255\t0")
    file.write("\nAu\t0\t0\t255")
    file.write("\nAu\t0\t255\t255")
    file.write("\nAu\t255\t0\t0")
    file.write("\nAu\t255\t255\t0")
    file.write("\nAu\t255\t0\t255")
    file.write("\nAu\t255\t255\t255")

    file.close()


def analysis(data_path, output_path):
    images_LL = ["12a.jpg", "12b.jpg", "12c.jpg", "12d.jpg", "12e.jpg"]
    images_ML = ["1a.jpg", "1b.jpg", "1c.jpg", "1d.jpg", "1e.jpg"]

    for i in range(len(images_LL)):
        img_1 = mh.imread(f"{data_path}{images_LL[i]}")
        img_2 = mh.imread(f"{data_path}{images_ML[i]}")

        cube_1 = color_cube_2(img_1)
        cube_2 = color_cube_2(img_2)

        fn, bz = intersection(cube_1, cube_2)

        name_fn = f"{i+1}_prunik"
        name_bz = f"{i+1}_shoda"

        save_xyz(fn, output_path, name_fn)
        save_xyz(bz, output_path, name_bz)
