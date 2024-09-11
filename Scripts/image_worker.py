import numpy as np
import mahotas as mh
from PIL import ImageFilter, Image


def unsharp_mask_img(img):
    """Unsharp masking je technika zvyšující ostrost obrázku.
    Vytvoří se kopie obrázku, která je rozostřená a následně se od původního obrázku odečte.
    Výsledný obrázek je ostřejší.
    Vedlejším efektem může být vytvoření nechtěného šumu.
    """
    RADIUS = 10
    PERCENT = 300
    THRESHOLD = 3

    img_pil = Image.fromarray(img, "RGB")

    bmp = img_pil.filter(
        ImageFilter.UnsharpMask(radius=RADIUS, percent=PERCENT, threshold=THRESHOLD)
    )

    return np.array(bmp)


def cell_repair(
    matrix_coordinates, cell_sizes, number_of_cells, width, height, n=3, m=5
):
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
                if matrix_one_cell[k][l] == True:  # noqa: E712 při záměně na `is True` se chová jinak
                    img_repair[k][l] = i

    # Odstranění pixelů co se dotýkají stěn
    for i in range(width):
        img_repair[0][i] = 0
        img_repair[height - 1][i] = 0

    for i in range(height):
        img_repair[i][0] = 0
        img_repair[i][width - 1] = 0

    return img_repair


def boundary_to_original_image(img, img_boundary, width, height, color=[255, 0, 0]):

    img_original_with_boundary = np.copy(img)

    for i in range(height):
        for j in range(width):
            if img_boundary[i][j] != 0:
                img_original_with_boundary[i][j][0] = color[0]
                img_original_with_boundary[i][j][1] = color[1]
                img_original_with_boundary[i][j][2] = color[2]

    return img_original_with_boundary


def remove_noise(img, mask_size=3, iterations=3):
    kernel = np.ones((mask_size, mask_size))

    # Eroze (zmenšení)
    for i in range(iterations):
        img = mh.erode(img, kernel)

    # Dilatace (zvětšení)
    for i in range(iterations):
        img = mh.dilate(img, kernel)

    return img


def remove_small_regions(img, min_size=200, is_bin=False):
    if is_bin:
        img, _ = mh.label(img)

    sizes = mh.labeled.labeled_size(img)

    img_without_small_regions = mh.labeled.remove_regions_where(img, sizes < min_size)

    return img_without_small_regions


def BGR_to_RGB(img):
    im = np.zeros_like(img)

    im[:, :, 0] = img[:, :, 2]
    im[:, :, 1] = img[:, :, 1]
    im[:, :, 2] = img[:, :, 0]

    return im


def boundary_with_centroids(img_boundary, centroids):
    img = np.copy(img_boundary)

    for i in range(1, centroids.shape[0]):
        x = int(centroids[i, 0])
        y = int(centroids[i, 1])

        img[y, x] = i

    return img


if __name__ == "__main__":
    print("Hello, home!")
