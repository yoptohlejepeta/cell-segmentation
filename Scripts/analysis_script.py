import matplotlib.pyplot as plt
import mahotas as mh

import datetime
import os
import cv2


# Moje scripty
import convert_worker as cw
import image_worker as iw
import shape_descriptors as sd


def get_names_from_directory(base_path):
    images = []

    for entry in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, entry)):
            images.append(entry)

    return images


def create_directories_for_results(path, N, list_of_input_data, note):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if note is None:
        path = f"{path}Analysis_{time}/"
    else:
        path = f"{path}Analysis_{time}_{note}/"

    os.mkdir(path)

    for i in range(N):
        n_path = f"{path}{list_of_input_data[i]}/"
        os.mkdir(n_path)

        path_images = f"{n_path}IMG/"
        path_graphs = f"{n_path}GRAPHS/"
        path_csv = f"{n_path}CSV_TXT/"

        os.mkdir(path_images)
        os.mkdir(path_graphs)
        os.mkdir(path_csv)

    return path


def analysis(data_path, output_path, note=None):
    print("Analysis just started")

    try:
        list_of_input_data = get_names_from_directory(data_path)
        print("Success with input path")
    except Exception as e:
        print("Something wrong with input path | ", e)
        return

    N = len(list_of_input_data)

    try:
        default_output_path = create_directories_for_results(
            output_path, N, list_of_input_data, note
        )
        print("Success with output path")
    except:
        print("Something wrong with output path")
        return

    for i in range(N):
        print(f"Analysis of {list_of_input_data[i]} just started")
        output_path = default_output_path + f"{list_of_input_data[i]}/"

        try:
            input_data = data_path + list_of_input_data[i]
            img = cv2.imread(input_data)
            img = iw.BGR_to_RGB(img)
            print("Success with input data")
        except:
            print("Something wrong with input data")
            continue

        process_1(img, output_path)

    print("Analysis just finished")

    # ------------------------------------------------------------------------------------------------------------------


def process_1(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_Input_data.png", img)

    # ------------------ Předem určený proměnný ---------------------

    sigma = 10
    W = [0.05, 0.9, 0.05]
    mask_size = 3
    iterations = 3
    min_size = 200

    # ------------------ Zde kód pro analýzu ------------------------

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}IMG/01_Unsharp.png", img_unsharp)

    img_grayscale = cw.convert_RGB_to_grayscale(img_unsharp, width, height, W)
    plt.imsave(f"{output_path}IMG/02_Grayscale.png", img_grayscale, cmap="gray")

    img_bin = cw.convert_grayscale_to_bin(img_grayscale, less_than=False)
    plt.imsave(f"{output_path}IMG/03_Bin.png", img_bin, cmap="gray")

    img_bin_remove_noise = iw.remove_noise(
        img_bin, mask_size=mask_size, iterations=iterations
    )
    plt.imsave(
        f"{output_path}IMG/04_Bin_remove_noise.png", img_bin_remove_noise, cmap="gray"
    )

    img_labeled_removed_small_regions = iw.remove_small_regions(
        img_bin_remove_noise, min_size=min_size, is_bin=True
    )
    plt.imsave(
        f"{output_path}IMG/05_Labeled_removed_small_regions.png",
        img_labeled_removed_small_regions,
        cmap="jet",
    )

    img_bin_removed_small_regions = cw.convert_labeled_to_bin(
        img_labeled_removed_small_regions, background=0
    )
    plt.imsave(
        f"{output_path}IMG/06_Bin_removed_small_regions.png",
        img_bin_removed_small_regions,
        cmap="gray",
    )

    img_grayscale_gauss_smooth = mh.gaussian_filter(img_grayscale.astype(float), sigma)
    plt.imsave(
        f"{output_path}IMG/07_Grayscale_gauss_smooth.png",
        img_grayscale_gauss_smooth,
        cmap="gray",
    )

    img_grayscale_gauss_smooth_stretch = mh.stretch(img_grayscale_gauss_smooth)
    plt.imsave(
        f"{output_path}IMG/08_Grayscale_gauss_smooth_stretch.png",
        img_grayscale_gauss_smooth_stretch,
        cmap="gray",
    )

    img_regional_max = mh.regmax(img_grayscale_gauss_smooth_stretch)
    plt.imsave(f"{output_path}IMG/09_Regional_max.png", img_regional_max, cmap="gray")

    img_regional_max_labeled, nr_object = mh.label(img_regional_max)
    plt.imsave(
        f"{output_path}IMG/10_Regional_max_labeled.png",
        img_regional_max_labeled,
        cmap="jet",
    )

    img_distance_transform = 255 - mh.stretch(
        mh.distance(img_bin_removed_small_regions)
    )
    plt.imsave(
        f"{output_path}IMG/11_Distance_transform.png",
        img_distance_transform,
        cmap="gray",
    )

    img_watershed = mh.cwatershed(img_distance_transform, img_regional_max_labeled)
    plt.imsave(f"{output_path}IMG/12_Watershed.png", img_watershed, cmap="jet")

    img_watershed_mask = img_watershed * img_bin_removed_small_regions
    plt.imsave(
        f"{output_path}IMG/13_Watershed_mask.png", img_watershed_mask, cmap="jet"
    )

    img_watershed_mask_removed_small_regions = iw.remove_small_regions(
        img_watershed_mask, min_size=min_size, is_bin=False
    )
    plt.imsave(
        f"{output_path}IMG/14_Watershed_removed_small_regions.png",
        img_watershed_mask_removed_small_regions,
        cmap="jet",
    )

    img_watershed_mask_removed_small_regions_and_bordering = (
        mh.labeled.remove_bordering(img_watershed_mask_removed_small_regions)
    )
    plt.imsave(
        f"{output_path}IMG/15_Watershed_removed_small_regions_and_bordering.png",
        img_watershed_mask_removed_small_regions_and_bordering,
        cmap="jet",
    )

    img_watershed_mask_removed_small_regions_and_bordering_relabeled, _ = (
        mh.labeled.relabel(img_watershed_mask_removed_small_regions_and_bordering)
    )
    plt.imsave(
        f"{output_path}IMG/16_Watershed_removed_small_regions_and_bordering_relabeled.png",
        img_watershed_mask_removed_small_regions_and_bordering_relabeled,
        cmap="jet",
    )

    cell_sizes = mh.labeled.labeled_size(
        img_watershed_mask_removed_small_regions_and_bordering_relabeled
    )
    cell_sizes[0] = 0
    number_of_cells = cell_sizes.shape[0]
    matrix_coordinates_of_cells = sd.get_coordinates_of_pixels(
        img_watershed_mask_removed_small_regions_and_bordering_relabeled,
        cell_sizes,
        number_of_cells,
        width,
        height,
    )

    img_final = iw.cell_repair(
        matrix_coordinates_of_cells, cell_sizes, number_of_cells, width, height
    )
    plt.imsave(f"{output_path}IMG/17_Watershed_final.png", img_final, cmap="jet")

    img_final_boundary = sd.get_boundary_4_connected(img_final, width, height)
    plt.imsave(
        f"{output_path}IMG/18_Boundary_label.png", img_final_boundary, cmap="jet"
    )

    img_final_boundary_bin = cw.convert_labeled_to_bin(img_final_boundary)
    plt.imsave(
        f"{output_path}IMG/19_Boundary_bin.png", img_final_boundary_bin, cmap="gray"
    )

    img_original_with_boundary = iw.boundary_to_original_image(
        img, img_final_boundary_bin, width, height
    )
    plt.imsave(
        f"{output_path}IMG/20_Boundary_in_original_image.png",
        img_original_with_boundary,
    )

    # -------------------- Shape Descriptors ------------------------
    #'''
    descriptor_mask = [True, True, True, False, True, False, True, True, True]

    sd.analysis(img_final, width, height, output_path, descriptor_mask)
    #'''

    # ----------------------- Histograms ---------------------------

    # vw.histogram_image(img_grayscale,'Stupně šedi','Stupeň šedi', 'Četnost', 'grayscale_hist',output_path)

    # ------------------ Končí kód pro analýzu ----------------------


if __name__ == "__main__":
    for f in ["00_all_images"]:
        t0 = datetime.datetime.now()

        DATA_PATH = f"../Images/{f}/"
        OUTPUT_PATH = "../Results/"
        NOTE = f"{f}"

        analysis(DATA_PATH, OUTPUT_PATH, NOTE)

        t1 = datetime.datetime.now()
        print(str(t1 - t0))
