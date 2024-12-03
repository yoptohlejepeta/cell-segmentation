import os
import time
from pathlib import Path

import colorcorrect.algorithm as cca
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from skimage import filters

import cellseg.src.convert_worker as cw
import cellseg.src.image_worker as iw
import cellseg.src.shape_descriptors as sd


def get_names_from_directory(base_path):
    images = []

    for entry in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, entry)):
            images.append(entry)

    return images


def create_directories_for_results(path: str, list_of_input_data: list[str]):
    for i in list_of_input_data:
        n_path = f"{path}{i}/"
        Path(n_path).mkdir(parents=True, exist_ok=True)

    return path


def check_dirs(data_path: str, output_path: str) -> tuple[list[str], str]:
    """Check the input and output paths.

    Args:
    ----
        data_path (str): Directory with images to be processed.
        output_path (str): Directory for the output

    Returns:
    -------
        tuple[list[str], str]: list of images, output_path

    """
    list_of_input_data = get_names_from_directory(data_path)
    mod_output = f"{output_path}output_images/"
    try:
        default_output_path = create_directories_for_results(mod_output, list_of_input_data)
    except Exception:
        logger.warning("Missing output path -> Creating default one")
        Path("Results/").mkdir(parents=True, exist_ok=True)
        Path("Results/output_images/").mkdir(parents=True, exist_ok=True)
        default_output_path = "Results/output_images/"

    return list_of_input_data, default_output_path


def img_processing_3(img: np.ndarray, output_path: str, steps: bool) -> None:
    """Metoda bude využívat knihovnu PIL na zlepšení hledání jader.

    Metoda bude využívat knihovnu Colorcorret na zredukování nežádoucího osvětlení na snímku.
    Na hledání cytoplazmy se použije green channel ( zvláštní ale funguje to )

    jádra - červeně
    cytoplazmu - zeleně

    :param img: (np.array) snímek, který se analyzuje
    :param output_path: (string) cesta kam se budou ukládat výsledky
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # Detekce jader

    img_unsharp = iw.unsharp_mask_img(img)

    r1, g1, b1 = cw.separate_layers(img_unsharp)
    if steps:
        plt.imsave(f"{output_path}02_1_red_channel_unsharp.jpg", r1, cmap="gray")
        plt.imsave(f"{output_path}02_2_green_channel_unsharp.jpg", g1, cmap="gray")
        plt.imsave(f"{output_path}02_3_blue_channel_unsharp.jpg", b1, cmap="gray")

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b1)
    plt.imsave(f"{output_path}03_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}04_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray")

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}05_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_labeled_nuclei = iw.remove_small_regions(img_labeled_nuclei, min_size=100)
    plt.imsave(f"{output_path}06_nuclei_labeled_removed_small.jpg", img_labeled_nuclei, cmap="jet")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu

    RGB_balanced = cca.luminance_weighted_gray_world(img)
    plt.imsave(f"{output_path}07_luminance_weighted_gray_world.jpg", RGB_balanced)

    r2, g2, b2 = cw.separate_layers(RGB_balanced)
    plt.imsave(f"{output_path}08_1_red_channel_balanced_luminance.jpg", r2, cmap="gray")
    plt.imsave(f"{output_path}08_2_green_channel_balanced_luminance.jpg", g2, cmap="gray")
    plt.imsave(f"{output_path}08_3_blue_channel_balanced_luminance.jpg", b2, cmap="gray")

    thresholds = filters.threshold_multiotsu(g2)
    regions = np.digitize(g2, bins=thresholds)
    plt.imsave(f"{output_path}09_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}10_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}11_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}12_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_labeled_cytoplasm = iw.remove_small_regions(img_labeled_cytoplasm, min_size=100)
    plt.imsave(
        f"{output_path}13_mul_otsu_labeled_removed_small_cyto.jpg",
        img_labeled_cytoplasm,
        cmap="jet",
    )

    # ---------------------------------------------------------------------------------------------------------------- #
    # Spojení

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_labeled_cytoplasm, img_labeled_nuclei, width, height
    )
    plt.imsave(f"{output_path}14_flooding_cytoplasm.jpg", img_cytoplasm_nuclei, cmap="jet")

    img_cytoplasm_nuclei = mh.labeled.remove_bordering(img_cytoplasm_nuclei)
    plt.imsave(
        f"{output_path}15_flooding_cytoplasm_no_bordering.jpg", img_cytoplasm_nuclei, cmap="jet"
    )

    img_labeled_nuclei = cw.convert_labeled_to_bin(img_labeled_nuclei) * img_cytoplasm_nuclei
    plt.imsave(f"{output_path}16_nuclei_no_bordering.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}17_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}18_boundary_in_original_img.jpg", img_boundary_in_original)

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(f"{output_path}19_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}20_boundary_in_original_img.jpg", img_boundary_in_original)

    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}21_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}22_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}23_flooding_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet"
    )

    img_original_boundary_cytoplasm_nuclei = iw.two_boundary_types_to_original_image(
        img, img_nuclei_boundary, img_cytoplasm_boundary, width, height
    )
    plt.imsave(f"{output_path}23_boundary_final_skoro.jpg", img_original_boundary_cytoplasm_nuclei)

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}24_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}25_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    b_bin_otsu_morp = cw.convert_labeled_to_bin(img_labeled_nuclei)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}26_only_cytoplasm_bin.jpg", img_only_cytoplasm_bin, cmap="gray")

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b2, img_only_cytoplasm_bin, width, height
    )
    plt.imsave(
        f"{output_path}27_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}28_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(
        f"{output_path}29_cytoplasm_and_nuclei_bin.jpg",
        img_cytoplasm_and_nuclei_bin,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}30_cytoplasm_and_nuclei_labeled.jpg",
        img_cytoplasm_and_nuclei_labeled,
        cmap="jet",
    )

    cytoplasm_sizes = mh.labeled.labeled_size(img_cytoplasm_and_nuclei_labeled)
    cytoplasm_sizes[0] = 0
    number_of_cells = cytoplasm_sizes.shape[0]

    coordinates_cytoplasm = sd.get_coordinates_of_pixels(
        img_cytoplasm_and_nuclei_labeled, cytoplasm_sizes, number_of_cells, width, height
    )

    img_repaired = iw.cell_repair(
        coordinates_cytoplasm, cytoplasm_sizes, number_of_cells, width, height
    )
    plt.imsave(f"{output_path}21_cytoplasm_and_nuclei_repaired.jpg", img_repaired, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cytoplasm_boundary_bin
    )
    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}32_boundary_in_original_img.jpg", img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(
        f"{output_path}33_only_cytoplasm_with_nuclei.jpg",
        img_only_cytoplasm_with_nuclei,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei), is_bin=True
    )
    plt.imsave(
        f"{output_path}34_only_cytoplasm_with_nuclei_without_small_reg.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}35_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_cytoplasm_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}36_only_cytoplasm_with_nuclei_boundary.jpg",
        boundary_img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}37_only_cytoplasm_which_have_nuclei.jpg",
        only_cytoplasm_which_have_nuclei_but_not_included,
        cmap="jet",
    )

    img_original_boundary_cytoplasm_nuclei = iw.two_boundary_types_to_original_image(
        img,
        img_nuclei_boundary,
        boundary_img_only_cytoplasm_with_nuclei_removed_small,
        width,
        height,
    )
    plt.imsave(f"{output_path}38_boundary_final.jpg", img_original_boundary_cytoplasm_nuclei)


if __name__ == "__main__":
    img = plt.imread("Images/BAL_images/2023_12_14_image_006.png")

    t1 = time.monotonic()
    img_processing_3(img, "output/", True)
    t2 = time.monotonic()
    print(f"Time: {t2-t1} s")
