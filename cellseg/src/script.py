import os
from pathlib import Path

import colorcorrect.algorithm as cca
import cv2
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from skimage import filters

import cellseg.src.convert_worker as cw
import cellseg.src.image_worker as iw
import cellseg.src.shape_descriptors as sd
import cellseg.src.something as s

# import src.visual_worker as vw


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


def img_processing_2(img, output_path):
    """Metoda bude využívat knihovnu PIL na zlepšení hledání jader.

    Metoda bude využívat knihovnu Colorcorret na zredukování nežádoucího osvětlení na snímku.
    Na hledání cytoplazmy se použije green channel ( zvláštní ale funguje to ). # TODO: :D

    jádra - červeně
    cytoplazmu - zeleně

    :param img: (np.array) snimek, který se analyzuje
    :param output_path: (string) cesta kam se budou ukládat výsledky
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # Detekce jader

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}01_unsharp_mask.jpg", img_unsharp)

    r1, g1, b1 = cw.separate_layers(img_unsharp)
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
    plt.imsave(f"{output_path}05_nuclei_labeled_removed_small.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}06_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}07_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu

    RGB_balanced = cca.luminance_weighted_gray_world(img)
    # RGB_balanced = img
    plt.imsave(f"{output_path}08_luminance_weighted_gray_world.jpg", RGB_balanced)

    r2, g2, b2 = cw.separate_layers(RGB_balanced)
    plt.imsave(f"{output_path}09_1_red_channel_balanced_luminance.jpg", r2, cmap="gray")
    plt.imsave(f"{output_path}09_2_green_channel_balanced_luminance.jpg", g2, cmap="gray")
    plt.imsave(f"{output_path}09_3_blue_channel_balanced_luminance.jpg", b2, cmap="gray")

    thresholds = filters.threshold_multiotsu(g2)
    regions = np.digitize(g2, bins=thresholds)
    plt.imsave(f"{output_path}10_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}11_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}12_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}13_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_labeled_cytoplasm = iw.remove_small_regions(img_labeled_cytoplasm, min_size=100)
    plt.imsave(
        f"{output_path}13_mul_otsu_labeled_removed_small_cyto.jpg",
        img_labeled_cytoplasm,
        cmap="jet",
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(f"{output_path}14_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}15_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Spojení

    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}16_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}17_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_labeled_cytoplasm, img_labeled_nuclei, width, height
    )
    plt.imsave(f"{output_path}18_flooding_cytoplasm.jpg", img_cytoplasm_nuclei, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}19_flooding_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet"
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}20_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}21_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}22_only_cytoplasm_bin.jpg", img_only_cytoplasm_bin, cmap="gray")

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b2, img_only_cytoplasm_bin, width, height
    )
    plt.imsave(
        f"{output_path}23_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}24_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(
        f"{output_path}25_cytoplasm_and_nuclei_bin.jpg",
        img_cytoplasm_and_nuclei_bin,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}26_cytoplasm_and_nuclei_labeled.jpg",
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
    plt.imsave(f"{output_path}27_cytoplasm_and_nuclei_repaired.jpg", img_repaired, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cytoplasm_boundary_bin
    )
    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}28_boundary_in_original_img.jpg", img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(
        f"{output_path}29_only_cytoplasm_with_nuclei.jpg",
        img_only_cytoplasm_with_nuclei,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei), is_bin=True
    )
    plt.imsave(
        f"{output_path}30_only_cytoplasm_with_nuclei_without_small_reg.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}31_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_cytoplasm_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}32_only_cytoplasm_with_nuclei_boundary.jpg",
        boundary_img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}33_only_cytoplasm_which_have_nuclei.jpg",
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
    plt.imsave(f"{output_path}34_boundary_final.jpg", img_original_boundary_cytoplasm_nuclei)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Histogramy
    # vw.histogram_2D_data_range(
    #     r1, "Red channel", "Value", "Frequency", "01_Red_pil", output_path, 0, 255, 1, txt_file=True
    # )
    # vw.histogram_2D_data_range(
    #     g1,
    #     "Green channel",
    #     "Value",
    #     "Frequency",
    #     "02_Green_pil",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )
    # vw.histogram_2D_data_range(
    #     b1,
    #     "Blue channel",
    #     "Value",
    #     "Frequency",
    #     "03_Blue_pil",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )

    # vw.histogram_2D_data_range(
    #     r2, "Red channel", "Value", "Frequency", "04_Red_cca", output_path, 0, 255, 1, txt_file=True
    # )
    # vw.histogram_2D_data_range(
    #     g2,
    #     "Green channel",
    #     "Value",
    #     "Frequency",
    #     "05_Green_cca",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )
    # vw.histogram_2D_data_range(
    #     b2,
    #     "Blue channel",
    #     "Value",
    #     "Frequency",
    #     "06_Blue_cca",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )


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

    img_unsharp = iw.unsharp_mask_img(img, output_path, steps=steps)

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

    # ---------------------------------------------------------------------------------------------------------------- #
    # Histogramy
    # vw.histogram_2D_data_range(
    #     r1, "Red channel", "Value", "Frequency", "01_Red_pil", output_path, 0, 255, 1, txt_file=True
    # )
    # vw.histogram_2D_data_range(
    #     g1,
    #     "Green channel",
    #     "Value",
    #     "Frequency",
    #     "02_Green_pil",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )
    # vw.histogram_2D_data_range(
    #     b1,
    #     "Blue channel",
    #     "Value",
    #     "Frequency",
    #     "03_Blue_pil",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )

    # vw.histogram_2D_data_range(
    #     r2, "Red channel", "Value", "Frequency", "04_Red_cca", output_path, 0, 255, 1, txt_file=True
    # )
    # vw.histogram_2D_data_range(
    #     g2,
    #     "Green channel",
    #     "Value",
    #     "Frequency",
    #     "05_Green_cca",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )
    # vw.histogram_2D_data_range(
    #     b2,
    #     "Blue channel",
    #     "Value",
    #     "Frequency",
    #     "06_Blue_cca",
    #     output_path,
    #     0,
    #     255,
    #     1,
    #     txt_file=True,
    # )


def img_processing_2_part(img, output_path):
    """Na hledání jader blue channel
    na hledání cytoplazmy se použije green channel ( zvláštní ale funguje to )

    jádra - červeně
    cytoplazmu - zeleně

    :param img: (np.array) snimek, který se analyzuje
    :param output_path: (string) cesta kam se budou ukládat výsledky
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # Detekce jader

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}01_unsharp_mask.jpg", img_unsharp)

    r1, g1, b1 = cw.separate_layers(img)
    plt.imsave(f"{output_path}02_1_red_channel_unsharp.jpg", r1, cmap="gray")
    plt.imsave(f"{output_path}02_2_green_channel_unsharp.jpg", g1, cmap="gray")
    plt.imsave(f"{output_path}02_3_blue_channel_unsharp.jpg", b1, cmap="gray")

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b1)
    plt.imsave(f"{output_path}03_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}04_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray")

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}05_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}06_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}07_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu

    r2, g2, b2 = cw.separate_layers(img)
    plt.imsave(f"{output_path}09_1_red_channel_balanced_luminance.jpg", r2, cmap="gray")
    plt.imsave(f"{output_path}09_2_green_channel_balanced_luminance.jpg", g2, cmap="gray")
    plt.imsave(f"{output_path}09_3_blue_channel_balanced_luminance.jpg", b2, cmap="gray")

    thresholds = filters.threshold_multiotsu(g2)
    regions = np.digitize(g2, bins=thresholds)
    plt.imsave(f"{output_path}10_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}11_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}12_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}13_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(f"{output_path}14_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}15_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #


def color_balancing(img, output_path):
    """Metoda kterou vužívám nemazat je dobrá
    vezme snímek u udělá na něm všechny metody na color balancing atd který zatím znám

    funguje dobře pro tuto verzi numpy
    pip install numpy==1.18.4

    :param img: snímek
    :param output_path: cesta kam se uloží
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    # Original image
    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # CV2
    wb = cv2.xphoto.createGrayworldWB()
    createGrayworldWB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}01_cv2_createGrayworldWB.jpg", createGrayworldWB)

    wb = cv2.xphoto.createSimpleWB()
    createSimpleWB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}02_cv2_createSimpleWB.jpg", createSimpleWB)

    # Article
    article_balance = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}03_article_balance.jpg", article_balance)

    # PIL
    unsharp_mask = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}04_PIL_unsharp_mask.jpg", unsharp_mask)

    # CCA
    max_white = cca.max_white(img)
    plt.imsave(f"{output_path}05_cca_max_white.jpg", max_white)

    retinex = cca.retinex(img)
    plt.imsave(f"{output_path}06_cca_retinex.jpg", retinex)

    automatic_color_equalization = cca.automatic_color_equalization(img)
    plt.imsave(
        f"{output_path}07_cca_automatic_color_equalization.jpg", automatic_color_equalization
    )

    luminance_weighted_gray_world = cca.luminance_weighted_gray_world(img)
    plt.imsave(
        f"{output_path}08_cca_luminance_weighted_gray_world.jpg", luminance_weighted_gray_world
    )

    standard_deviation_weighted_grey_world = cca.standard_deviation_weighted_grey_world(img)
    plt.imsave(
        f"{output_path}09_cca_standard_deviation_weighted_grey_world.jpg",
        standard_deviation_weighted_grey_world,
    )

    standard_deviation_and_luminance_weighted_gray_world = (
        cca.standard_deviation_and_luminance_weighted_gray_world(img)
    )
    plt.imsave(
        f"{output_path}10_cca_standard_deviation_and_luminance_weighted_gray_world.jpg",
        standard_deviation_and_luminance_weighted_gray_world,
    )


def cytoplasm_RGB_channels(img, output_path):
    """Hledání vhodného kanálu na hledání cytoplazmy

    cytoplazmu - zeleně

    :param img: (np.array) snimek, který se analyzuje
    :param output_path: (string) cesta kam se budou ukládat výsledky
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # Detekce obalu

    r2, g2, b2 = cw.separate_layers(img)

    # ---------------- red ----------------
    plt.imsave(f"{output_path}02_red_0_channel_.jpg", r2, cmap="gray")

    thresholds = filters.threshold_multiotsu(r2)
    regions = np.digitize(r2, bins=thresholds)
    plt.imsave(f"{output_path}02_red_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}03_red_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}04_red_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}05_red_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}06_red_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}07_red_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------- green ----------------
    plt.imsave(f"{output_path}08_green_0_channel_.jpg", g2, cmap="gray")

    thresholds = filters.threshold_multiotsu(g2)
    regions = np.digitize(g2, bins=thresholds)
    plt.imsave(f"{output_path}08_green_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}09_green_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}10_green_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}11_green_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}12_green_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}13_green_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------- blue ----------------
    plt.imsave(f"{output_path}14_blue_0_channel_.jpg", b2, cmap="gray")

    thresholds = filters.threshold_multiotsu(b2)
    regions = np.digitize(b2, bins=thresholds)
    plt.imsave(f"{output_path}14_blue_multi_otsu_regions.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}15_blue_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}16_blue_mul_otsu_bin_morp.jpg", img_bin_morp, cmap="gray")

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(img_bin_morp)
    plt.imsave(f"{output_path}17_blue_mul_otsu_labeled.jpg", img_labeled_cytoplasm, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}18_blue_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [0, 255, 0]
    )
    plt.imsave(f"{output_path}19_blue_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #


def nuclei_RGB_channels(img, output_path):
    """Metoda zkouší knihovnu PIL na úpravu snímku a následně otsu na binarizaci
    binarizace se provásí na všechny 3 kanály (R,G,B)

    z výsledků jde vidět že blue channel je nejlepší pro identifikaci jader

    :param img: snímke
    :param output_path: cesta
    :return: None
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # Detekce jader

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}01_unsharp_mask.jpg", img_unsharp)

    r, g, b = cw.separate_layers(img_unsharp)

    # red --------------------------------------------------------------------------------------------------------------

    plt.imsave(f"{output_path}02_red_channel_unsharp.jpg", r, cmap="gray")

    r_bin_otsu = cw.convert_grayscale_to_bin_otsu(r)
    plt.imsave(f"{output_path}03_red_channel_otsu.jpg", r_bin_otsu, cmap="gray")

    r_bin_otsu_morp = iw.close_holes_remove_noise(r_bin_otsu)
    plt.imsave(f"{output_path}04_red_channel_otsu_noise_removed.jpg", r_bin_otsu_morp, cmap="gray")

    img_labeled_nuclei, nr_nuclei = mh.label(r_bin_otsu_morp)
    plt.imsave(f"{output_path}05_red_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}06_red_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}07_red_boundary_in_original_img.jpg", img_boundary_in_original)

    # green ------------------------------------------------------------------------------------------------------------

    plt.imsave(f"{output_path}08_green_channel_unsharp.jpg", g, cmap="gray")

    g_bin_otsu = cw.convert_grayscale_to_bin_otsu(g)
    plt.imsave(f"{output_path}09_green_channel_otsu.jpg", g_bin_otsu, cmap="gray")

    g_bin_otsu_morp = iw.close_holes_remove_noise(g_bin_otsu)
    plt.imsave(
        f"{output_path}10_green_channel_otsu_noise_removed.jpg", g_bin_otsu_morp, cmap="gray"
    )

    img_labeled_nuclei, nr_nuclei = mh.label(g_bin_otsu_morp)
    plt.imsave(f"{output_path}11_green_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}12_green_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}13_green_boundary_in_original_img.jpg", img_boundary_in_original)

    # blue -------------------------------------------------------------------------------------------------------------

    plt.imsave(f"{output_path}14_blue_channel_unsharp.jpg", b, cmap="gray")

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}15_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}16_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray")

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}17_blue_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}18_blue_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}19_blue_boundary_in_original_img.jpg", img_boundary_in_original)


# def rel_representation_in_xyz(img, output_path):
#     """Metoda vygeneruje soubor *.xyz
#     R -> x
#     G -> y
#     B -> z
#     a pak uloží snímky podle podmínek pouze vodík a ostatní
#     :param img:
#     :param output_path:
#     :return:
#     """
#     width = img.shape[1]
#     height = img.shape[0]

#     plt.imsave(f"{output_path}00_input_img.jpg", img)

#     # ------------------ Zde kód pro analýzu ------------------------
#     e = [
#         "H",
#         "He",
#         "Li",
#         "Be",
#         "B",
#         "C",
#         "N",
#         "O",
#         "F",
#         "Ne",
#         "Na",
#         "Mg",
#         "Al",
#         "Si",
#         "P",
#         "S",
#         "Cl",
#         "Ar",
#         "K",
#         "Ca",
#         "Sc",
#         "Ti",
#         "V",
#         "Cr",
#         "Mn",
#         "Fe",
#         "Co",
#         "Ni",
#         "Cu",
#         "Zn",
#         "Ga",
#         "Ge",
#         "As",
#         "Se",
#         "Br",
#         "Kr",
#         "Rb",
#         "Sr",
#         "Y",
#         "Zr",
#         "Nb",
#         "Mo",
#         "Tc",
#         "Ru",
#         "Rh",
#         "Pd",
#         "Ag",
#         "Cd",
#         "In",
#         "Sn",
#     ]

#     cube_shape = 256
#     cube = np.zeros((cube_shape, cube_shape, cube_shape))
#     cube_labels = np.zeros((cube_shape, cube_shape, cube_shape))

#     all_pixels = width * height

#     for i in range(height):
#         for j in range(width):
#             x = img[i][j][0]
#             y = img[i][j][1]
#             z = img[i][j][2]

#             cube[x, y, z] += 1

#     bin_cube = cube > 0
#     num = sum(sum(sum(bin_cube)))

#     cube = cube / all_pixels

#     mn = np.min(cube[np.nonzero(cube)])
#     mx = np.max(cube[np.nonzero(cube)])

#     d = mx - mn

#     fn = d / (len(e) - 1)

#     file = open(f"{output_path}CSV_TXT/soubor.xyz", "w")

#     file.write(str(num + 8) + "\n")

#     for x in range(cube_shape):
#         for y in range(cube_shape):
#             for z in range(cube_shape):
#                 if cube[x, y, z] != 0:
#                     el = int((cube[x, y, z] - mn) / fn)
#                     file.write(f"\n{e[el]}\t{x}\t{y}\t{z}")
#                     cube_labels[x][y][z] = int(el + 1)

#     file.write("\nAu\t0\t0\t0")
#     file.write("\nAu\t0\t255\t0")
#     file.write("\nAu\t0\t0\t255")
#     file.write("\nAu\t0\t255\t255")
#     file.write("\nAu\t255\t0\t0")
#     file.write("\nAu\t255\t255\t0")
#     file.write("\nAu\t255\t0\t255")
#     file.write("\nAu\t255\t255\t255")

#     file.close()

#     # Pouze vodík

#     bin_h = np.zeros((height, width))
#     bin_other_el = np.zeros((height, width))

#     for i in range(height):
#         for j in range(width):
#             x = img[i][j][0]
#             y = img[i][j][1]
#             z = img[i][j][2]

#             if cube_labels[x][y][z] == 1:
#                 bin_h[i, j] = 1

#             if cube_labels[x][y][z] > 1:
#                 bin_other_el[i, j] = 1

#     plt.imsave(f"{output_path}01_only_h.jpg", bin_h, cmap="gray")
#     plt.imsave(f"{output_path}02_other_el.jpg", bin_other_el, cmap="gray")


def color_cube_in_otsu_mask(img, output_path):
    """Metoda vytvoří color_cube pouze v masce otsu bohužel jak mám asi spojit 3D a 2D ?

    Metoda vygeneruje soubor *.xyz
    R -> x
    G -> y
    B -> z

    :param img:
    :param output_path:
    :return:
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    _ = s.color_cube(img, "color_cube", output_path)

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}01_unsharp_mask.jpg", img_unsharp)

    r1, g1, b1 = cw.separate_layers(img_unsharp)

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b1)
    plt.imsave(f"{output_path}02_otsu_blue_channel.jpg", b_bin_otsu, cmap="gray")

    img_XYZ_mask = np.copy(img)

    img_XYZ_mask[:, :, 2] = img_XYZ_mask[:, :, 2] * b_bin_otsu
    plt.imsave(f"{output_path}03_weird_img.jpg", img_XYZ_mask)

    _ = s.color_cube(img_XYZ_mask, "color_cube_mask", output_path)


def nucleus(img, output_path):
    """:param img:
    :param output_path:
    :return:
    """
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    # static variables
    min_size = 100
    mask_size = 3
    iterations = 3

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}01_unsharp_mask.jpg", img_unsharp)

    r1, g1, b1 = cw.separate_layers(img_unsharp)
    plt.imsave(f"{output_path}02_1_red_channel_unsharp.jpg", r1, cmap="gray")
    plt.imsave(f"{output_path}02_2_green_channel_unsharp.jpg", g1, cmap="gray")
    plt.imsave(f"{output_path}02_3_blue_channel_unsharp.jpg", b1, cmap="gray")

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b1)
    plt.imsave(f"{output_path}03_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    # TODO
    # Rozepsat ať můžu vidět i mezi kroky
    # Nebudu používat už ty funkce co mám protože nemusí být vždy ideální
    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}04_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray")

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}05_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_labeled_nuclei = iw.remove_small_regions(img_labeled_nuclei, min_size=min_size)
    plt.imsave(f"{output_path}06_nuclei_labeled_removed_small.jpg", img_labeled_nuclei, cmap="jet")

    # vw.array_2d_to_txt(img_labeled_nuclei, width, height, output_path, "labeled_img")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}17_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}18_boundary_in_original_img.jpg", img_boundary_in_original)
