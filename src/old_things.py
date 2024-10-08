import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
from matplotlib.font_manager import FontProperties
import cv2
import math
from skimage import filters
from matplotlib.colors import LogNorm
import colorcorrect.algorithm as cca
import hdbscan

# Moje scripty
import convert_worker as cw
import visual_worker as vw
import image_worker as iw
import shape_descriptors as sd
import coef_worker as cow
import comparator as com


def analysis(data_path, output_path, note=""):
    print("Analysis just started")

    try:
        list_of_input_data = get_names_from_directory(data_path)
    except:
        print("Something wrong with input path")
        return

    N = len(list_of_input_data)

    try:
        default_output_path = create_directories_for_results(
            output_path, N, list_of_input_data, note
        )
    except:
        print("Something wrong with output path")
        return

    for i in range(N):
        # nastavení výstupní cesty pro daný obrázek
        output_path = default_output_path + f"{list_of_input_data[i]}/"

        # Zkusím načíst snímek
        try:
            input_data = data_path + list_of_input_data[i]
            img = mh.imread(input_data)
        except:
            print("Something wrong with input data or input path")
            continue

        # Zde volám nějakou funkci co chce obrázek a outputpath nic jinýho zbytek volá ona
        img_processing(img, output_path)

    print("Analysis just finished")

    # ------------------------------------------------------------------------------------------------------------------
    """
    print('Post analysis just started')

    list_of_csv = ['01_Red','02_Green','03_Blue','04_Hue','05_Saturation','06_Luminance']
    #list_of_csv = ['03_Blue', '04_Hue', '06_Luminance']

    post_analysis_1(default_output_path, list_of_input_data,N,list_of_csv)

    print('Post analysis just finished')
    #"""


def analysis_1(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    # ------------------ Zde kód pro analýzu ------------------------

    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    r, g, b = cw.separate_layers(RGB_balanced, width, height)
    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_RGB_B_otsu.jpg", b_bin_otsu, cmap="gray")

    # otsu na HSL je na nic
    HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(HSL, width, height)
    h_bin_otsu = cw.convert_grayscale_to_bin_otsu(h)
    plt.imsave(f"{output_path}IMG/03_HSL_H_otsu.jpg", h_bin_otsu, cmap="gray")

    threshold_angle_s = 0.62 - 0.1
    threshold_angle_f = 0.62 + 0.1

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")

    vw.histogram_2D_data(
        h_norm, "Hue", "Value", "Frequency", "01_Hue", f"{output_path}GRAPHS/", bins=100
    )

    img_h_bin = cw.convert_HSL_to_bin_by_angle(h_norm, threshold_angle_s, threshold_angle_f)
    plt.imsave(f"{output_path}IMG/04_HSL_h_bin.jpg", img_h_bin, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}IMG/05_RGB_B_OTSU_MORP.jpg", b_bin_otsu_morp, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(img_h_bin)
    plt.imsave(f"{output_path}IMG/06_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    img_b_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(
        f"{output_path}IMG/07_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )
    plt.imsave(f"{output_path}IMG/08_RGB_B_labeled_nuclei.jpg", img_b_labeled_nuclei, cmap="jet")

    # tak tady začínám vymýšlet ten postup

    # Hrany

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_nuclei_boundary = sd.get_boundary_4_connected(img_b_labeled_nuclei, width, height)
    plt.imsave(f"{output_path}IMG/09_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet")
    plt.imsave(f"{output_path}IMG/10_nuclei_boundary.jpg", img_nuclei_boundary, cmap="jet")

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)

    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)

    plt.imsave(
        f"{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    # velikosti
    cytoplasm_sizes = mh.labeled.labeled_size(img_h_labeled_cytoplasm)
    nuclei_sizes = mh.labeled.labeled_size(img_b_labeled_nuclei)

    cytoplasm_sizes[0] = 0  # pozadí nastavím na nulu
    nuclei_sizes[0] = 0  # pozadí nastavím na nulu

    number_of_cytoplasm = cytoplasm_sizes.shape[0]
    number_of_nuclei = nuclei_sizes.shape[0]

    perimeter_cytoplasm = mh.labeled.labeled_size(img_cytoplasm_boundary)
    perimeter_nuclei = mh.labeled.labeled_size(img_nuclei_boundary)

    perimeter_cytoplasm[0] = 0  # pozadí nastavím na nulu
    perimeter_nuclei[0] = 0  # pozadí nastavím na nulu

    coordinates_nuclei_boundary = sd.get_coordinates_of_pixels(
        img_nuclei_boundary, perimeter_nuclei, number_of_nuclei, width, height
    )
    centroids_nuclei = sd.get_centroids(
        coordinates_nuclei_boundary, perimeter_nuclei, number_of_nuclei
    )

    # tato část není důležitá
    """# tedka zjištuji kolik má cytoplasma jader
    cytoplasm_nuclei = np.zeros((number_of_cytoplasm))
    for i in range(number_of_nuclei):
        x = int(centroids_nuclei[i][0])
        y = int(centroids_nuclei[i][1])

        index = img_h_labeled_cytoplasm[y][x]
        cytoplasm_nuclei[index] += 1

    # print(cytoplasm_nuclei)"""

    img_separated_cytoplasm = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_b_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/13_castecne_rozdelena_cytoplazma.jpg",
        img_separated_cytoplasm,
        cmap="jet",
    )

    img_separated_cytoplasm_boundary = sd.get_boundary_4_connected(
        img_separated_cytoplasm, width, height
    )
    plt.imsave(
        f"{output_path}IMG/14_castecne_rozdelena_cytoplazma_hranice.jpg",
        img_separated_cytoplasm_boundary,
        cmap="jet",
    )

    img_separated_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(
        img_separated_cytoplasm_boundary
    )

    cytoplasm_separated_nuclei_boundary = (
        img_separated_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    )
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    b_norm = cw.convert_img_to_norm_img(b, "RGB_B")

    coordinates_nuclei = sd.get_coordinates_of_pixels(
        img_b_labeled_nuclei, nuclei_sizes, number_of_nuclei, width, height
    )
    cytoplasm_sizes = mh.labeled.labeled_size(img_separated_cytoplasm)
    number_of_cytoplasm = cytoplasm_sizes.shape[0]

    average_values_of_nuclei = iw.get_average_values_of_nuclei(
        b_norm, coordinates_nuclei, nuclei_sizes, number_of_nuclei
    )

    coordinates_of_cytoplasm = sd.get_coordinates_of_pixels(
        img_separated_cytoplasm, cytoplasm_sizes, number_of_cytoplasm, width, height
    )

    DEV = 0.05
    for i in range(6):
        fn = round(DEV, 2)
        img_cytoplasm_by_average_value_of_nuclei = iw.check_cytoplasm_by_average_value_of_nuclei(
            b_norm,
            coordinates_of_cytoplasm,
            cytoplasm_sizes,
            average_values_of_nuclei,
            number_of_nuclei,
            number_of_cytoplasm,
            width,
            height,
            deviation=DEV,
        )
        plt.imsave(
            f"{output_path}IMG/17_DEV_{i}_{fn}_cytoplasm_by_average_value_of_nuclei.jpg",
            img_cytoplasm_by_average_value_of_nuclei,
            cmap="jet",
        )
        DEV += 0.05

    # ------------------ Končí kód pro analýzu ----------------------


def analysis_2(img, output_path):
    # použit b_channel

    width = img.shape[1]
    height = img.shape[0]

    # ------------------ Zde kód pro analýzu ------------------------
    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    r, g, b = cw.separate_layers(RGB_balanced, width, height)
    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_RGB_B_otsu.jpg", b_bin_otsu, cmap="gray")

    HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(HSL, width, height)

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")
    vw.histogram_2D_data(
        h_norm, "Hue", "Value", "Frequency", "01_Hue", f"{output_path}GRAPHS/", bins=100
    )

    threshold_angle_s = 0.62 - 0.1
    threshold_angle_f = 0.62 + 0.1

    img_h_bin = cw.convert_HSL_to_bin_by_angle(h_norm, threshold_angle_s, threshold_angle_f)
    plt.imsave(f"{output_path}IMG/04_HSL_h_bin.jpg", img_h_bin, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}IMG/05_RGB_B_OTSU_MORP.jpg", b_bin_otsu_morp, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(img_h_bin)
    plt.imsave(f"{output_path}IMG/06_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    img_b_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(
        f"{output_path}IMG/07_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )
    plt.imsave(f"{output_path}IMG/08_RGB_B_labeled_nuclei.jpg", img_b_labeled_nuclei, cmap="jet")

    # Hranice spojení jader a cytoplaz a ukládání
    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_nuclei_boundary = sd.get_boundary_4_connected(img_b_labeled_nuclei, width, height)
    plt.imsave(f"{output_path}IMG/09_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet")
    plt.imsave(f"{output_path}IMG/10_nuclei_boundary.jpg", img_nuclei_boundary, cmap="jet")
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )
    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    # velikosti
    cytoplasm_sizes = mh.labeled.labeled_size(img_h_labeled_cytoplasm)
    nuclei_sizes = mh.labeled.labeled_size(img_b_labeled_nuclei)

    cytoplasm_sizes[0] = 0  # pozadí nastavím na nulu
    nuclei_sizes[0] = 0  # pozadí nastavím na nulu

    # počet jader a cytoplasm
    number_of_cytoplasm = cytoplasm_sizes.shape[0]
    number_of_nuclei = nuclei_sizes.shape[0]

    # obvody
    perimeter_cytoplasm = mh.labeled.labeled_size(img_cytoplasm_boundary)
    perimeter_nuclei = mh.labeled.labeled_size(img_nuclei_boundary)

    perimeter_cytoplasm[0] = 0  # pozadí nastavím na nulu
    perimeter_nuclei[0] = 0  # pozadí nastavím na nulu

    # souřadnice hranic
    coordinates_nuclei_boundary = sd.get_coordinates_of_pixels(
        img_nuclei_boundary, perimeter_nuclei, number_of_nuclei, width, height
    )

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_b_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/13_castecne_rozdelena_cytoplazma.jpg", img_cytoplasm_nuclei, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}IMG/14_castecne_rozdelena_cytoplazma_hranice.jpg",
        img_cytoplasm_boundary,
        cmap="jet",
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    b_norm = cw.convert_img_to_norm_img(b, "RGB_B")

    coordinates_nuclei = sd.get_coordinates_of_pixels(
        img_b_labeled_nuclei, nuclei_sizes, number_of_nuclei, width, height
    )
    cytoplasm_sizes = mh.labeled.labeled_size(img_cytoplasm_nuclei)  # zahrnuje i jádra v cyto
    number_of_cytoplasm = cytoplasm_sizes.shape[0]
    cytoplasm_sizes[0] = 0

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    img_b_in_mask = img_only_cytoplasm_bin * b
    vw.histogram_2D_data_in_mask(
        img_b_in_mask,
        "",
        "value",
        "frequency",
        "hodnoty_pouze_cyto",
        f"{output_path}/GRAPHS/",
        bins=50,
        norm=False,
    )
    plt.imsave(f"{output_path}IMG/17_onlycytobin.jpg", img_only_cytoplasm_bin, cmap="gray")
    # plt.imsave(f'{output_path}IMG/18_onlycytoinB.jpg', img_b_in_mask, cmap='gray')

    img_b_nuclei = b * b_bin_otsu_morp
    vw.histogram_2D_data_in_mask(
        img_b_nuclei,
        "",
        "value",
        "frequency",
        "hodnoty_v_jadrech",
        f"{output_path}/GRAPHS/",
        bins=50,
        norm=False,
    )

    tv = -1

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height, threshold_value=tv
    )
    plt.imsave(f"{output_path}IMG/19_onlycytoinmaskB.jpg", img_b_in_cytoplasm_mask, cmap="gray")
    plt.imsave(
        f"{output_path}IMG/20_onlycytothreholdedbin.jpg", img_only_cytoplasm_threshold, cmap="gray"
    )

    img_cyto_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(f"{output_path}IMG/21_cyto and nuclei.jpg", img_cyto_and_nuclei_bin, cmap="gray")

    img_cyto_and_nuclei_labeled = img_cyto_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}IMG/22_cytonucleilabeled.jpg", img_cyto_and_nuclei_labeled, cmap="jet"
    )

    cytoplasm_sizes = mh.labeled.labeled_size(img_cyto_and_nuclei_labeled)
    cytoplasm_sizes[0] = 0
    number_of_cells = cytoplasm_sizes.shape[0]

    coordinates_cytoplasm = sd.get_coordinates_of_pixels(
        img_cyto_and_nuclei_labeled, cytoplasm_sizes, number_of_cells, width, height
    )

    img_repaired = iw.cell_repair(
        coordinates_cytoplasm, cytoplasm_sizes, number_of_cells, width, height
    )

    plt.imsave(f"{output_path}IMG/23_repaired.jpg", img_repaired, cmap="jet")

    img_cyto_hranice = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cyto_hranice_bin = cw.convert_labeled_to_bin(img_cyto_hranice)

    img_ori_hranice = iw.boundary_to_original_image(img, img_cyto_hranice_bin, width, height)
    plt.imsave(f"{output_path}IMG/24_hranice.jpg", img_ori_hranice)

    img_cyto_nucleu_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cyto_hranice_bin
    )
    img_ori_hranice = iw.boundary_to_original_image(img, img_cyto_nucleu_boundary, width, height)
    plt.imsave(f"{output_path}IMG/25_hranice.jpg", img_ori_hranice)

    # ----

    img_only_coty_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, number_of_nuclei)
    plt.imsave(
        f"{output_path}IMG/26_onlycytowhereisnuclei.jpg", img_only_coty_with_nuclei, cmap="jet"
    )

    img_only_coty_with_nuclei_removed_small = iw.remove_small_regions(img_only_coty_with_nuclei)
    plt.imsave(
        f"{output_path}IMG/27_onlycytowhereisnucleiwithoutsmallreg.jpg",
        img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_coty_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_coty_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}IMG/28_onlycytowhereisnucleiwithoutsmallreg_repa.jpg",
        img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_coty_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_coty_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}IMG/29_onlycytowhereisnuclei_boundary.jpg",
        boundary_img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    img_ori_hranice_cyto_nuclei = iw.boundary_cytoplasm_nuclei_to_original_image(
        img, img_nuclei_boundary, boundary_img_only_coty_with_nuclei_removed_small, width, height
    )
    plt.imsave(f"{output_path}IMG/30_hranice.jpg", img_ori_hranice_cyto_nuclei)

    only_cyto_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_coty_with_nuclei_removed_small, img_b_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/31_onlycyto.jpg",
        only_cyto_which_have_nuclei_but_not_included,
        cmap="jet",
    )

    cytoplasm_which_have_nuclei_sizes = mh.labeled.labeled_size(
        only_cyto_which_have_nuclei_but_not_included
    )
    cytoplasm_which_have_nuclei_sizes[0] = 0

    vw.histogram_1D_data(
        cytoplasm_which_have_nuclei_sizes,
        "",
        "Value",
        "frequency",
        "velikosti_cytoplasmy",
        f"{output_path}/GRAPHS/",
        bins=20,
    )

    vw.histogram_1D_data(
        nuclei_sizes, "", "Value", "frequency", "velikosti_jader", f"{output_path}/GRAPHS/", bins=20
    )

    # ------------------ Končí kód pro analýzu ----------------------


def analysis_3(img, output_path):
    # použit grayscale z RGB

    width = img.shape[1]
    height = img.shape[0]

    # ------------------ Zde kód pro analýzu ------------------------
    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    b = cw.convert_RGB_to_grayscale(RGB_balanced, width, height)

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_RGB_B_otsu.jpg", b_bin_otsu, cmap="gray")

    HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(HSL, width, height)

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")
    vw.histogram_2D_data(
        h_norm, "Hue", "Value", "Frequency", "01_Hue", f"{output_path}GRAPHS/", bins=100
    )

    threshold_angle_s = 0.62 - 0.1
    threshold_angle_f = 0.62 + 0.1

    img_h_bin = cw.convert_HSL_to_bin_by_angle(h_norm, threshold_angle_s, threshold_angle_f)
    plt.imsave(f"{output_path}IMG/04_HSL_h_bin.jpg", img_h_bin, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(f"{output_path}IMG/05_RGB_B_OTSU_MORP.jpg", b_bin_otsu_morp, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(img_h_bin)
    plt.imsave(f"{output_path}IMG/06_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    img_b_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(
        f"{output_path}IMG/07_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )
    plt.imsave(f"{output_path}IMG/08_RGB_B_labeled_nuclei.jpg", img_b_labeled_nuclei, cmap="jet")

    # Hranice spojení jader a cytoplaz a ukládání
    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_nuclei_boundary = sd.get_boundary_4_connected(img_b_labeled_nuclei, width, height)
    plt.imsave(f"{output_path}IMG/09_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet")
    plt.imsave(f"{output_path}IMG/10_nuclei_boundary.jpg", img_nuclei_boundary, cmap="jet")
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )
    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    # velikosti
    cytoplasm_sizes = mh.labeled.labeled_size(img_h_labeled_cytoplasm)
    nuclei_sizes = mh.labeled.labeled_size(img_b_labeled_nuclei)

    cytoplasm_sizes[0] = 0  # pozadí nastavím na nulu
    nuclei_sizes[0] = 0  # pozadí nastavím na nulu

    # počet jader a cytoplasm
    number_of_cytoplasm = cytoplasm_sizes.shape[0]
    number_of_nuclei = nuclei_sizes.shape[0]

    # obvody
    perimeter_cytoplasm = mh.labeled.labeled_size(img_cytoplasm_boundary)
    perimeter_nuclei = mh.labeled.labeled_size(img_nuclei_boundary)

    perimeter_cytoplasm[0] = 0  # pozadí nastavím na nulu
    perimeter_nuclei[0] = 0  # pozadí nastavím na nulu

    # souřadnice hranic
    coordinates_nuclei_boundary = sd.get_coordinates_of_pixels(
        img_nuclei_boundary, perimeter_nuclei, number_of_nuclei, width, height
    )

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_b_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/13_castecne_rozdelena_cytoplazma.jpg", img_cytoplasm_nuclei, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}IMG/14_castecne_rozdelena_cytoplazma_hranice.jpg",
        img_cytoplasm_boundary,
        cmap="jet",
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    b_norm = cw.convert_img_to_norm_img(b, "RGB_B")

    coordinates_nuclei = sd.get_coordinates_of_pixels(
        img_b_labeled_nuclei, nuclei_sizes, number_of_nuclei, width, height
    )
    cytoplasm_sizes = mh.labeled.labeled_size(img_cytoplasm_nuclei)  # zahrnuje i jádra v cyto
    number_of_cytoplasm = cytoplasm_sizes.shape[0]
    cytoplasm_sizes[0] = 0

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}IMG/17_onlycytobin.jpg", img_only_cytoplasm_bin, cmap="gray")
    img_b_in_mask = img_only_cytoplasm_bin * b
    # vw.histogram_2D_data_in_mask(img_b_in_mask, '', 'value', 'frequency', 'hodnoty_pouze_cyto',f'{output_path}/GRAPHS/', bins=50, norm=False)

    # plt.imsave(f'{output_path}IMG/18_onlycytoinB.jpg', img_b_in_mask, cmap='gray')

    img_b_nuclei = b * b_bin_otsu_morp
    vw.histogram_2D_data_in_mask(
        img_b_nuclei,
        "",
        "value",
        "frequency",
        "hodnoty_v_jadrech",
        f"{output_path}/GRAPHS/",
        bins=50,
        norm=False,
    )

    tv = -1

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height, threshold_value=tv
    )
    plt.imsave(f"{output_path}IMG/19_onlycytoinmaskB.jpg", img_b_in_cytoplasm_mask, cmap="gray")
    plt.imsave(
        f"{output_path}IMG/20_onlycytothreholdedbin.jpg", img_only_cytoplasm_threshold, cmap="gray"
    )

    img_cyto_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(f"{output_path}IMG/21_cyto and nuclei.jpg", img_cyto_and_nuclei_bin, cmap="gray")

    img_cyto_and_nuclei_labeled = img_cyto_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}IMG/22_cytonucleilabeled.jpg", img_cyto_and_nuclei_labeled, cmap="jet"
    )

    cytoplasm_sizes = mh.labeled.labeled_size(img_cyto_and_nuclei_labeled)
    cytoplasm_sizes[0] = 0
    number_of_cells = cytoplasm_sizes.shape[0]

    coordinates_cytoplasm = sd.get_coordinates_of_pixels(
        img_cyto_and_nuclei_labeled, cytoplasm_sizes, number_of_cells, width, height
    )

    img_repaired = iw.cell_repair(
        coordinates_cytoplasm, cytoplasm_sizes, number_of_cells, width, height
    )

    plt.imsave(f"{output_path}IMG/23_repaired.jpg", img_repaired, cmap="jet")

    img_cyto_hranice = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cyto_hranice_bin = cw.convert_labeled_to_bin(img_cyto_hranice)

    img_ori_hranice = iw.boundary_to_original_image(img, img_cyto_hranice_bin, width, height)
    plt.imsave(f"{output_path}IMG/24_hranice.jpg", img_ori_hranice)

    img_cyto_nucleu_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cyto_hranice_bin
    )
    img_ori_hranice = iw.boundary_to_original_image(img, img_cyto_nucleu_boundary, width, height)
    plt.imsave(f"{output_path}IMG/25_hranice.jpg", img_ori_hranice)

    img_only_coty_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, number_of_nuclei)
    plt.imsave(
        f"{output_path}IMG/26_onlycytowhereisnuclei.jpg", img_only_coty_with_nuclei, cmap="jet"
    )

    img_only_coty_with_nuclei_removed_small = iw.remove_small_regions(img_only_coty_with_nuclei)
    plt.imsave(
        f"{output_path}IMG/27_onlycytowhereisnucleiwithoutsmallreg.jpg",
        img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_coty_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_coty_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}IMG/28_onlycytowhereisnucleiwithoutsmallreg_repa.jpg",
        img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_coty_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_coty_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}IMG/29_onlycytowhereisnuclei_boundary.jpg",
        boundary_img_only_coty_with_nuclei_removed_small,
        cmap="jet",
    )

    img_ori_hranice_cyto_nuclei = iw.boundary_cytoplasm_nuclei_to_original_image(
        img, img_nuclei_boundary, boundary_img_only_coty_with_nuclei_removed_small, width, height
    )
    plt.imsave(f"{output_path}IMG/30_hranice.jpg", img_ori_hranice_cyto_nuclei)

    only_cyto_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_coty_with_nuclei_removed_small, img_b_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/31_onlycyto.jpg",
        only_cyto_which_have_nuclei_but_not_included,
        cmap="jet",
    )

    cytoplasm_which_have_nuclei_sizes = mh.labeled.labeled_size(
        only_cyto_which_have_nuclei_but_not_included
    )
    cytoplasm_which_have_nuclei_sizes[0] = 0

    vw.histogram_1D_data(
        cytoplasm_which_have_nuclei_sizes,
        "",
        "Value",
        "frequency",
        "velikosti_cytoplasmy",
        f"{output_path}/GRAPHS/",
        bins=20,
    )

    vw.histogram_1D_data(
        nuclei_sizes, "", "Value", "frequency", "velikosti_jader", f"{output_path}/GRAPHS/", bins=20
    )

    # ------------------ Končí kód pro analýzu ----------------------


def analysis_4(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/blue_00_input_img_00.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/blue_01_blur_0.jpg", img_blur)

    # cv2.imwrite(f'{output_path}IMG/blue_01_blur_1.jpg', img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    h, s, v = cw.separate_layers(img_hsv)

    plt.imsave(f"{output_path}IMG/blue_02_0_h.jpg", h, cmap="gray")
    plt.imsave(f"{output_path}IMG/blue_02_1_s.jpg", s, cmap="gray")
    plt.imsave(f"{output_path}IMG/blue_02_2_v.jpg", v, cmap="gray")

    vw.histogram_2D_data(
        h,
        "Hue",
        "Value",
        "Frequency",
        "blue_01_hue",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        s,
        "Saturation",
        "Value",
        "Frequency",
        "blue_02_saturation",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        v,
        "Value",
        "Value",
        "Frequency",
        "blue_03_value",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )

    # Hranice pro modrou barvu v prostoru HSV
    lower_blue = np.array([60, 25, 50])
    upper_blue = np.array([125, 200, 220])

    mask_final = cv2.inRange(img_hsv, lower_blue, upper_blue)
    plt.imsave(f"{output_path}IMG/blue_04_final_mask.jpg", mask_final, cmap="gray")

    kernel = np.ones((3, 3))
    mask_final_opening = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=2)
    plt.imsave(f"{output_path}IMG/blue_05_mask_h_opening.jpg", mask_final_opening, cmap="gray")

    img_boundary_blue = sd.get_boundary_4_connected(mask_final_opening, width, height)
    plt.imsave(f"{output_path}IMG/blue_06_boundary.jpg", img_boundary_blue, cmap="gray")

    img_original_with_boundary_blue = iw.boundary_to_original_image(
        img, img_boundary_blue, width, height, [0, 0, 255]
    )
    plt.imsave(
        f"{output_path}IMG/blue_07_boundary_in_original_image.jpg",
        img_original_with_boundary_blue,
        cmap="gray",
    )

    # ----------------
    # ---------------- BROWN
    # ----------------
    # '''
    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/brown_01_blur.jpg", img_blur)

    # Hranice pro hnědou barvu v prostoru HSV
    lower_brown_1 = np.array([0, 10, 10])
    upper_brown_1 = np.array([25, 210, 180])

    lower_brown_2 = np.array([125, 10, 10])
    upper_brown_2 = np.array([360, 210, 150])

    mask_1 = cv2.inRange(img_hsv, lower_brown_1, upper_brown_1)
    mask_2 = cv2.inRange(img_hsv, lower_brown_2, upper_brown_2)

    mask_final = np.logical_or(mask_1, mask_2).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_02_final_mask.jpg", mask_final, cmap="gray")

    kernel = np.ones((3, 3), np.uint8)
    mask_h_opening = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
    plt.imsave(f"{output_path}IMG/brown_03_mask_h_opening.jpg", mask_h_opening, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening, width, height)
    plt.imsave(f"{output_path}IMG/brown_04_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [150, 75, 0]
    )
    plt.imsave(
        f"{output_path}IMG/brown_05_boundary_in_original_image.jpg", img_original_with_boundary
    )

    # ----------------
    # ---------------- BOTH BOUNDARIES
    # ----------------

    img_both_boundary = iw.two_boundary_types_to_original_image(
        img, img_boundary_blue, img_boundary_brown, width, height, [0, 0, 255], [150, 75, 0]
    )
    plt.imsave(f"{output_path}IMG/final_both_boundary_in_original_image.jpg", img_both_boundary)
    # '''


def analysis_5(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img_00.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    # ----------------
    # ---------------- Shared
    # ----------------

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/01_blur.jpg", img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    h, s, v = cw.separate_layers(img_hsv)

    # plt.imsave(f'{output_path}IMG/02_0_h.jpg', h, cmap='gray')
    # plt.imsave(f'{output_path}IMG/02_1_s.jpg', s, cmap='gray')
    # plt.imsave(f'{output_path}IMG/02_2_v.jpg', v, cmap='gray')

    vw.histogram_2D_data(
        h, "Hue", "Value", "Frequency", "01_hue", output_path, bins=100, norm=False, txt_file=False
    )
    vw.histogram_2D_data(
        s,
        "Saturation",
        "Value",
        "Frequency",
        "02_saturation",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        v,
        "Value",
        "Value",
        "Frequency",
        "03_value",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )

    kernel = np.ones((3, 3))

    # ----------------
    # ---------------- BLUE
    # ----------------

    # HSV - blue color definition
    lower_blue = np.array([60, 25, 50])
    upper_blue = np.array([125, 200, 220])

    mask_final_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    plt.imsave(f"{output_path}IMG/blue_01_final_mask.jpg", mask_final_blue, cmap="gray")

    mask_final_opening_blue = cv2.morphologyEx(
        mask_final_blue, cv2.MORPH_OPEN, kernel, iterations=2
    )
    plt.imsave(f"{output_path}IMG/blue_02_mask_h_opening.jpg", mask_final_opening_blue, cmap="gray")

    img_boundary_blue = sd.get_boundary_4_connected(mask_final_opening_blue, width, height)
    plt.imsave(f"{output_path}IMG/blue_03_boundary.jpg", img_boundary_blue, cmap="gray")

    img_original_with_boundary_blue = iw.boundary_to_original_image(
        img, img_boundary_blue, width, height, [0, 0, 255]
    )
    plt.imsave(
        f"{output_path}IMG/blue_04_boundary_in_original_image.jpg",
        img_original_with_boundary_blue,
        cmap="gray",
    )

    # ----------------
    # ---------------- BROWN
    # ----------------

    # HSV - brown color definition
    # '''
    lower_brown_1 = np.array([0, 10, 10])
    upper_brown_1 = np.array([25, 210, 180])

    lower_brown_2 = np.array([125, 10, 10])
    upper_brown_2 = np.array([360, 210, 150])

    wb = cv2.xphoto.createGrayworldWB()

    img_balanced = wb.balanceWhite(img)

    plt.imsave(f"{output_path}IMG/brown_00_balanced.jpg", img_balanced)

    img_blur = cv2.blur(img_balanced, (5, 5))

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    mask_1 = cv2.inRange(img_hsv, lower_brown_1, upper_brown_1)
    plt.imsave(f"{output_path}IMG/brown_01_mask_1.jpg", mask_1, cmap="gray")

    mask_2 = cv2.inRange(img_hsv, lower_brown_2, upper_brown_2)
    plt.imsave(f"{output_path}IMG/brown_01_mask_2.jpg", mask_2, cmap="gray")

    mask_final_brown = np.logical_or(mask_1, mask_2).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_02_final_mask.jpg", mask_final_brown, cmap="gray")
    # '''

    mask_h_opening_brown = cv2.morphologyEx(mask_final_brown, cv2.MORPH_OPEN, kernel, iterations=2)
    plt.imsave(f"{output_path}IMG/brown_03_mask_h_opening.jpg", mask_h_opening_brown, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_04_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary_brown = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [255, 255, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_05_boundary_in_original_image.jpg",
        img_original_with_boundary_brown,
    )

    # ----------------
    # ---------------- COMBINATION
    # ----------------

    img_both_boundary = iw.two_boundary_types_to_original_image(
        img, img_boundary_blue, img_boundary_brown, width, height, [0, 0, 255], [255, 255, 255]
    )
    plt.imsave(f"{output_path}IMG/final_both_boundary_in_original_image.jpg", img_both_boundary)


def analysis_6(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img_00.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/01_blur.jpg", img_blur)

    r, g, b = cw.separate_layers(img)

    mask_brown = cw.convert_grayscale_to_bin(b, threshold_value=80, less_than=True).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_01_mask_h_opening.jpg", mask_brown, cmap="gray")

    kernel = np.ones((5, 5))
    mask_final_opening_brown = cv2.morphologyEx(mask_brown, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imsave(
        f"{output_path}IMG/brown_02_mask_h_opening.jpg", mask_final_opening_brown, cmap="gray"
    )

    img_boundary_blue = sd.get_boundary_4_connected(mask_final_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_03_boundary.jpg", img_boundary_blue, cmap="gray")

    img_original_with_boundary_blue = iw.boundary_to_original_image(
        img, img_boundary_blue, width, height, [0, 0, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_04_boundary_in_original_image.jpg",
        img_original_with_boundary_blue,
        cmap="gray",
    )


def analysis_7(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img_00.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    # ----------------
    # ---------------- Shared
    # ----------------

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/01_blur.jpg", img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    h, s, v = cw.separate_layers(img_hsv)

    # plt.imsave(f'{output_path}IMG/02_0_h.jpg', h, cmap='gray')
    # plt.imsave(f'{output_path}IMG/02_1_s.jpg', s, cmap='gray')
    # plt.imsave(f'{output_path}IMG/02_2_v.jpg', v, cmap='gray')

    vw.histogram_2D_data(
        h, "Hue", "Value", "Frequency", "01_hue", output_path, bins=100, norm=False, txt_file=False
    )
    vw.histogram_2D_data(
        s,
        "Saturation",
        "Value",
        "Frequency",
        "02_saturation",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        v,
        "Value",
        "Value",
        "Frequency",
        "03_value",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )

    kernel = np.ones((3, 3))

    # ----------------
    # ---------------- BLUE
    # ----------------

    # HSV - blue color definition
    lower_blue = np.array([60, 25, 50])
    upper_blue = np.array([125, 200, 220])

    mask_final_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    plt.imsave(f"{output_path}IMG/blue_01_final_mask.jpg", mask_final_blue, cmap="gray")

    mask_final_opening_blue = cv2.morphologyEx(
        mask_final_blue, cv2.MORPH_OPEN, kernel, iterations=2
    )
    plt.imsave(f"{output_path}IMG/blue_02_mask_h_opening.jpg", mask_final_opening_blue, cmap="gray")

    img_boundary_blue = sd.get_boundary_4_connected(mask_final_opening_blue, width, height)
    plt.imsave(f"{output_path}IMG/blue_03_boundary.jpg", img_boundary_blue, cmap="gray")

    img_original_with_boundary_blue = iw.boundary_to_original_image(
        img, img_boundary_blue, width, height, [0, 0, 255]
    )
    plt.imsave(
        f"{output_path}IMG/blue_04_boundary_in_original_image.jpg",
        img_original_with_boundary_blue,
        cmap="gray",
    )

    # ----------------
    # ---------------- BROWN
    # ----------------

    # HSV - brown color definition
    lower_brown_1 = np.array([0, 20, 30])
    upper_brown_1 = np.array([40, 165, 140])

    vw.write_limits(lower_brown_1, upper_brown_1, "HSV_brown", output_path)

    lower_brown_2 = np.array([125, 10, 10])
    upper_brown_2 = np.array([360, 210, 150])

    mask_1 = cv2.inRange(img_hsv, lower_brown_1, upper_brown_1)
    plt.imsave(f"{output_path}IMG/brown_01_mask_1.jpg", mask_1, cmap="gray")

    mask_2 = cv2.inRange(img_hsv, lower_brown_2, upper_brown_2)
    plt.imsave(f"{output_path}IMG/brown_01_mask_2.jpg", mask_2, cmap="gray")

    mask_final_brown = np.logical_or(mask_1, mask_2).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_02_final_mask.jpg", mask_final_brown, cmap="gray")

    mask_h_opening_brown = cv2.morphologyEx(mask_final_brown, cv2.MORPH_OPEN, kernel, iterations=1)
    plt.imsave(f"{output_path}IMG/brown_02_mask_h_opening.jpg", mask_h_opening_brown, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_03_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary_brown = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [255, 255, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_04_boundary_in_original_image.jpg",
        img_original_with_boundary_brown,
    )

    # ----------------
    # ---------------- COMBINATION
    # ----------------

    img_both_boundary = iw.two_boundary_types_to_original_image(
        img, img_boundary_blue, img_boundary_brown, width, height, [0, 0, 255], [255, 255, 255]
    )
    plt.imsave(f"{output_path}IMG/final_both_boundary_in_original_image.jpg", img_both_boundary)


def analysis_8(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    kernel = np.ones((3, 3))

    # ------------------ Zde kód pro analýzu ------------------------

    # ----------------
    # ---------------- BROWN
    # ----------------

    plt.imsave(f"{output_path}IMG/brown_00_input_img.jpg", img)

    # HSV - brown color definition
    lower_brown_1 = np.array([0, 10, 10])
    upper_brown_1 = np.array([25, 210, 180])

    lower_brown_2 = np.array([125, 10, 10])
    upper_brown_2 = np.array([360, 210, 150])

    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.1)
    img_balanced = wb.balanceWhite(img)
    plt.imsave(f"{output_path}IMG/brown_01_balanced.jpg", img_balanced)

    img_blur = cv2.blur(img_balanced, (5, 5))
    plt.imsave(f"{output_path}IMG/brown_02_blur.jpg", img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    mask_1 = cv2.inRange(img_hsv, lower_brown_1, upper_brown_1)
    plt.imsave(f"{output_path}IMG/brown_03_mask_1.jpg", mask_1, cmap="gray")

    mask_2 = cv2.inRange(img_hsv, lower_brown_2, upper_brown_2)
    plt.imsave(f"{output_path}IMG/brown_03_mask_2.jpg", mask_2, cmap="gray")

    mask_final_brown = np.logical_or(mask_1, mask_2).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_04_final_mask.jpg", mask_final_brown, cmap="gray")

    mask_h_opening_brown = cv2.morphologyEx(mask_final_brown, cv2.MORPH_OPEN, kernel, iterations=2)
    plt.imsave(f"{output_path}IMG/brown_05_mask_h_opening.jpg", mask_h_opening_brown, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_06_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary_brown = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [255, 255, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_07_boundary_in_original_image.jpg",
        img_original_with_boundary_brown,
    )


def analysis_9(img, output_path):
    # multi otsu

    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------
    """
    # HSV - blue color definition
    lower_blue = np.array([60, 25, 50])
    upper_blue = np.array([125, 200, 220])

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f'{output_path}IMG/blue_01_blur.jpg', img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)

    mask_final_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    plt.imsave(f'{output_path}IMG/blue_02_final_mask.jpg', mask_final_blue, cmap='gray')
    """

    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    r, g, b = cw.separate_layers(RGB_balanced)

    thresholds = filters.threshold_multiotsu(r)

    regions = np.digitize(r, bins=thresholds)
    plt.imsave(f"{output_path}IMG/02_multi_otsu.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}IMG/03_bin_cytoplasm_nuclei.jpg", bin_cyto_nuclei, cmap="gray")

    bin_two = cw.convert_labeled_to_bin(regions, background=0)

    bin_cyto = np.logical_and(bin_cyto_nuclei, bin_two)
    plt.imsave(f"{output_path}IMG/04_bin_only_cytoplasm.jpg", bin_cyto, cmap="gray")

    bin_nuclei = np.logical_not(bin_two)
    plt.imsave(f"{output_path}IMG/05_bin_only_nuclei.jpg", bin_nuclei, cmap="gray")

    # ______________________________________________________________

    b_bin_otsu_morp = iw.close_holes_remove_noise(bin_nuclei)
    plt.imsave(
        f"{output_path}IMG/06_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray"
    )

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}IMG/07_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}IMG/08_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/09_boundary_in_original_img.jpg", img_boundary_in_original)

    # ______________________________________________________________

    img_h_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}IMG/10_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    plt.imsave(
        f"{output_path}IMG/11_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}IMG/12_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    # ______________________________________________________________

    # Spojení
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}IMG/13_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/14_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_labeled_nuclei, width, height
    )
    plt.imsave(f"{output_path}IMG/15_flooding_cytoplasm.jpg", img_cytoplasm_nuclei, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}IMG/16_flooding_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet"
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/17_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/18_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}IMG/19_only_cytoplasm_bin.jpg", img_only_cytoplasm_bin, cmap="gray")

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height
    )
    plt.imsave(
        f"{output_path}IMG/20_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}IMG/21_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(
        f"{output_path}IMG/22_cytoplasm_and_nuclei_bin.jpg",
        img_cytoplasm_and_nuclei_bin,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}IMG/23_cytoplasm_and_nuclei_labeled.jpg",
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
    plt.imsave(f"{output_path}IMG/24_cytoplasm_and_nuclei_repaired.jpg", img_repaired, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cytoplasm_boundary_bin
    )
    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/25_boundary_in_original_img.jpg", img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(
        f"{output_path}IMG/26_only_cytoplasm_with_nuclei.jpg",
        img_only_cytoplasm_with_nuclei,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei), is_bin=True
    )
    plt.imsave(
        f"{output_path}IMG/27_only_cytoplasm_with_nuclei_without_small_reg.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}IMG/28_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_cytoplasm_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}IMG/29_only_cytoplasm_with_nuclei_boundary.jpg",
        boundary_img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/30_only_cytoplasm_which_have_nuclei.jpg",
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
    plt.imsave(f"{output_path}IMG/31_boundary_final.jpg", img_original_boundary_cytoplasm_nuclei)


def analysis_10(img, output_path):
    # FFT

    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    r, g, b = cw.separate_layers(RGB_balanced)
    plt.imsave(f"{output_path}IMG/02_blue_channel.jpg", b, cmap="gray")

    fft_b_channel = np.fft.fft2(b)
    fft_b_channel_mag = 20 * np.log(np.abs(fft_b_channel))
    plt.imsave(f"{output_path}IMG/03_fft_magnitude.jpg", fft_b_channel_mag, cmap="gray")

    fft_b_channel_shift = np.fft.fftshift(fft_b_channel)
    fft_b_channel_shift_mag = 20 * np.log(np.abs(fft_b_channel_shift))
    plt.imsave(f"{output_path}IMG/04_fft_shift_magnitude.jpg", fft_b_channel_shift_mag, cmap="gray")

    mask = iw.fft_filter_circle(0.3, width, height)
    plt.imsave(f"{output_path}IMG/05_mask.jpg", mask, cmap="gray")

    fft_b_channel_shift_mask = (fft_b_channel_shift.real * mask) + (fft_b_channel_shift.imag * mask)
    plt.imsave(
        f"{output_path}IMG/06_fft_b_channel_shift_mask.jpg",
        abs(fft_b_channel_shift_mask),
        cmap="gray",
    )

    fft_ishift = np.fft.ifftshift(fft_b_channel_shift_mask)
    plt.imsave(f"{output_path}IMG/07_fft_ishift.jpg", abs(fft_ishift), cmap="gray")

    ifft = np.fft.ifft2(fft_ishift)
    plt.imsave(f"{output_path}IMG/08_ifft.jpg", abs(ifft), cmap="gray")


def analysis_11(img, output_path):
    # zde vyplnuji obrázek a pak uložím okometricky porovnám s moji analýzou

    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    r, g, b = cw.separate_layers(img)

    img_bin = cw.convert_labeled_to_bin(r)

    plt.imsave(f"{output_path}IMG/02_bin_boundary.jpg", img_bin, cmap="gray")

    img_labeled, _ = mh.label(img_bin)

    nr_objects = np.amax(img_labeled)

    labeled_filled_img = np.zeros((height, width))

    for i in range(1, nr_objects + 1):
        # hranice jedné buňky do bin
        current_img = img_labeled == i
        # vyplnění objekt
        current_img = com.fill_boundaries(current_img)
        # label vrácený zpět
        current_img = current_img * i
        # přidán do výsledného snímku
        labeled_filled_img = labeled_filled_img + current_img
        # oprava překrytí vždycky ten co je přidán podlední má překrytí
        labeled_filled_img[labeled_filled_img > i] -= i

    plt.imsave(f"{output_path}IMG/02_labeled_filled.jpg", labeled_filled_img, cmap="jet")


def analysis_12(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    # Preprocessing
    # RGB_balanced = img
    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce jader
    r, g, b = cw.separate_layers(RGB_balanced)

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(
        f"{output_path}IMG/03_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray"
    )

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}IMG/04_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    nuclei_sizes = mh.labeled.labeled_size(img_labeled_nuclei)
    nuclei_sizes[0] = 0
    vw.histogram_1D_data(
        nuclei_sizes, "Nuclei sizes", "size", "frequency", "Nuclei_sizes", output_path
    )

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}IMG/05_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/06_boundary_in_original_img.jpg", img_boundary_in_original)

    mean_value_in_nuclei = iw.mean_in_mask(b, b_bin_otsu_morp)
    print(mean_value_in_nuclei)
    vw.histogram_2D_data_range(
        b, "Blue channel", "Value", "Frequency", "03_Blue", output_path, 0, 255, 1, txt_file=True
    )
    interval = 50

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu
    img_HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(img_HSL)

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")

    center = 0.66
    sigma = 0.07
    lower = center - sigma
    upper = center + sigma

    img_h_bin = cw.convert_grayscale_to_bin_by_range(h_norm, lower, upper)
    plt.imsave(f"{output_path}IMG/07_HSL_h_bin.jpg", img_h_bin, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(img_h_bin)
    plt.imsave(f"{output_path}IMG/08_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    plt.imsave(
        f"{output_path}IMG/09_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}IMG/10_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    # ret2, img_h_bin_otsu = cv2.threshold(h, 0, 6.4, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imsave(f'{output_path}IMG/10_cytoplasm_bin_otsu.jpg', img_h_bin_otsu, cmap='gray')
    val = filters.threshold_otsu(h)
    res = cw.convert_grayscale_to_bin(h, threshold_value=val, less_than=False)
    plt.imsave(f"{output_path}IMG/10_cytoplasm_bin_otsu.jpg", res, cmap="gray")

    # ---------------------------------------------------------------------------------------------------------------- #
    # Spojení
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_labeled_nuclei, width, height
    )
    plt.imsave(f"{output_path}IMG/13_flooding_cytoplasm.jpg", img_cytoplasm_nuclei, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}IMG/14_flooding_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet"
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}IMG/17_only_cytoplasm_bin.jpg", img_only_cytoplasm_bin, cmap="gray")

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height
    )
    plt.imsave(
        f"{output_path}IMG/18_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}IMG/19_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    img_only_cytoplasm_threshold, img_b_in_nuclei_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height, threshold_value=mean_value_in_nuclei + interval
    )
    plt.imsave(
        f"{output_path}IMG/20_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}IMG/21_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    """
    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(f'{output_path}IMG/20_cytoplasm_and_nuclei_bin.jpg', img_cytoplasm_and_nuclei_bin, cmap='gray')

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(f'{output_path}IMG/21_cytoplasm_and_nuclei_labeled.jpg', img_cytoplasm_and_nuclei_labeled, cmap='jet')

    cytoplasm_sizes = mh.labeled.labeled_size(img_cytoplasm_and_nuclei_labeled)
    cytoplasm_sizes[0] = 0
    number_of_cells = cytoplasm_sizes.shape[0]

    coordinates_cytoplasm = sd.get_coordinates_of_pixels(img_cytoplasm_and_nuclei_labeled, cytoplasm_sizes, number_of_cells,width, height)

    img_repaired = iw.cell_repair(coordinates_cytoplasm, cytoplasm_sizes, number_of_cells, width, height)
    plt.imsave(f'{output_path}IMG/22_cytoplasm_and_nuclei_repaired.jpg', img_repaired, cmap='jet')

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(img_nuclei_boundary_bin + img_cytoplasm_boundary_bin)
    img_boundary_in_original = iw.boundary_to_original_image(img, img_cytoplasm_nuclei_boundary, width, height)
    plt.imsave(f'{output_path}IMG/23_boundary_in_original_img.jpg', img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(f'{output_path}IMG/24_only_cytoplasm_with_nuclei.jpg', img_only_cytoplasm_with_nuclei, cmap='jet')

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei),is_bin = True)
    plt.imsave(f'{output_path}IMG/25_only_cytoplasm_with_nuclei_without_small_reg.jpg', img_only_cytoplasm_with_nuclei_removed_small,cmap='jet')

    img_only_cytoplasm_with_nuclei_removed_small = cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    plt.imsave(f'{output_path}IMG/26_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg', img_only_cytoplasm_with_nuclei_removed_small, cmap='jet')

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(img_only_cytoplasm_with_nuclei_removed_small, width, height)
    plt.imsave(f'{output_path}IMG/27_only_cytoplasm_with_nuclei_boundary.jpg', boundary_img_only_cytoplasm_with_nuclei_removed_small, cmap='jet')

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height)
    plt.imsave(f'{output_path}IMG/28_only_cytoplasm_which_have_nuclei.jpg', only_cytoplasm_which_have_nuclei_but_not_included, cmap='jet')

    img_original_boundary_cytoplasm_nuclei = iw.two_boundary_types_to_original_image(img, img_nuclei_boundary,boundary_img_only_cytoplasm_with_nuclei_removed_small,width, height)
    plt.imsave(f'{output_path}IMG/29_boundary_final.jpg', img_original_boundary_cytoplasm_nuclei)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Histogramy
    vw.histogram_2D_data_range(r, 'Red channel', 'Value', 'Frequency', '01_Red', output_path, 0,255,1, txt_file=True)
    vw.histogram_2D_data_range(g, 'Green channel', 'Value', 'Frequency', '02_Green', output_path, 0,255,1, txt_file=True)
    vw.histogram_2D_data_range(b, 'Blue channel', 'Value', 'Frequency', '03_Blue', output_path, 0,255,1, txt_file=True)

    vw.histogram_2D_data_range(h, 'H channel', 'Value', 'Frequency', '04_Hue', output_path, 0,6.3,0.1, txt_file=True)
    vw.histogram_2D_data_range(s, 'S channel', 'Value', 'Frequency', '05_Saturation', output_path, 0,1,0.02, txt_file=True)
    vw.histogram_2D_data_range(l, 'L channel', 'Value', 'Frequency', '06_Luminance', output_path, 0,765,5, txt_file=True)
    #"""


def analysis_13(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    # Preprocessing
    # RGB_balanced = img
    RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce jader
    r, g, b = cw.separate_layers(RGB_balanced)

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(
        f"{output_path}IMG/03_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray"
    )

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}IMG/04_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}IMG/05_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/06_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu
    img_HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(img_HSL)

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")

    center = 0.66
    sigma = 0.07
    lower = center - sigma
    upper = center + sigma

    img_h_bin = cw.convert_grayscale_to_bin_by_range(h_norm, lower, upper)
    plt.imsave(f"{output_path}IMG/07_HSL_h_bin.jpg", img_h_bin, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(img_h_bin)
    plt.imsave(f"{output_path}IMG/08_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    plt.imsave(
        f"{output_path}IMG/09_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}IMG/10_cytoplasm_boundary1.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/10_cytoplasm_boundary2.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Spojení
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(
        f"{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg", cytoplasm_nuclei_boundary, cmap="gray"
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg", boundary_original_img)

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(
        img_h_labeled_cytoplasm, img_labeled_nuclei, width, height
    )
    plt.imsave(f"{output_path}IMG/13_flooding_cytoplasm.jpg", img_cytoplasm_nuclei, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(
        f"{output_path}IMG/14_flooding_cytoplasm_boundary.jpg", img_cytoplasm_boundary, cmap="jet"
    )

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(
        cytoplasm_separated_nuclei_boundary
    )
    plt.imsave(
        f"{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg",
        cytoplasm_separated_nuclei_boundary,
        cmap="gray",
    )

    boundary_original_img = iw.boundary_to_original_image(
        img, cytoplasm_separated_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg", boundary_original_img)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f"{output_path}IMG/17_only_cytoplasm_bin.jpg", img_only_cytoplasm_bin, cmap="gray")

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(
        b, img_only_cytoplasm_bin, width, height
    )
    plt.imsave(
        f"{output_path}IMG/18_only_cytoplasm_in_blue_channel.jpg",
        img_b_in_cytoplasm_mask,
        cmap="gray",
    )
    plt.imsave(
        f"{output_path}IMG/19_only_cytoplasm_threshold_in_mask.jpg",
        img_only_cytoplasm_threshold,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(
        f"{output_path}IMG/20_cytoplasm_and_nuclei_bin.jpg",
        img_cytoplasm_and_nuclei_bin,
        cmap="gray",
    )

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(
        f"{output_path}IMG/21_cytoplasm_and_nuclei_labeled.jpg",
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
    plt.imsave(f"{output_path}IMG/22_cytoplasm_and_nuclei_repaired.jpg", img_repaired, cmap="jet")

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(
        img_nuclei_boundary_bin + img_cytoplasm_boundary_bin
    )
    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_nuclei_boundary, width, height
    )
    plt.imsave(f"{output_path}IMG/23_boundary_in_original_img.jpg", img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(
        f"{output_path}IMG/24_only_cytoplasm_with_nuclei.jpg",
        img_only_cytoplasm_with_nuclei,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei), is_bin=True
    )
    plt.imsave(
        f"{output_path}IMG/25_only_cytoplasm_with_nuclei_without_small_reg.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    img_only_cytoplasm_with_nuclei_removed_small = (
        cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    )
    plt.imsave(
        f"{output_path}IMG/26_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg",
        img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(
        img_only_cytoplasm_with_nuclei_removed_small, width, height
    )
    plt.imsave(
        f"{output_path}IMG/27_only_cytoplasm_with_nuclei_boundary.jpg",
        boundary_img_only_cytoplasm_with_nuclei_removed_small,
        cmap="jet",
    )

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(
        img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height
    )
    plt.imsave(
        f"{output_path}IMG/28_only_cytoplasm_which_have_nuclei.jpg",
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
    plt.imsave(f"{output_path}IMG/29_boundary_final.jpg", img_original_boundary_cytoplasm_nuclei)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Histogramy
    vw.histogram_2D_data_range(
        r, "Red channel", "Value", "Frequency", "01_Red", output_path, 0, 255, 1, txt_file=True
    )
    vw.histogram_2D_data_range(
        g, "Green channel", "Value", "Frequency", "02_Green", output_path, 0, 255, 1, txt_file=True
    )
    vw.histogram_2D_data_range(
        b, "Blue channel", "Value", "Frequency", "03_Blue", output_path, 0, 255, 1, txt_file=True
    )

    vw.histogram_2D_data_range(
        h, "H channel", "Value", "Frequency", "04_Hue", output_path, 0, 6.3, 0.1, txt_file=True
    )
    vw.histogram_2D_data_range(
        s,
        "S channel",
        "Value",
        "Frequency",
        "05_Saturation",
        output_path,
        0,
        1,
        0.02,
        txt_file=True,
    )
    vw.histogram_2D_data_range(
        l, "L channel", "Value", "Frequency", "06_Luminance", output_path, 0, 765, 5, txt_file=True
    )


def post_analysis_1(input_path, list_of_directories, N, list_of_csv):
    fontP = FontProperties()
    fontP.set_size("xx-small")

    # zobrazení křivek
    for name in list_of_csv:
        plt.figure(figsize=(12, 8))

        for i in range(N):
            input_path_of_current_data = input_path + f"{list_of_directories[i]}/GRAPHS/{name}.csv"
            data = pd.read_csv(input_path_of_current_data, sep=";", header=None)

            x = data[0]
            y = data[1]

            plt.plot(x, y, label=list_of_directories[i])

        plt.title(name)
        plt.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc="upper left", prop=fontP)
        plt.savefig(f"{input_path}SUMMARY/{name}.png")
        plt.clf()
        plt.close()

    # výpočet integrálu
    for name in list_of_csv:
        for i in range(N):
            input_path_of_current_data = input_path + f"{list_of_directories[i]}/GRAPHS/{name}.csv"
            data = pd.read_csv(input_path_of_current_data, sep=";", header=None)

            x = data[0]
            y = data[1]

            coefficient_of_integral = cow.integral(x, y)

            file = open(f"{input_path}{list_of_directories[i]}/GRAPHS/integral_{name}.txt", "w")
            file.write("coefficient of integral :\n" + str(coefficient_of_integral))
            file.close()

    # výpočet průměru
    for name in list_of_csv:
        plt.figure(figsize=(12, 8))

        for i in range(N):
            input_path_of_current_data = input_path + f"{list_of_directories[i]}/GRAPHS/{name}.csv"
            data = pd.read_csv(input_path_of_current_data, sep=";", header=None)

            if i == 0:
                x_1 = data[0]
                y_1 = data[1]
            else:
                x_1 += data[0]
                y_1 += data[1]

        x_1 = x_1 / N
        y_1 = y_1 / N

        plt.plot(x_1, y_1, label="average_value")
        plt.title(name)
        plt.savefig(f"{input_path}SUMMARY/{name}_average.png")
        plt.clf()
        plt.close()

        file = open(f"{input_path}SUMMARY/{name}.csv", "w")

        for i in range(len(x_1)):
            file.write(str(x_1[i]) + ";" + str(y_1[i]) + "\n")

        file.close()


def threshold_types(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img_0.jpg", img)

    img = cv2.medianBlur(img, 5)
    plt.imsave(f"{output_path}IMG/00_input_img_1.jpg", img)

    img_grayscale_RGB_mean = cw.convert_RGB_to_grayscale(img, width, height)
    img_grayscale_RGB_mean = img_grayscale_RGB_mean.astype(np.uint8)

    ret2, img_grayscale_RGB_mean_1 = cv2.threshold(
        img_grayscale_RGB_mean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    img_grayscale_RGB_mean_2 = cv2.adaptiveThreshold(
        img_grayscale_RGB_mean, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    img_grayscale_RGB_mean_3 = cv2.adaptiveThreshold(
        img_grayscale_RGB_mean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    r, g, b = cw.separate_layers(img, width, height)
    b = b.astype(np.uint8)

    ret2, b_1 = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b_2 = cv2.adaptiveThreshold(b, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    b_3 = cv2.adaptiveThreshold(
        b, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    b_gaussian_blur = cv2.GaussianBlur(b, (5, 5), 0)
    b_gaussian_blur = b_gaussian_blur.astype(np.uint8)

    ret2, b_gaussian_blur_1 = cv2.threshold(
        b_gaussian_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    b_gaussian_blur_2 = cv2.adaptiveThreshold(
        b_gaussian_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    b_gaussian_blur_3 = cv2.adaptiveThreshold(
        b_gaussian_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    plt.imsave(f"{output_path}IMG/01_RGB_mean.jpg", img_grayscale_RGB_mean, cmap="gray")
    plt.imsave(f"{output_path}IMG/02_RGB_mean_otsu.jpg", img_grayscale_RGB_mean_1, cmap="gray")
    plt.imsave(
        f"{output_path}IMG/03_RGB_mean_adaptive_mean.jpg", img_grayscale_RGB_mean_2, cmap="gray"
    )
    plt.imsave(
        f"{output_path}IMG/04_RGB_mean_adaptive_gaussian.jpg", img_grayscale_RGB_mean_3, cmap="gray"
    )

    plt.imsave(f"{output_path}IMG/11_b_channel.jpg", b, cmap="gray")
    plt.imsave(f"{output_path}IMG/12_b_otsu.jpg", b_1, cmap="gray")
    plt.imsave(f"{output_path}IMG/13_b_adaptive_mean.jpg", b_2, cmap="gray")
    plt.imsave(f"{output_path}IMG/14_b_adaptive_gaussian.jpg", b_3, cmap="gray")

    plt.imsave(f"{output_path}IMG/21_b_gaussian_blur.jpg", b_gaussian_blur, cmap="gray")
    plt.imsave(f"{output_path}IMG/22_b_gaussian_blur_otsu.jpg", b_gaussian_blur_1, cmap="gray")
    plt.imsave(
        f"{output_path}IMG/23_b_gaussian_blur_adaptive_mean.jpg", b_gaussian_blur_2, cmap="gray"
    )
    plt.imsave(
        f"{output_path}IMG/24_b_gaussian_blur_adaptive_gaussian.jpg", b_gaussian_blur_3, cmap="gray"
    )


def color_balancing(img, output_path):
    """
    Metoda kterou vužívám nemazat je dobrá
    vezme snímek u udělá na něm všechny metody na color balancing atd který zatím znám
    :param img: snímek
    :param output_path: cesta kam se uloží
    :return: None
    """

    width = img.shape[1]
    height = img.shape[0]

    # Original image
    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # CV2
    wb = cv2.xphoto.createGrayworldWB()
    createGrayworldWB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}IMG/01_CV2_createGrayworldWB.jpg", createGrayworldWB)

    wb = cv2.xphoto.createSimpleWB()
    createSimpleWB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}IMG/02_CV2_createSimpleWB.jpg", createSimpleWB)

    # Article
    article_balance = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/03_article_balance.jpg", article_balance)

    # PIL
    unsharp_mask = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}IMG/04_PIL_unsharp_mask.jpg", unsharp_mask)

    # CCA
    max_white = cca.max_white(img)
    plt.imsave(f"{output_path}IMG/05_cca_max_white.jpg", max_white)

    retinex = cca.retinex(img)
    plt.imsave(f"{output_path}IMG/06_cca_retinex.jpg", retinex)

    automatic_color_equalization = cca.automatic_color_equalization(img)
    plt.imsave(
        f"{output_path}IMG/07_cca_automatic_color_equalization.jpg", automatic_color_equalization
    )

    luminance_weighted_gray_world = cca.luminance_weighted_gray_world(img)
    plt.imsave(
        f"{output_path}IMG/08_cca_luminance_weighted_gray_world.jpg", luminance_weighted_gray_world
    )

    standard_deviation_weighted_grey_world = cca.standard_deviation_weighted_grey_world(img)
    plt.imsave(
        f"{output_path}IMG/09_cca_standard_deviation_weighted_grey_world.jpg",
        standard_deviation_weighted_grey_world,
    )

    standard_deviation_and_luminance_weighted_gray_world = (
        cca.standard_deviation_and_luminance_weighted_gray_world(img)
    )
    plt.imsave(
        f"{output_path}IMG/10_cca_standard_deviation_and_luminance_weighted_gray_world.jpg",
        standard_deviation_and_luminance_weighted_gray_world,
    )


def show_all_norm_separated_color_systems(img_RGB, output_path=""):
    width = img_RGB.shape[1]
    height = img_RGB.shape[0]

    # Převod RGB do ostatních systémů
    img_HSL_A = cw.convert_RGB_to_HSL_A(img_RGB, width, height)
    img_HSL_N = cw.convert_RGB_to_HSL_N(img_RGB, width, height)
    img_XYZ = cw.convert_RGB_to_XYZ(img_RGB, width, height)
    img_Luv = cw.convert_XYZ_to_Luv(img_XYZ, width, height)

    # Separace jednotlivých vrstev
    img_HSL_A_H, img_HSL_A_S, img_HSL_A_L = cw.separate_layers(img_HSL_A, width, height)
    img_HSL_N_H, img_HSL_N_S, img_HSL_N_L = cw.separate_layers(img_HSL_N, width, height)
    img_XYZ_X, img_XYZ_Y, img_XYZ_Z = cw.separate_layers(img_XYZ, width, height)
    img_Luv_L, img_Luv_u, img_Luv_v = cw.separate_layers(img_Luv, width, height)
    img_RGB_R, img_RGB_G, img_RGB_B = cw.separate_layers(img_RGB, width, height)

    # Normalizace dat
    img_HSL_A_H = cw.convert_img_to_norm_img(img_HSL_A_H, "HSL_A_H")
    img_HSL_A_S = cw.convert_img_to_norm_img(img_HSL_A_S, "HSL_A_S")
    img_HSL_A_L = cw.convert_img_to_norm_img(img_HSL_A_L, "HSL_A_L")

    img_HSL_N_H = cw.convert_img_to_norm_img(img_HSL_N_H, "HSL_N_H")
    img_HSL_N_S = cw.convert_img_to_norm_img(img_HSL_N_S, "HSL_N_S")
    img_HSL_N_L = cw.convert_img_to_norm_img(img_HSL_N_L, "HSL_N_L")

    img_XYZ_X = cw.convert_img_to_norm_img(img_XYZ_X, "XYZ_X")
    img_XYZ_Y = cw.convert_img_to_norm_img(img_XYZ_Y, "XYZ_Y")
    img_XYZ_Z = cw.convert_img_to_norm_img(img_XYZ_Z, "XYZ_Z")

    img_Luv_L = cw.convert_img_to_norm_img(img_Luv_L, "Luv_L")
    img_Luv_u = cw.convert_img_to_norm_img(img_Luv_u, "Luv_u")
    img_Luv_v = cw.convert_img_to_norm_img(img_Luv_v, "Luv_v")

    img_RGB_R = cw.convert_img_to_norm_img(img_RGB_R, "RGB_R")
    img_RGB_G = cw.convert_img_to_norm_img(img_RGB_G, "RGB_G")
    img_RGB_B = cw.convert_img_to_norm_img(img_RGB_B, "RGB_B")

    # cesta kam uložit histogramy
    output_path_hist = output_path + "/COLOR_SYSTEMS/"

    # tvorba histogramů
    vw.histogram_2D_data(
        img_HSL_A_H,
        "HSL_A_H",
        "value",
        "frequency",
        "51_HSL_A_H",
        output_path_hist,
        bins=50,
        norm=False,
    )
    vw.histogram_2D_data(
        img_HSL_A_S,
        "HSL_A_S",
        "value",
        "frequency",
        "52_HSL_A_S",
        output_path_hist,
        bins=50,
        norm=False,
    )
    vw.histogram_2D_data(
        img_HSL_A_L,
        "HSL_A_L",
        "value",
        "frequency",
        "53_HSL_A_L",
        output_path_hist,
        bins=50,
        norm=False,
    )

    vw.histogram_2D_data(
        img_HSL_N_H,
        "HSL_N_H",
        "value",
        "frequency",
        "54_HSL_N_H",
        output_path_hist,
        bins=50,
        norm=False,
    )
    vw.histogram_2D_data(
        img_HSL_N_S,
        "HSL_N_S",
        "value",
        "frequency",
        "55_HSL_N_S",
        output_path_hist,
        bins=50,
        norm=False,
    )
    vw.histogram_2D_data(
        img_HSL_N_L,
        "HSL_N_L",
        "value",
        "frequency",
        "56_HSL_N_L",
        output_path_hist,
        bins=50,
        norm=False,
    )

    vw.histogram_2D_data(
        img_XYZ_X, "XYZ_X", "value", "frequency", "57_XYZ_X", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_XYZ_Y, "XYZ_Y", "value", "frequency", "58_XYZ_Y", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_XYZ_Z, "XYZ_Z", "value", "frequency", "59_XYZ_Z", output_path_hist, bins=50, norm=False
    )

    vw.histogram_2D_data(
        img_Luv_L, "Luv_L", "value", "frequency", "60_Luv_L", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_Luv_u, "Luv_u", "value", "frequency", "61_Luv_u", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_Luv_v, "Luv_v", "value", "frequency", "62_Luv_v", output_path_hist, bins=50, norm=False
    )

    vw.histogram_2D_data(
        img_RGB_R, "RGB_R", "value", "frequency", "63_RGB_R", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_RGB_G, "RGB_G", "value", "frequency", "64_RGB_G", output_path_hist, bins=50, norm=False
    )
    vw.histogram_2D_data(
        img_RGB_B, "RGB_B", "value", "frequency", "65_RGB_B", output_path_hist, bins=50, norm=False
    )

    # Ukládání obrázků
    #'''
    plt.imsave(f"{output_path}COLOR_SYSTEMS/01_HSL_A_H.jpg", img_HSL_A_H, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/02_HSL_A_S.jpg", img_HSL_A_S, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/03_HSL_A_L.jpg", img_HSL_A_L, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/04_HSL_N_H.jpg", img_HSL_N_H, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/05_HSL_N_S.jpg", img_HSL_N_S, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/06_HSL_N_L.jpg", img_HSL_N_L, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/07_XYZ_X.jpg", img_XYZ_X, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/08_XYZ_Y.jpg", img_XYZ_Y, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/09_XYZ_Z.jpg", img_XYZ_Z, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/10_Luv_L.jpg", img_Luv_L, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/11_Luv_u.jpg", img_Luv_u, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/12_Luv_v.jpg", img_Luv_v, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/13_RGB_R.jpg", img_RGB_R, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/14_RGB_G.jpg", img_RGB_G, cmap="gray")
    plt.imsave(f"{output_path}COLOR_SYSTEMS/15_RGB_B.jpg", img_RGB_B, cmap="gray")
    #'''

    # jen hledáni extrémů
    """
    # ======================================= #
    print(f'{np.amax(img_HSL_A_H)}')
    print(f'{np.amax(img_HSL_A_S)}')
    print(f'{np.amax(img_HSL_A_L)}')

    print(f'{np.amax(img_HSL_N_H)}')
    print(f'{np.amax(img_HSL_N_S)}')
    print(f'{np.amax(img_HSL_N_L)}')

    print(f'{np.amax(img_RGB_R)}')
    print(f'{np.amax(img_RGB_G)}')
    print(f'{np.amax(img_RGB_B)}')

    print(f'{np.amax(img_XYZ_X)}')
    print(f'{np.amax(img_XYZ_Y)}')
    print(f'{np.amax(img_XYZ_Z)}')

    print(f'{np.amax(img_Luv_L)}')
    print(f'{np.amax(img_Luv_u)}')
    print(f'{np.amax(img_Luv_v)}')
    # --------------------------------------- #
    print(f'{np.amin(img_HSL_A_H)}')
    print(f'{np.amin(img_HSL_A_S)}')
    print(f'{np.amin(img_HSL_A_L)}')

    print(f'{np.amin(img_HSL_N_H)}')
    print(f'{np.amin(img_HSL_N_S)}')
    print(f'{np.amin(img_HSL_N_L)}')

    print(f'{np.amin(img_RGB_R)}')
    print(f'{np.amin(img_RGB_G)}')
    print(f'{np.amin(img_RGB_B)}')

    print(f'{np.amin(img_XYZ_X)}')
    print(f'{np.amin(img_XYZ_Y)}')
    print(f'{np.amin(img_XYZ_Z)}')

    print(f'{np.amin(img_Luv_L)}')
    print(f'{np.amin(img_Luv_u)}')
    print(f'{np.amin(img_Luv_v)}')
    # ======================================= #
    # """


def analysis_3_brown(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img_00.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    img_blur = cv2.blur(img, (5, 5))
    plt.imsave(f"{output_path}IMG/01_blur.jpg", img_blur)

    img_hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    h, s, v = cw.separate_layers(img_hsv)

    plt.imsave(f"{output_path}IMG/02_0_h.jpg", h, cmap="gray")
    plt.imsave(f"{output_path}IMG/02_1_s.jpg", s, cmap="gray")
    plt.imsave(f"{output_path}IMG/02_2_v.jpg", v, cmap="gray")

    vw.histogram_2D_data(
        h, "Hue", "Value", "Frequency", "01_hue", output_path, bins=100, norm=False, txt_file=False
    )
    vw.histogram_2D_data(
        s,
        "Saturation",
        "Value",
        "Frequency",
        "02_saturation",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        v,
        "Value",
        "Value",
        "Frequency",
        "03_value",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )

    kernel = np.ones((3, 3))

    # HSV - brown color definition
    lower_brown_1 = np.array([0, 10, 10])
    upper_brown_1 = np.array([25, 210, 180])

    lower_brown_2 = np.array([125, 10, 10])
    upper_brown_2 = np.array([360, 210, 150])

    mask_1 = cv2.inRange(img_hsv, lower_brown_1, upper_brown_1)
    plt.imsave(f"{output_path}IMG/brown_0_01_mask_1.jpg", mask_1, cmap="gray")

    mask_2 = cv2.inRange(img_hsv, lower_brown_2, upper_brown_2)
    plt.imsave(f"{output_path}IMG/brown_0_01_mask_2.jpg", mask_2, cmap="gray")

    mask_final_brown = np.logical_or(mask_1, mask_2).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_0_02_final_mask.jpg", mask_final_brown, cmap="gray")

    mask_h_opening_brown = cv2.morphologyEx(mask_final_brown, cv2.MORPH_OPEN, kernel)
    plt.imsave(f"{output_path}IMG/brown_0_03_mask_h_opening.jpg", mask_h_opening_brown, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_0_04_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary_brown = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [255, 255, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_0_05_boundary_in_original_image.jpg",
        img_original_with_boundary_brown,
    )

    # ------------------ Zde kód pro analýzu ------------------------

    r, g, b = cw.separate_layers(img)

    # ---------- red
    c_r = 150
    sigma_r = 20

    lower_r = c_r - sigma_r
    upper_r = c_r + sigma_r

    mask_r = cw.convert_grayscale_to_bin_by_range(r, lower_r, upper_r)

    # ---------- green
    c_g = 75
    sigma_g = 20

    lower_g = c_g - sigma_g
    upper_g = c_g + sigma_g

    mask_g = cw.convert_grayscale_to_bin_by_range(g, lower_g, upper_g)

    # ---------- blue
    max_b = 50

    mask_b = cw.convert_grayscale_to_bin(b, max_b)

    # ---------- save
    plt.imsave(f"{output_path}IMG/brown_1_01_r_channel.jpg", r, cmap="gray")
    plt.imsave(f"{output_path}IMG/brown_1_02_g_channel.jpg", g, cmap="gray")
    plt.imsave(f"{output_path}IMG/brown_1_03_b_channel.jpg", b, cmap="gray")

    plt.imsave(f"{output_path}IMG/brown_1_04_mask_r.jpg", mask_r, cmap="gray")
    plt.imsave(f"{output_path}IMG/brown_1_05_mask_g.jpg", mask_g, cmap="gray")
    plt.imsave(f"{output_path}IMG/brown_1_06_mask_b.jpg", mask_b, cmap="gray")

    vw.histogram_2D_data(
        r, "RED", "Value", "Frequency", "04_red", output_path, bins=100, norm=False, txt_file=False
    )
    vw.histogram_2D_data(
        g,
        "GREEN",
        "Value",
        "Frequency",
        "05_green",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )
    vw.histogram_2D_data(
        b,
        "BLUE",
        "Value",
        "Frequency",
        "06_blue",
        output_path,
        bins=100,
        norm=False,
        txt_file=False,
    )

    # ---------- logical and
    mask_final_brown = np.logical_and(mask_r, mask_g, mask_b).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/brown_1_07_final_mask.jpg", mask_final_brown, cmap="gray")

    # ---------- classic steps
    mask_h_opening_brown = cv2.morphologyEx(mask_final_brown, cv2.MORPH_OPEN, kernel)
    plt.imsave(f"{output_path}IMG/brown_1_08_mask_h_opening.jpg", mask_h_opening_brown, cmap="gray")

    img_boundary_brown = sd.get_boundary_4_connected(mask_h_opening_brown, width, height)
    plt.imsave(f"{output_path}IMG/brown_1_09_boundary.jpg", img_boundary_brown, cmap="gray")

    img_original_with_boundary_brown = iw.boundary_to_original_image(
        img, img_boundary_brown, width, height, [255, 255, 255]
    )
    plt.imsave(
        f"{output_path}IMG/brown_1_10_boundary_in_original_image.jpg",
        img_original_with_boundary_brown,
    )


def set_labels_from_HDB_to_img(matrix_XYZ, labels, width, height):
    img_labeled_HDB = np.zeros((height, width))

    for i in range(matrix_XYZ.shape[0]):
        x = int(matrix_XYZ[i][0])
        y = int(matrix_XYZ[i][1])

        img_labeled_HDB[y][x] = labels[i]

    return img_labeled_HDB


def color_balancing_types(img, output_path):
    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    max_white = cca.max_white(img)
    plt.imsave(f"{output_path}IMG/cca_max_white.jpg", max_white)

    wb = cv2.xphoto.createGrayworldWB()
    gray_world_WB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}IMG/gray_world_WB.jpg", gray_world_WB)

    wb = cv2.xphoto.createSimpleWB()
    simple_WB = wb.balanceWhite(img)
    plt.imsave(f"{output_path}IMG/simpleWB.jpg", simple_WB)

    article_balance = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/article_balance.jpg", article_balance)

    deviation_weighted_grey_world = cca.standard_deviation_weighted_grey_world(img)
    plt.imsave(
        f"{output_path}IMG/cca_deviation_weighted_grey_world.jpg", deviation_weighted_grey_world
    )

    deviation_and_luminance_weighted_gray_world = (
        cca.standard_deviation_and_luminance_weighted_gray_world(img)
    )
    plt.imsave(
        f"{output_path}IMG/cca_deviation_and_luminance_weighted_gray_world.jpg",
        deviation_and_luminance_weighted_gray_world,
    )

    retinex = cca.retinex(img)
    plt.imsave(f"{output_path}IMG/cca_retinex.jpg", retinex)

    luminance_weighted_gray_world = cca.luminance_weighted_gray_world(img)
    plt.imsave(
        f"{output_path}IMG/cca_luminance_weighted_gray_world.jpg", luminance_weighted_gray_world
    )

    automatic_color_equalization = cca.automatic_color_equalization(img)
    plt.imsave(
        f"{output_path}IMG/cca_automatic_color_equalization.jpg", automatic_color_equalization
    )

    img_unsharp = iw.unsharp_mask_img(img)
    plt.imsave(f"{output_path}IMG/unsharp_pil.jpg", img_unsharp)


def use_HDB_scan(data):
    METRIC = "manhattan"
    MIN_CLUSTER_SIZE = 5

    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(data)
    labels = clusterer.labels_  # np array labels

    return labels + 2


def get_XYZ_matrix(img_mask, img_grayscale, width, height):
    number_of_pixels = img_mask.sum()

    matrix_XYZ = np.zeros((number_of_pixels, 3))

    y = 0

    for i in range(height):
        for j in range(width):
            if img_mask[i][j] == 1:
                matrix_XYZ[y][0] = j
                matrix_XYZ[y][1] = i
                matrix_XYZ[y][2] = img_grayscale[i][j]  # int(round(img_grayscale[i][j]))
                y = y + 1

    return matrix_XYZ


def identify_image(img):
    height = img.shape[0]
    width = img.shape[1]

    r, g, b = cw.separate_layers(img, width, height)

    img_HSL = cw.convert_RGB_to_HSL_A(img, width, height)
    h, s, l = cw.separate_layers(img_HSL, width, height)

    # Blue channel
    data_list = list(b.ravel())
    hist_info = plt.hist(data_list, bins=BINS_BLUE)
    y_b = np.array(hist_info[0]) / len(data_list)

    # Hue channel
    data_list = list(h.ravel())
    hist_info = plt.hist(data_list, bins=BINS_HUE)
    y_h = np.array(hist_info[0]) / len(data_list)

    # Luminance channel
    data_list = list(l.ravel())
    hist_info = plt.hist(data_list, bins=BINS_LUMINANCE)
    y_l = np.array(hist_info[0]) / len(data_list)

    # Identifikace podle blue channel
    blue_deviation_LL = sum((y_b - LL_BLUE_AVERAGE) ** 2)
    blue_deviation_ML = sum((y_b - ML_BLUE_AVERAGE) ** 2)
    blue_deviation_HL = sum((y_b - HL_BLUE_AVERAGE) ** 2)

    if blue_deviation_LL < blue_deviation_ML and blue_deviation_LL < blue_deviation_HL:
        print(f"Blue_LL")
    if blue_deviation_ML < blue_deviation_LL and blue_deviation_ML < blue_deviation_HL:
        print(f"Blue_ML")
    if blue_deviation_HL < blue_deviation_LL and blue_deviation_HL < blue_deviation_ML:
        print(f"Blue_HL")

    # Identifikace podle hue channel
    hue_deviation_LL = sum((y_h - LL_HUE_AVERAGE) ** 2)
    hue_deviation_ML = sum((y_h - ML_HUE_AVERAGE) ** 2)
    hue_deviation_HL = sum((y_h - HL_HUE_AVERAGE) ** 2)

    if hue_deviation_LL < hue_deviation_ML and hue_deviation_LL < hue_deviation_HL:
        print(f"Hue_LL")
    if hue_deviation_ML < hue_deviation_LL and hue_deviation_ML < hue_deviation_HL:
        print(f"Hue_ML")
    if hue_deviation_HL < hue_deviation_LL and hue_deviation_HL < hue_deviation_ML:
        print(f"Hue_HL")

    # Identifikace podle luminance channel
    luminance_deviation_LL = sum((y_l - LL_LUMINANCE_AVERAGE) ** 2)
    luminance_deviation_ML = sum((y_l - ML_LUMINANCE_AVERAGE) ** 2)
    luminance_deviation_HL = sum((y_l - HL_LUMINANCE_AVERAGE) ** 2)

    if (
        luminance_deviation_LL < luminance_deviation_ML
        and luminance_deviation_LL < luminance_deviation_HL
    ):
        print(f"Luminance_LL")
    if (
        luminance_deviation_ML < luminance_deviation_LL
        and luminance_deviation_ML < luminance_deviation_HL
    ):
        print(f"Luminance_ML")
    if (
        luminance_deviation_HL < luminance_deviation_LL
        and luminance_deviation_HL < luminance_deviation_ML
    ):
        print(f"Luminance_HL")


def TODO():
    print("Hello home")
    # TODO color balancing
    # TODO HSL - H otsu
    # TODO erode dilate -> jádra snad a centroids
    # TODO průměry a pak postupně rozšiřovat udělat souřadnice jader cyto a pak +- něco v cyto patří k jádru
    # TODO shape descriptory uprav hranice někde jsi to už dělal hledej flooding cytoplasm
    # TODO histogramy v cytoplasmy
    # TODO pro jádra histogramy
    # TODO průměr ve všech oblastech
    #
    # TODO histogramy normalizovat a nakreslit do sebe

    # Cesta ke snímkům
    # DATA_PATH = '../Images/01_test/'

    # Názvy jednotlivých snímků ----> test
    # IMAGES = ['img_1.jpg','img_2.jpg','img_3.jpg','img_4.jpg','img_5.jpg','img_6.jpg','img_7.jpg']  # Všechny snímky
    # IMAGES = ['img_1.jpg','img_2.jpg','img_6.jpg','img_7.jpg']  # Snímky jen s velkým přiblížením
    # IMAGES = ['img_7.jpg']  # ten nejmenší snímek na zkoušky
    #


def img_processing_1(img, output_path):
    """
    metoda používá color_balancing z článku
    na určení cytoplazmy má multi otsu

    je zde chyba používám red channel místo b channel u multi_otsu (to i ve všech předchozích verzích)

    kod by chtěl upravit, ale jen pokud to bude potřeba HSL -> multi otsu a nahradit R za B

    :param img: snímek
    :param output_path: cesta kde se uloží výsledky
    :return: None
    """
    # otsu combined - nejlepší verze

    width = img.shape[1]
    height = img.shape[0]

    plt.imsave(f"{output_path}IMG/00_input_img.jpg", img)

    # ------------------ Zde kód pro analýzu ------------------------

    # Preprocessing
    RGB_balanced = img
    # RGB_balanced = iw.color_balancing(img, width, height).astype(np.uint8)
    plt.imsave(f"{output_path}IMG/01_RGB_Balanced.jpg", RGB_balanced)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce jader
    r, g, b = cw.separate_layers(RGB_balanced)

    plt.imsave(f"{output_path}IMG/01_zr.jpg", r, cmap="gray")
    plt.imsave(f"{output_path}IMG/01_zg.jpg", g, cmap="gray")
    plt.imsave(f"{output_path}IMG/01_zb.jpg", b, cmap="gray")

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b)
    plt.imsave(f"{output_path}IMG/02_blue_channel_otsu.jpg", b_bin_otsu, cmap="gray")

    b_bin_otsu_morp = iw.close_holes_remove_noise(b_bin_otsu)
    plt.imsave(
        f"{output_path}IMG/03_blue_channel_otsu_noise_removed.jpg", b_bin_otsu_morp, cmap="gray"
    )

    img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    plt.imsave(f"{output_path}IMG/04_nuclei_labeled.jpg", img_labeled_nuclei, cmap="jet")

    img_nuclei_boundary = sd.get_boundary_4_connected(img_labeled_nuclei, width, height)
    img_nuclei_boundary_bin = cw.convert_labeled_to_bin(img_nuclei_boundary)
    plt.imsave(f"{output_path}IMG/05_nuclei_boundary.jpg", img_nuclei_boundary_bin, cmap="gray")

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_nuclei_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/06_boundary_in_original_img.jpg", img_boundary_in_original)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Detekce obalu
    img_HSL = cw.convert_RGB_to_HSL_A(RGB_balanced, width, height)
    h, s, l = cw.separate_layers(img_HSL)

    h_norm = cw.convert_img_to_norm_img(h, "HSL_A_H")

    center = 0.66
    sigma = 0.07
    lower = center - sigma
    upper = center + sigma

    thresholds = filters.threshold_multiotsu(r)

    regions = np.digitize(r, bins=thresholds)
    plt.imsave(f"{output_path}IMG/02_multi_otsu.jpg", regions)

    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    plt.imsave(f"{output_path}IMG/07_bin_multi_otsu_cyto.jpg", bin_cyto_nuclei, cmap="gray")

    img_h_bin_morp = iw.close_holes_remove_noise(bin_cyto_nuclei)
    plt.imsave(f"{output_path}IMG/08_HSL_H_MORP.jpg", img_h_bin_morp, cmap="gray")

    img_h_labeled_cytoplasm, nr_cytoplasm = mh.label(img_h_bin_morp)
    plt.imsave(
        f"{output_path}IMG/09_HSL_H_labeled_cytoplasm.jpg", img_h_labeled_cytoplasm, cmap="jet"
    )

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_h_labeled_cytoplasm, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    plt.imsave(
        f"{output_path}IMG/10_cytoplasm_boundary.jpg", img_cytoplasm_boundary_bin, cmap="gray"
    )

    img_boundary_in_original = iw.boundary_to_original_image(
        img, img_cytoplasm_boundary, width, height, [255, 0, 0]
    )
    plt.imsave(f"{output_path}IMG/11_0_boundary_in_original_img.jpg", img_boundary_in_original)
    """
    # ---------------------------------------------------------------------------------------------------------------- #
    # Spojení
    cytoplasm_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_nuclei_boundary)
    plt.imsave(f'{output_path}IMG/11_cytoplasm_nuclei_boundary.jpg', cytoplasm_nuclei_boundary, cmap='gray')

    boundary_original_img = iw.boundary_to_original_image(img, cytoplasm_nuclei_boundary, width, height)
    plt.imsave(f'{output_path}IMG/12_cytoplasm_nuclei_boundary.jpg', boundary_original_img)

    img_cytoplasm_nuclei = iw.flooding_cytoplasm(img_h_labeled_cytoplasm, img_labeled_nuclei, width, height)
    plt.imsave(f'{output_path}IMG/13_flooding_cytoplasm.jpg', img_cytoplasm_nuclei, cmap='jet')

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_cytoplasm_nuclei, width, height)
    plt.imsave(f'{output_path}IMG/14_flooding_cytoplasm_boundary.jpg', img_cytoplasm_boundary, cmap='jet')

    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)
    cytoplasm_separated_nuclei_boundary = img_cytoplasm_boundary_bin + img_nuclei_boundary_bin
    cytoplasm_separated_nuclei_boundary = cw.convert_labeled_to_bin(cytoplasm_separated_nuclei_boundary)
    plt.imsave(f'{output_path}IMG/15_cytoplasm_nuclei_boundary.jpg', cytoplasm_separated_nuclei_boundary, cmap='gray')

    boundary_original_img = iw.boundary_to_original_image(img, cytoplasm_separated_nuclei_boundary, width, height)
    plt.imsave(f'{output_path}IMG/16_cytoplasm_nuclei_boundary_original.jpg', boundary_original_img)

    img_only_cytoplasm = iw.get_cytoplasm_only(b_bin_otsu_morp, img_cytoplasm_nuclei)
    img_only_cytoplasm_bin = cw.convert_labeled_to_bin(img_only_cytoplasm)
    plt.imsave(f'{output_path}IMG/17_only_cytoplasm_bin.jpg', img_only_cytoplasm_bin, cmap='gray')

    img_only_cytoplasm_threshold, img_b_in_cytoplasm_mask = iw.threshold_in_mask(b, img_only_cytoplasm_bin, width, height)
    plt.imsave(f'{output_path}IMG/18_only_cytoplasm_in_blue_channel.jpg', img_b_in_cytoplasm_mask, cmap='gray')
    plt.imsave(f'{output_path}IMG/19_only_cytoplasm_threshold_in_mask.jpg', img_only_cytoplasm_threshold, cmap='gray')

    img_cytoplasm_and_nuclei_bin = img_only_cytoplasm_threshold + b_bin_otsu_morp
    plt.imsave(f'{output_path}IMG/20_cytoplasm_and_nuclei_bin.jpg', img_cytoplasm_and_nuclei_bin, cmap='gray')

    img_cytoplasm_and_nuclei_labeled = img_cytoplasm_and_nuclei_bin * img_cytoplasm_nuclei
    plt.imsave(f'{output_path}IMG/21_cytoplasm_and_nuclei_labeled.jpg', img_cytoplasm_and_nuclei_labeled, cmap='jet')

    cytoplasm_sizes = mh.labeled.labeled_size(img_cytoplasm_and_nuclei_labeled)
    cytoplasm_sizes[0] = 0
    number_of_cells = cytoplasm_sizes.shape[0]

    coordinates_cytoplasm = sd.get_coordinates_of_pixels(img_cytoplasm_and_nuclei_labeled, cytoplasm_sizes, number_of_cells,width, height)

    img_repaired = iw.cell_repair(coordinates_cytoplasm, cytoplasm_sizes, number_of_cells, width, height)
    plt.imsave(f'{output_path}IMG/22_cytoplasm_and_nuclei_repaired.jpg', img_repaired, cmap='jet')

    img_cytoplasm_boundary = sd.get_boundary_4_connected(img_repaired, width, height)
    img_cytoplasm_boundary_bin = cw.convert_labeled_to_bin(img_cytoplasm_boundary)

    img_cytoplasm_nuclei_boundary = cw.convert_labeled_to_bin(img_nuclei_boundary_bin + img_cytoplasm_boundary_bin)
    img_boundary_in_original = iw.boundary_to_original_image(img, img_cytoplasm_nuclei_boundary, width, height)
    plt.imsave(f'{output_path}IMG/23_boundary_in_original_img.jpg', img_boundary_in_original)

    img_only_cytoplasm_with_nuclei = iw.get_cytoplasm_which_have_nuclei(img_repaired, nr_nuclei)
    plt.imsave(f'{output_path}IMG/24_only_cytoplasm_with_nuclei.jpg', img_only_cytoplasm_with_nuclei, cmap='jet')

    img_only_cytoplasm_with_nuclei_removed_small = iw.remove_small_regions(cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei),is_bin = True)
    plt.imsave(f'{output_path}IMG/25_only_cytoplasm_with_nuclei_without_small_reg.jpg', img_only_cytoplasm_with_nuclei_removed_small,cmap='jet')

    img_only_cytoplasm_with_nuclei_removed_small = cw.convert_labeled_to_bin(img_only_cytoplasm_with_nuclei_removed_small) * img_repaired
    plt.imsave(f'{output_path}IMG/26_only_cytoplasm_with_nuclei_without_small_reg_repaired.jpg', img_only_cytoplasm_with_nuclei_removed_small, cmap='jet')

    boundary_img_only_cytoplasm_with_nuclei_removed_small = sd.get_boundary_4_connected(img_only_cytoplasm_with_nuclei_removed_small, width, height)
    plt.imsave(f'{output_path}IMG/27_only_cytoplasm_with_nuclei_boundary.jpg', boundary_img_only_cytoplasm_with_nuclei_removed_small, cmap='jet')

    only_cytoplasm_which_have_nuclei_but_not_included = iw.get_remove_nuclei_from_cytoplasm(img_only_cytoplasm_with_nuclei_removed_small, img_labeled_nuclei, width, height)
    plt.imsave(f'{output_path}IMG/28_only_cytoplasm_which_have_nuclei.jpg', only_cytoplasm_which_have_nuclei_but_not_included, cmap='jet')

    img_original_boundary_cytoplasm_nuclei = iw.two_boundary_types_to_original_image(img, img_nuclei_boundary,boundary_img_only_cytoplasm_with_nuclei_removed_small,width, height)
    plt.imsave(f'{output_path}IMG/29_boundary_final.jpg', img_original_boundary_cytoplasm_nuclei)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Histogramy
    vw.histogram_2D_data_range(r, 'Red channel', 'Value', 'Frequency', '01_Red', output_path, 0,255,1, txt_file=True)
    vw.histogram_2D_data_range(g, 'Green channel', 'Value', 'Frequency', '02_Green', output_path, 0,255,1, txt_file=True)
    vw.histogram_2D_data_range(b, 'Blue channel', 'Value', 'Frequency', '03_Blue', output_path, 0,255,1, txt_file=True)

    vw.histogram_2D_data_range(h, 'H channel', 'Value', 'Frequency', '04_Hue', output_path, 0,6.3,0.1, txt_file=True)
    vw.histogram_2D_data_range(s, 'S channel', 'Value', 'Frequency', '05_Saturation', output_path, 0,1,0.02, txt_file=True)
    vw.histogram_2D_data_range(l, 'L channel', 'Value', 'Frequency', '06_Luminance', output_path, 0,765,5, txt_file=True)
    """


def reidentifikator(bunky, n):
    a = np.zeros(n)

    file = open(bunky, "r")
    lines = file.readlines()

    for line in lines:
        number = int(line)
        a[number] = 1

    output = open(f"cells.txt", "w")

    for i in range(n):
        output.write(f"{a[i]}")

    output.close()
