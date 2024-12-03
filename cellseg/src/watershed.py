from pathlib import Path

import colorcorrect.algorithm as cca
import mahotas as mh
import numpy as np
from rich import print
from rich.panel import Panel
from scipy.ndimage import gaussian_filter
from skimage import filters
from sklearn.metrics import f1_score

import cellseg.src.convert_worker as cw
import cellseg.src.image_worker as iw


def watershed_cytoplasm(
    img_path: Path,
    mask_size: int,
    iterations: int,
    min_size: int,
    slope: int,
    limit: int,
    samples: int,
    subwidth: int,
    subheight: int,
    sigma: float,
    label_path: Path,
) -> np.ndarray:
    """watershed

    If label is provided, it will calculate the f1 score.

    Args:
        img (np.ndarray): Original image
        mask_size (int): Mask for noise removal
        iterations (int): Number of iterations for noise removal
        min_size (int): Minimum size of region to keep
        slope (int): Slope for color correction
        limit (int): Limit for color correction
        samples (int): Samples for color correction
        subwidth (int): Subwidth for color correction
        subheight (int): Subheight for color correction
        sigma (float): Sigma for gaussian filter
        label (np.ndarray, optional): Ground truth labels. Defaults to None.

    Returns:
        np.ndarray:

    """
    print(
        Panel(
            f"Segmentation started!\n\n" f"Image: {img_path.stem}\n",
            title="Cytoplasm segmentation",
            border_style="yellow",
            expand=False,
        )
    )

    img = mh.imread(img_path)
    label = np.load(label_path)

    img_c_corrected = cca.automatic_color_equalization(
        img, slope=slope, limit=limit, samples=samples
    )

    img_blurred = gaussian_filter(img_c_corrected, sigma=sigma)

    RGB_balanced = cca.luminance_weighted_gray_world(
        img_blurred, subwidth=subwidth, subheight=subheight
    )
    r, g, b = RGB_balanced[:, :, 0], RGB_balanced[:, :, 1], RGB_balanced[:, :, 2]

    thresholds = filters.threshold_multiotsu(g, classes=3)
    regions = np.digitize(g, bins=thresholds)
    bin_cyto_nuclei = cw.convert_labeled_to_bin(regions, background=2)
    bin_cyto_nuclei = iw.close_holes_remove_noise(
        bin_cyto_nuclei, mask_size=mask_size, iterations=iterations
    )

    img_labeled_cytoplasm, nr_cytoplasm = mh.label(bin_cyto_nuclei)
    img_labeled_cytoplasm = iw.remove_small_regions(img_labeled_cytoplasm, min_size=min_size)
    cytoplasm = mh.labeled.remove_bordering(img_labeled_cytoplasm)

    if label.any():
        f1 = f1_score(label.flatten(), cytoplasm.flatten(), average="micro")
        return cytoplasm, f1

    return cytoplasm, None


def watershed_nucleus(
    img_path: Path,
    mask_size: int,
    iterations: int,
    min_size: int,
    radius: int,
    percent: float,
    threshold: float,
    label_path: Path,
) -> np.ndarray:
    """watershed

    If label is provided, it will calculate the f1 score.

    Args:
        img (np.ndarray): Original image
        mask_size (int): Mask for noise removal
        iterations (int): Number of iterations for noise removal
        min_size (int): Minimum size of region to keep
        radius (int): Radius for unsharp mask
        percent (float): Percent for unsharp mask
        threshold (float): Threshold for unsharp mask
        label (np.ndarray, optional): Ground truth labels. Defaults to None.

    Returns:
        np.ndarray:

    """
    print(
        Panel(
            f"Segmentation started!\n\n" f"Image: {img_path.stem}\n",
            title="Nucleus segmentation",
            border_style="yellow",
            expand=False,
        )
    )

    img = mh.imread(img_path)
    label = np.load(label_path)

    img_unsharp = iw.unsharp_mask_img(img, radius=radius, percent=percent, threshold=threshold)
    r1, g1, b1 = img_unsharp[:, :, 0], img_unsharp[:, :, 1], img_unsharp[:, :, 2]

    b_bin_otsu = cw.convert_grayscale_to_bin_otsu(b1)
    b_bin_otsu_morp = iw.close_holes_remove_noise(
        b_bin_otsu, mask_size=mask_size, iterations=iterations
    )

    # img_labeled_nuclei, nr_nuclei = mh.label(b_bin_otsu_morp)
    img_labeled_nuclei = iw.remove_small_regions(b_bin_otsu_morp, min_size=min_size)
    # img_labeled_nuclei = mh.labeled.remove_bordering(img_labeled_nuclei)

    if label.any():
        f1 = f1_score(label.flatten(), img_labeled_nuclei.flatten())
        return img_labeled_nuclei, f1

    return img_labeled_nuclei, None
