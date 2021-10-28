import os
from collections import namedtuple

import bigfish.detection as bf_detection
import bigfish.stack as bf_stack
import bigfish.plot as bf_plot
import numpy as np
import skimage
from skimage import exposure, io, morphology
from skimage.filters import threshold_otsu

BoundingBox = namedtuple("BoundingBox", ["ymin", "ymax", "xmin", "xmax"])


def read_bit_img(img_file, bits=12):
    """Read an image and return as a 16-bit image."""
    img = exposure.rescale_intensity(
        io.imread(img_file),
        in_range=(0, 2 ** (bits) - 1)
        #         out_range=(0, )
    )
    return skimage.img_as_uint(img)


def select_signal(image, p_in_focus=0.75, margin_width=10):
    """
    Generate bounding box of FISH image to select on areas where signal is present.

    Parameters
    ----------
    image : np.ndarray
        3D FISH image
    p_in_focus : float, optional
        Percent of in-focus slices to retain for 2D projection, by default 0.75.
    margin_width : int, optional
        Number of pixels to pad selection by. Default is 10.

    Returns
    -------
    namedtuple
        minimum and maximum coordinate values of the bounding box in the xy plane
    """
    image = image.astype(np.uint16)
    focus = bf_stack.compute_focus(image)
    selected = bf_stack.in_focus_selection(image, focus, p_in_focus)
    projected_2d = bf_stack.maximum_projection(selected)
    foreground = np.where(projected_2d > threshold_otsu(projected_2d))
    limits = BoundingBox(
        foreground[0].min() - margin_width,
        foreground[0].max() + margin_width,
        foreground[1].min() - margin_width,
        foreground[1].max() + margin_width,
    )
    return limits


def crop_to_selection(img, bbox):
    """
    Crop image to selection defined by bounding box.

    Crops a 3D image to specified x and y coordinates.
    Parameters
    ----------
    img : np.ndarray
        3Dimensional image to crop
    bbox : namedtuple
        Tuple defining minimum and maximum coordinates for x and y planes.

    Returns
    -------
    np.ndarray
        3D image cropped to the specified selection.
    """
    return img[:, bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax]


def count_spots_in_labels(spots, labels):
    """
    Count the number of RNA molecules in specified labels.

    Parameters
    ----------
    spots : np.ndarray
        Coordinates in original image where RNA molecules were detected.
    labels : np.ndarray
        Integer array of same shape as `img` denoting regions to interest to quantify.
        Each separate region should be uniquely labeled.

    Returns
    -------
    dict
        dictionary containing the number of molecules contained in each labeled region.
    """
    assert spots.shape[1] == len(labels.shape)
    n_labels = np.unique(labels) - 1  # subtract one for backgroudn
    counts = {i: 0 for i in range(1, n_labels + 1)}
    for each in spots:
        if len(each) == 3:
            cell_label = labels[each[0], each[1], each[2]]
        else:
            cell_label = labels[each[0], each[1]]
        if cell_label != 0:
            counts[cell_label] += 1
    return counts


def count_spots(
    img,
    scale=True,
    scale_percentile=99.9,
    voxel_size_z=None,
    voxel_size_yx=0.67,
    psf_yx=1,
    whitehat=True,
    whitehat_selem=None,
    smooth_method="log",
    smooth_sigma=1,
    decompose_alpha=0.7,
    decompose_beta=1,
    decompose_gamma=5,
    verbose=False,
):
    """
    Count the number of molecules in an smFISH image

    Parameters
    ----------
    img : np.ndarray
        Image in which to perform molecule counting
    scale : boolean, optional
        Whether to scale image using contrast stretching, but default True.
    scale_percentile : float, optional
        Percentile to scale to using contrast stretching. Values should fall between 0
        and 100. Default is 99.9.
    voxel_size_z : float, optional
        The number of nanometers between each z-slice, by default None and two-dimensional
        quantification is assumed.
    voxel_size_yx : float, optional
        The space occupied by each pixel in nanometers, by default 0.67.
    psf_yx : int, optional
        Theoretical size of the gaussian point-spread function emitted by a spot in the xy plane.
        By default 1.
    whitehat : bool, optional
        Whether to perform white tophat filtering prior to image de-noising, by default True
    whitehat_selem : [int, np.ndarray], optional
        Structuring element to use for white tophat filtering.
    smooth_method : str, optional
        Method to use for image de-noising. Possible values are "log" and "gaussian" for
        Laplacian of Gaussians and Gaussian background subtraction, respectively. By default "log".
    smooth_sigma : [int, np.ndarray], optional
        Sigma value to use for smoothing function, by default 1
    decompose_alpha : float, optional
        Intensity percentile used to compute the reference spot, between 0 and 1.
        By default 0.7. For more information, see:
        https://big-fish.readthedocs.io/en/stable/detection/dense.html
    decompose_beta : int, optional
        Multiplicative factor for the intensity threshold of a dense region,
        by default 1. For more information, see:
        https://big-fish.readthedocs.io/en/stable/detection/dense.html
    decompose_gamma : int, optional
        Multiplicative factor use to compute a gaussian scale, by default 5.
        For more information, see:
        https://big-fish.readthedocs.io/en/stable/detection/dense.html
    verbose : bool, optional
        Whether to verbosely print results and progress.

    Returns
    -------
    (np.ndarray)
        np.ndarray: positions of all identified mRNA molecules.

    """
    if verbose:
        print("Cropping image to only include areas with signal...")
    limits = select_signal(img)
    cropped_img = crop_to_selection(img, limits)
    if scale:
        cropped_img = bf_stack.rescale(
            cropped_img, stretching_percentile=scale_percentile
        )
    if whitehat:
        cropped_img = morphology.white_tophat(cropped_img, whitehat_selem)
    if smooth_method == "log":
        smoothed = bf_stack.log_filter(cropped_img, sigma=smooth_sigma)
    elif smooth_method == "gaussian":
        smoothed = bf_stack.remove_background_gaussian(cropped_img, sigma=smooth_sigma)
    else:
        raise ValueError(f"Unsupported background filter: {smooth_method}")
    psf_z = None
    if voxel_size_z is not None:
        psf_z = psf_yx * voxel_size_z / voxel_size_yx
        sigma_z, sigma_yx, sigma_yx = bf_stack.get_sigma(
            voxel_size_z, voxel_size_yx, psf_z, psf_yx
        )
        if verbose:
            print(
                "standard deviation of the PSF (z axis): {:0.3f} pixels".format(sigma_z)
            )
    else:
        sigma_yx, sigma_yx = bf_stack.get_sigma(
            voxel_size_z, voxel_size_yx, psf_z, psf_yx
        )
        sigma_z = None
    if verbose:
        print(
            "standard deviation of the PSF (yx axis): {:0.3f} pixels".format(sigma_yx)
        )
    spots, threshold = bf_detection.detect_spots(
        smoothed,
        return_threshold=True,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx,
        psf_z=psf_z,
        psf_yx=psf_yx,
    )
    try:
        (
            spots_post_decomposition,
            dense_regions,
            reference_spot,
        ) = bf_detection.decompose_dense(
            smoothed,
            spots,
            voxel_size_z,
            voxel_size_yx,
            psf_z,
            psf_yx,
            alpha=decompose_alpha,  # alpha impacts the number of spots per candidate region
            beta=decompose_beta,  # beta impacts the number of candidate regions to decompose
            gamma=decompose_gamma,  # gamma the filtering step to denoise the image
        )
    except RuntimeError:
        print("decomposition failed, using originally identified spots")
        spots_post_decomposition = spots
    if verbose:
        print(f"detected spots before decomposition: {spots.shape[0]}")
        print(
            f"detected spots after decomposition: {spots_post_decomposition.shape[0]}"
        )
        bf_plot.plot_reference_spot(reference_spot, rescale=True)
    # spots are in cropped coordinates, shift back to original
    spots_post_decomposition[:, 1] += limits.ymin
    spots_post_decomposition[:, 2] += limits.xmin

    return spots_post_decomposition
