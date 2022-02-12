import os
from collections import namedtuple

import numpy as np
import skimage
from skimage import exposure, io, morphology, measure
from skimage.filters import threshold_otsu

import bigfish.detection as bf_detection
import bigfish.stack as bf_stack
import bigfish.plot as bf_plot

BoundingBox = namedtuple("BoundingBox", ["ymin", "ymax", "xmin", "xmax"])


def normalize_zstack(z_stack, bits):
    """Normalize z-slices in a 3D image using contrast stretching

    Parameters
    ----------
    z_stack : numpy.ndarray
        3 dimensional confocal FISH image.
    bits : int
        Bit depth of image.

    Returns
    -------
    numpy.ndarray
        Z-corrected image with each slice minimum and maximum matched
    """
    out = np.array(
        [
            exposure.rescale_intensity(
                x, in_range=(0, 2 ** bits - 1), out_range=(z_stack.min(), z_stack.max())
            )
            for x in z_stack
        ]
    )
    return skimage.img_as_uint(exposure.rescale_intensity(out))


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


def preprocess_image(
    img, smooth_method="gaussian", sigma=7, whitehat=True, selem=None, stretch=99.99
):
    scaled = exposure.rescale_intensity(
        img, in_range=tuple(np.percentile(img, [0, stretch]))
    )
    if smooth_method == "log":
        smooth_func = bf_stack.log_filter
        to_smooth = bf_stack.cast_img_float64(scaled)
    elif smooth_method == "gaussian":
        smooth_func = bf_stack.remove_background_gaussian
        to_smooth = bf_stack.cast_img_uint16(scaled)
    else:
        raise ValueError(f"Unsupported background filter: {smooth_method}")
    if whitehat:
        f = lambda x, s: morphology.white_tophat(smooth_func(x, s), selem)
    else:
        f = lambda x, s: smooth_func(x, s)
    smoothed = np.stack([f(img_slice, sigma) for img_slice in to_smooth])
    return bf_stack.cast_img_float64(np.stack(smoothed))


def quantify_expression(
    img,
    labels,
    voxel_size_nm,
    dot_radius_nm,
    whitehat=True,
    whitehat_selem=None,
    smooth_method="gaussian",
    smooth_sigma=1,
    decompose_alpha=0.5,
    decompose_beta=1,
    decompose_gamma=5,
    bits=12,
    verbose=False,
):
    """
    Count the number of molecules in an smFISH image

    Parameters
    ----------
    img : np.ndarray
        Image in which to perform molecule counting
    labels : np.ndarray
        Integer array of same shape as `img` denoting regions to interest to quantify.
        Each separate region should be uniquely labeled.
    voxel_size_nm : tuple(float, int)
        Physical dimensions of each voxel in ZYX order.
    dot_radius_nm : tuple(float, int)
        Physical size of expected dots.
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
    bits : int, optional
        Bit depth of original image. Used for scaling image while maintaining
        ob
    verbose : bool, optional
        Whether to verbosely print results and progress.

    Returns
    -------
    (np.ndarray, dict)
        np.ndarray: positions of all identified mRNA molecules.
        dict: dictionary containing the number of molecules contained in each labeled region.
    """
    limits, __ = select_signal(img)
    cropped_img = crop_to_selection(img, limits)
    smoothed = preprocess_image(
        cropped_img, smooth_method, smooth_sigma, whitehat, whitehat_selem, 99.99
    )

    cropped_labels = crop_to_selection(labels, limits)
    n_labels = len(np.unique(labels)) - 1  # subtract one for background

    if verbose:
        spot_radius_px = bf_detection.get_object_radius_pixel(
            voxel_size_nm=voxel_size_nm, object_radius_nm=dot_radius_nm, ndim=3
        )
        print("spot radius (z axis): {:0.3f} pixels".format(spot_radius_px[0]))
        print("spot radius (yx plan): {:0.3f} pixels".format(spot_radius_px[-1]))
    spots, threshold = bf_detection.detect_spots(
        smoothed,
        return_threshold=True,
        voxel_size=voxel_size_nm,
        spot_radius=dot_radius_nm,
    )
    if verbose:
        print(f"{spots.shape[0]} spots detected...")
        print("plotting threshold optimization for spot detection...")
        bf_plot.plot_elbow(
            smoothed,
            voxel_size=voxel_size_nm,
            spot_radius=dot_radius_nm,
        )
    decompose_cast = {
        "gaussian": bf_stack.cast_img_uint16,
        "log": bf_stack.cast_img_float64,
    }
    try:
        (
            spots_post_decomposition,
            dense_regions,
            reference_spot,
        ) = bf_detection.decompose_dense(
            smoothed,
            spots,
            voxel_size=voxel_size_nm,
            spot_radius=dot_radius_nm,
            alpha=decompose_alpha,  # alpha impacts the number of spots per candidate region
            beta=decompose_beta,  # beta impacts the number of candidate regions to decompose
            gamma=decompose_gamma,  # gamma the filtering step to denoise the image
        )
        if verbose:
            print(
                f"detected spots before decomposition: {spots.shape[0]}\n"
                f"detected spots after decomposition: {spots_post_decomposition.shape[0]}\n"
                f"shape of reference spot for decomposition: {reference_spot.shape}"
            )
            bf_plot.plot_reference_spot(reference_spot, rescale=True)
    except RuntimeError:
        print("decomposition failed, using originally identified spots")
        spots_post_decomposition = spots
    each = spots[0]
    counts = {i: 0 for i in range(1, n_labels + 1)}
    for each in spots_post_decomposition:
        if len(each) == 3:
            cell_label = cropped_labels[each[0], each[1], each[2]]
        else:
            cell_label = cropped_labels[each[0], each[1]]
        if cell_label != 0:
            counts[cell_label] += 1
    intensities = {i: 0 for i in range(1, n_labels + 1)}
    z_normed_smooth = (smoothed - smoothed.mean()) / smoothed.std()
    for region in measure.regionprops(cropped_labels, z_normed_smooth):
        intensities[region.label] = region.mean_intensity
    # spots are in cropped coordinates, shift back to original
    spots_post_decomposition[:, 1] += limits.ymin
    spots_post_decomposition[:, 2] += limits.xmin
    return spots_post_decomposition, {"counts": counts, "intensity": intensities}


def get_channel_index(channels, channel):
    """Get numerical index for channel of interest by searching a string of ';' separated channel names"""
    channel_index = [
        i for i, x in enumerate(channels.split(";")) if x.lower() == channel.lower()
    ][0]
    return channel_index


if __name__ == "__main__":
    import h5py
    import pandas as pd
    from aicsimageio import AICSImage

    try:
        snakemake
    except NameError:
        snakemake = None
    if snakemake is not None:
        img = AICSImage(snakemake.input["image"])
        labels = np.array(h5py.File(snakemake.input["labels"], "r")["image"])
        print(f"{len(np.unique(labels) - 1)} labels detected...")
        start = snakemake.params["z_start"]
        stop = snakemake.params["z_end"]
        gene_params = snakemake.params["gene_params"]
        genes = gene_params.keys()
        channels = [get_channel_index(snakemake.params["channels"], x) for x in genes]
        fish_counts = {}
        embryo = snakemake.wildcards["embryo"]
        for (gene, fish_channel) in zip(genes, channels):
            fish_data = img.get_image_data("ZYX", C=fish_channel)[start:stop, :, :]
            spots, quant = quantify_expression(
                fish_data,
                labels,
                voxel_size_nm=[x * 1000 for x in img.physical_pixel_sizes],
                dot_radius_nm=gene_params[gene]["radius"],
                whitehat=True,
                smooth_method="gaussian",
                smooth_sigma=5,
                verbose=False,
                bits=img.metadata["attributes"].bitsPerComponentSignificant,
            )
            fish_counts[f"{gene}_spots"] = quant["counts"]
            fish_counts[f"{gene}_intensity"] = quant["intensity"]
        exprs_df = pd.DataFrame.from_dict(fish_counts)
        exprs_df["embryo"] = embryo
        exprs_df.to_csv(snakemake.output["csv"])
