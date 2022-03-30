import h5py
import numpy as np
from scipy import signal, spatial
from scipy import ndimage as ndi
from scipy.spatial.qhull import QhullError
from skimage import filters, measure, morphology, segmentation

import logging


def smooth(x, window_len=11, window="hanning"):
    """
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    Source
    ------
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[(window_len // 2) : -(window_len // 2 - 1)]


def assign_to_label(src, region, slc, new_label):
    """Assign image region to label

    Parameters
    ----------
    src : numpy.ndarray
        Label-containing image.
    region : skimage.measure.RegionProperties
        Region to label
    slc : slice
        Slice defining which part of the region to label. If None, selects
        entire region
    new_label : int
        New label to set
    """
    if slc is None:
        slc = tuple([slice(None)] * len(region.image.shape))
    src[region.slice][slc][region.image[slc]] = new_label


def get_z_regions(region):
    """Break 3D region into separate 2D regions.

    Parameters
    ----------
    region : skimage.measure.RegionProperties
        3D regino to break down

    Returns
    -------
    list[skimage.measure.RegionProperties]
        Separte 2D regions comprising `region`
    """
    return [measure.regionprops(x.astype(int))[0] for x in region.image]


def split_labels_by_area(labels, region, n_labels):
    """Split a label based on local minimas of measured areas.

    Splits a long label along the z-axis at points of locally minimal area.
    Necessary to separate stacked PMCs.

    Parameters
    ----------
    labels : np.ndarray
        3D image containing labelled regions
    region : skimage.measure.RegionProperties
        Long region to split
    n_labels : int
        The total number of objects found in `labels`.

    Returns
    -------
    int
        The new total number of objects found in `labels`.
    """
    z_regions = get_z_regions(region)
    areas = np.array([x.area for x in z_regions])
    splits = signal.argrelextrema(smooth(areas, 2, "hamming"), np.less)[0]
    for i, split in enumerate(splits):
        new_label = n_labels + 1
        if split != splits[-1]:
            z_slice = slice(split, splits[i + 1])
        else:
            z_slice = slice(split, None)
        assign_to_label(labels, region, (z_slice, slice(None), slice(None)), new_label)
        n_labels += 1
    return n_labels


def filter_small_ends(labels, region, min_pixels=5):
    """Remove small z-ends of labelled regions

    Parameters
    ----------
    labels : np.ndarray
        3D image containing labelled regions.
    region : skimage.measure.RegionProperties
        Region to filter.
    min_pixels : int, optional
        Minimum number of pixels for a label in each z-slice, by default 5.
    """
    z_regions = get_z_regions(region)
    i = 0
    # forward remove
    while i < len(z_regions) and z_regions[i].area <= min_pixels:
        assign_to_label(labels, region, (i, slice(None), slice(None)), 0)
        i += 1
    # backward remove
    i = len(z_regions) - 1
    while i >= 0 and z_regions[i].area <= min_pixels:
        assign_to_label(labels, region, (i, slice(None), slice(None)), 0)
        i -= 1


def backpropogate_split_labels(z_stack, labels):
    """Propogates split labels in a Z stack to lower slices.

    Parameters
    ----------
    z_stack : numpy.ndarray
        3D image stack from z=0 ... z=Z
    labels : numpy.ndarray
        Labels for current z-slice, should be z=Z slice.

    Returns
    -------
    numpy.ndarray
        Newly labelled `z_stack`
    """
    new_stack = np.zeros_like(z_stack, dtype=int)
    new_stack[-1, :, :] = labels
    for i in range(1, z_stack.shape[0])[::-1]:
        new_stack[i - 1, :, :] = segmentation.watershed(
            np.zeros_like(z_stack[i, :, :]),
            markers=new_stack[i, :, :],
            mask=z_stack[i - 1, :, :],
        )
    return new_stack


def split_z_disconeccted_labels(region, n_labels):
    """Split labels along the z-plane.

    Splits label along the z-plane if labels in individual z-slices have
    multiple connected components.

    Parameters
    ----------
    region : skimage.measure.RegionProperty
        Region to check and possibly split.
    n_labels : int
        Total number of current labels.

    Returns
    -------
    tuple : (numpy.ndarray, int)
        numpy.ndarray : Previous label split along z-axis if the label should be
            split. Otherwise, an array of zeros.
        int : Total number of current labels
    """
    new_labs = np.zeros_like(region.image, dtype=int)
    expand_labels = False
    n_new_labels = 0
    n_split = 0
    for i, z_slice in enumerate(region.image):
        segmented = measure.label(z_slice)
        new_labs[i, :, :] = segmented
        z_regions = measure.regionprops(new_labs[i, :, :])
        if len(z_regions) > 1 and not expand_labels:
            n_split += 1
            centroids = [r.centroid for r in z_regions]
            if n_split > 1 or max(spatial.distance.pdist(centroids)) > 5:
                expand_labels = True
                n_new_labels = len(np.unique(segmented)) - 1  # -1 for zeros
                if i != 0:
                    new_labs[: (i + 1), :, :] = backpropogate_split_labels(
                        new_labs[: (i + 1), :, :], new_labs[i, :, :]
                    )

        elif expand_labels and i > 0:
            # fill in z labels with z-1 labels
            z_filled = segmentation.watershed(
                np.zeros_like(new_labs[i, :, :]),
                markers=new_labs[i - 1, :, :],
                mask=new_labs[i, :, :],
            )
            # check if all regions in z labels are accounted for, otherwise
            # create new label
            for r in z_regions:
                if np.isin(0, z_filled[r.slice][r.image]):
                    n_new_labels += 1
                    z_filled[r.slice][r.image] = n_new_labels
            new_labs[i, :, :] = z_filled
    # re-assign label values to avoid conflict with previous labels + maintain
    # consistency
    for r in measure.regionprops(new_labs):
        if r.label == 1:
            assign_to_label(new_labs, r, None, region.label)
        else:
            assign_to_label(new_labs, r, None, n_labels + 1)

            n_labels += 1
    return new_labs, n_labels


def filter_by_area_length_ratio(labels, region, min_ratio=15):
    """Filter cylindrical-like labels from a 3D label image.

    Filters thin, long regions by comparing the (Total Area) / (Z length) ratio.

    Parameters
    ----------
    labels : numpy.ndarray
        Original 3D label image containing labels to filter.
    region : skimage.RegionProperty
        Region to possibly filter
    min_ratio : int, optional
        Minumum area / z length ratio to keep labels By default 15, and any
        label with a smaller ratio will be removed
    """
    if region.area / region.image.shape[0] < min_ratio:
        logging.info(f"Filtering region {region.label} for being too cylindrical")
        assign_to_label(labels, region, None, 0)


def renumber_labels(labels):
    """Renumber labels to that N labels will be labelled from 1 ... N"""
    for i, region in enumerate(measure.regionprops(labels)):
        assign_to_label(labels, region, None, i + 1)


def generate_labels(
    stain,
    pmc_probs,
    p_low=0.5,
    p_high=0.8,
    selem=None,
    max_stacks=7,
    min_stacks=3,
    split_disconnected=True,
):
    """Generate PMC labels within a confocal image.

    Parameters
    ----------
    stain : numpy.ndarray
        3D image containing PMC stain.
    pmc_probs : numpy.ndarray
        3D image containing probabilities of each pixel in `stain` containing a
        PMC.
    p_low : float, optional
        Lower probabilitiy bound for considering a pixel a PMC, by default 0.5.
        Used in `filters.apply_hysteresis_threshold()`
    p_high : float, optional
        Higher probabilitiy bound for considering a pixel a PMC, by default 0.5.
        Used in `filters.apply_hysteresis_threshold()`
    selem : np.ndarray, optional
        Structuring element used for morhpological opening / clsoing. If None,
        uses `skimage` defaults.
    max_stacks : int, optional
        The maximum number of z-slices a label can occupy before assuming stacked
        PMCs, by default 7.
    min_stacks : int, optional
        The minimum number of slices a label should occupy before removing,
        by default 3


    Returns
    -------
    np.ndarray
        3D image containing PMC segmentations.
    """
    if selem is None:
        selem = morphology.disk(2)
    pmc_seg = filters.apply_hysteresis_threshold(pmc_probs, p_low, p_high)
    # seeds = measure.label(morphology.binary_opening(pmc_seg, selem=selem))
    seeds = measure.label(
        np.array([morphology.binary_opening(x, selem=selem) for x in pmc_seg])
    )
    try:
        gradients = filters.sobel(stain, axis=0)
    except TypeError:
        gradients = np.array([filters.sobel(x) for x in stain])
    labels = segmentation.watershed(
        np.abs(gradients),
        seeds,
        mask=np.stack(
            morphology.closing(
                ndi.binary_fill_holes(x, selem),
                None,
            )
            for x in pmc_probs > 0.5
        ),
    )
    n_labels = len(np.unique(labels)) - 1

    # close up any holes in labels
    for region in measure.regionprops(labels):
        filled = morphology.binary_closing(region.image)
        labels[region.slice][filled] = region.label
        # assign_to_label(labels, region, (slice(None), slice(None), slice(None)), region.label)

    for region in measure.regionprops(labels):
        if region.area > 600:
            strict_pmc_prediction(region, pmc_probs, labels, 0.8)
    # check for abnormally large labels, possibly merged PMCS, so divide labels
    # if they separate in the z-plane
    for region in measure.regionprops(labels):
        try:
            region.feret_diameter_max
        except QhullError:
            continue
        if region.feret_diameter_max > 20 and split_disconnected:
            logging.info(
                f"Region {region.label} exceeded diamter theshold, splitting on Z disconnects"
            )
            split, n_labels_new = split_z_disconeccted_labels(region, n_labels)
            if n_labels != n_labels_new:
                logging.info(
                    f"Region {region.label} split into {n_labels_new - n_labels} regions."
                )
                assign_to_label(labels, region, None, 0)
                for each in np.unique(split):
                    if each != 0:
                        labels[region.slice][split == each] = each
            n_labels = n_labels_new

    # a check to ensure we're keeping track properly
    # assert n_labels == len(measure.regionprops(labels))

    # find abnormally long tracks, check for local minima in area size that
    # would indicate stacked pmcs. Additionally clean up small tails in labels.
    for region in measure.regionprops(labels):
        n_stacks = np.unique(region.coords[:, 0]).size
        if n_stacks > max_stacks:
            logging.info(
                f"Region {region.label} exceed maximum z-length, splitting by area."
            )
            n_labels = split_labels_by_area(labels, region, n_labels)
        elif n_stacks < min_stacks:
            logging.info(
                f"Region {region.label} only spans {region.image.shape[0]} z-slices: removing."
            )
            assign_to_label(labels, region, None, 0)
        filter_by_area_length_ratio(labels, region, min_ratio=15)
        filter_small_ends(labels, region, min_pixels=5)

    # renumber labels from 1...N
    renumber_labels(labels)

    return labels


def combine_segmentations(s1, s2):
    """Combine PMC segmentations in a union-like way.

    Combines PMC segmentations by prioritizing segmentations with *more* labels
    per identified region. This is preferred in our case because splitting PMCs
    is substantially more rare than merging PMCs.

    Parameters
    ----------
    s1 : numpy.ndarray
        3D image containing first segmentation of PMCs.
    s2 : numpy.ndarray
        3D image containing separate segmentation of PMCs.

    Returns
    -------
    numpy.ndarray
        Combined segmentation from `s1` and `s2`.
    """
    s1_regions = measure.regionprops(s1)
    s2_regions = {r.label: r for r in measure.regionprops(s2)}
    final = np.zeros_like(s1)
    n_labels = 0
    for region in s1_regions:
        s1_in_s2 = s2[region.slice][region.image]
        s2_labels = np.unique(s1_in_s2[s1_in_s2 != 0])
        # If multiple labels from s2 exist in region defined by s1, take s2
        # labels
        if len(s2_labels) > 1:
            for l in s2_labels:
                n_labels += 1
                if l in s2_regions:
                    # assign if label space is onoccupied, remove assigned region
                    if final[s2_regions[l].slice][s2_regions[l].image].sum() == 0:
                        final[s2_regions[l].slice][s2_regions[l].image] = n_labels
                    s2_regions.pop(l)
        # if same number of regions, just use s1 region
        else:
            n_labels += 1
            final[region.slice][region.image] = n_labels
    # now add any regions from s2 that weren't covered by regions in s1
    for l, region in s2_regions.items():
        if final[region.slice][region.image].sum() == 0:
            n_labels += 1
            final[region.slice][region.image] = n_labels
    return final


def strict_pmc_prediction(region, pmc_probs, labels, threshold):
    logging.info(
        f"Label {region.label} exceeds area with A={region.area}, attempting to split."
    )
    # smooth probabilities due to strict thresholding
    smoothed = np.array([filters.gaussian(x) for x in pmc_probs[region.slice]])
    split = False
    t = threshold
    # def split_origin_label
    while t < 1 and not split:
        split_labels = measure.label(
            np.array([ndi.binary_opening(x) for x in smoothed > t])
        )

        if split_labels.max() > 1:
            split = True
            logging.info(
                f"Label {region.label} split into {split_labels.max()} regions with t={t}."
            )
            # bounding box may include non-label object, remove from new labels
            split_labels[~region.image] = 0

            # fill in new labels to original area
            flooded = segmentation.watershed(
                np.zeros_like(region.image), split_labels, mask=region.image
            )
            # increment for new label numbers
            flooded += labels.max()
            for next in measure.regionprops(flooded):
                if next.area > 600:
                    logging.debug("area threshold exceeded in split label")
            labels[region.slice][region.image] = flooded[region.image]
        t += 0.025


def find_pmcs(
    stain,
    pmc_probs,
    max_stacks=7,
    min_stacks=3,
    p_low=0.45,
    p_high=0.5,
    selem=None,
    area_thresh=600,
    strict_threshold=0.8,
):
    """
    Segment each PMC in a 3D confocal image.

    Segments an PMC by combining "loose" and "strict" PMC segmentations.

    Parameters
    ----------
    stain : numpy.ndarray
        3D image containing PMC stain.
    pmc_probs : numpy.ndarray
        3D image containing probabilities of each pixel in `stain` containing a
        PMC.
    max_stacks : int, optional
        The maximum number of z-slices a label can occupy before assuming stacked
        PMCs, by default 7.
    min_stacks : int, optional
        The minimum number of slices a label should occupy before removing,
        by default 3
    p_low : float, optional
        Lower probabilitiy bound for considering a pixel a PMC during loose
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.45
    p_high : float, optional
        Higher probabilitiy bound for considering a pixel a PMC during loose
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.5
    selem : np.ndarray, optional
        Structuring element for morphological operations. Default is None, and
        skimage/scipy.ndimage defaults will be used.
    area_thresh : float, optional
        Maximum area for a single label. Any label exceeding the threshold will
        be attempted to be split into separate labels using stricter thresholding
        and segmentation. Default is 600.
    strict_threshold : float, optional
        Higher probabilitiy bound for considering a pixel a PMC during stricter
        segmentation. Used if the area of a region exceeds `area_thresh`, by
        default 0.8

    Returns
    -------
    np.ndarray
        Integer array with the same size of `stain` and `pmc_probs` where each
        numbered region represents a unique PMC.
    """

    labels = generate_labels(
        stain,
        pmc_probs,
        p_low=p_low,
        p_high=p_high,
        selem=selem,
        max_stacks=max_stacks,
        min_stacks=min_stacks,
        split_disconnected=False,
    )
    # check labels for large regions, attempt to separate with stricter thresholding
    for region in measure.regionprops(labels):
        if region.area > area_thresh:
            strict_pmc_prediction(region, pmc_probs, labels, strict_threshold)
    renumber_labels(labels)
    return labels


def labels_to_hdf5(image, filename):
    f = h5py.File(filename, "w")
    dataset = f.create_dataset("image", image.shape, h5py.h5t.NATIVE_INT16, data=image)
    f.close()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    try:
        snakemake
    except NameError:
        snakemake = None
    if snakemake is not None:
        pmc_probs = np.array(h5py.File(snakemake.input["probs"], "r")["exported_data"])[
            :, :, :, 1
        ]
        pmc_stain = np.array(h5py.File(snakemake.input["stain"], "r")["image"])
        pmc_segmentation = generate_labels(
            pmc_stain,
            pmc_probs,
            p_low=0.45,
            p_high=0.5,
            selem=None,
            max_stacks=7,
            min_stacks=2,
            split_disconnected=False,
        )
        labels_to_hdf5(pmc_segmentation, snakemake.output["labels"])
    else:
        import os
        import napari

        start_file = "DMSO/Replicate3/18hpf_DMSO_R3_emb1005.nd2"
        wc = start_file.replace(".nd2", "").replace("_", "-").replace("/", "_")
        pmc_probs = np.array(
            h5py.File(os.path.join("data", "pmc_probs", f"{wc}.h5"), "r")[
                "exported_data"
            ]
        )[:, :, :, 1]
        pmc_stain = np.array(
            h5py.File(os.path.join("data", "pmc_norm", f"{wc}.h5"))["image"]
        )
        pmc_segmentation = generate_labels(
            pmc_stain,
            pmc_probs,
            p_low=0.45,
            p_high=0.5,
            selem=None,
            max_stacks=7,
            min_stacks=2,
            split_disconnected=False,
        )
        labels2 = find_pmcs(pmc_stain, pmc_probs, strict_threshold=0.8)
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(pmc_stain, name="pmc", scale=[3.1, 1, 1])
            viewer.add_labels(pmc_segmentation, scale=[3.1, 1, 1], name="labels")
            try:
                viewer.add_labels(labels2, scale=[3.1, 1, 1], name="labels-adjusted")
            except NameError:
                pass
            viewer.add_image(
                pmc_probs,
                scale=[3.1, 1, 1],
                colormap="red",
                blending="additive",
                name="probs",
            )
