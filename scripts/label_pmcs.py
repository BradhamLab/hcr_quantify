import h5py
import numpy as np
from scipy import signal
from skimage import filters, measure, morphology, segmentation


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
        Slice defining which part of the region to label.
    new_label : int
        New label to set
    """
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


def generate_labels(
    stain, pmc_probs, p_low=0.5, p_high=0.8, selem=None, max_stacks=7, min_stacks=3
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
        disk = morphology.disk(2)
        selem = np.zeros((3, *disk.shape))
        selem[0, 1:4, 1:4] = morphology.disk(1)
        selem[1, :] = disk
        selem[2, 1:4, 1:4] = morphology.disk(1)
    pmc_seg = filters.apply_hysteresis_threshold(pmc_probs, p_low, p_high)
    seeds = measure.label(morphology.binary_opening(pmc_seg, selem=selem))
    seeds = measure.label(seeds > 0)
    gradients = filters.sobel(stain, axis=0)
    labels = segmentation.watershed(
        np.abs(gradients),
        seeds,
        # mask=pmc_seg,
        mask=morphology.binary_closing(pmc_seg),
    )
    n_labels = len(np.unique(labels)) - 1

    for region in measure.regionprops(labels):
        n_stacks = np.unique(region.coords[:, 0]).size
        if n_stacks > max_stacks:
            n_labels = split_labels_by_area(labels, region, n_labels)
        elif n_stacks < min_stacks:
            assign_to_label(labels, region, (slice(None), slice(None), slice(None)), 0)
        filter_small_ends(labels, region, min_pixels=5)

    # we've now filtered labels, so re-analyze regions
    for region in measure.regionprops(labels):
        closed = morphology.closing(
            region.image,
        )
        labels[region.slice][closed] = region.label

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


def find_pmcs(
    stain,
    pmc_probs,
    max_stacks=7,
    min_stacks=3,
    p_low_loose=0.5,
    p_high_loose=0.8,
    p_low_strict=0.6,
    p_high_strict=0.85,
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
    p_low_loose : float, optional
        Lower probabilitiy bound for considering a pixel a PMC during loose
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.5
    p_high_loose : float, optional
        Higher probabilitiy bound for considering a pixel a PMC during loose
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.8
    p_low_strict : float, optional
        Lower probabilitiy bound for considering a pixel a PMC during stricter
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.6
    p_high_strict : float, optional
        Higher probabilitiy bound for considering a pixel a PMC during stricter
        segmentation. Used in `filters.apply_hysteresis_threshold()`,
        by default 0.85

    Returns
    -------
    np.ndarray
        Integer array with the same size of `stain` and `pmc_probs` where each
        numbered region represents a unique PMC.
    """

    labels1 = generate_labels(
        stain,
        pmc_probs,
        p_low=p_low_loose,
        p_high=p_high_loose,
        max_stacks=max_stacks,
        min_stacks=min_stacks,
    )
    labels2 = generate_labels(
        stain,
        pmc_probs,
        p_low=p_low_strict,
        p_high=p_high_strict,
        max_stacks=max_stacks,
        min_stacks=min_stacks,
    )
    return combine_segmentations(labels1, labels2)


def labels_to_hdf5(image, filename):
    f = h5py.File(filename, "w")
    dataset = f.create_dataset("image", image.shape, h5py.h5t.NATIVE_INT16, data=image)
    f.close()


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        snakemake = None
    if snakemake is not None:

        pmc_probs = np.array(h5py.File(snakemake.input["probs"], "r")["exported_data"])[
            :, :, :, 1
        ]
        pmc_stain = np.array(h5py.File(snakemake.input["norm_stain"], "r")["image"])
        pmc_segmentation = find_pmcs(pmc_stain, pmc_probs)
        labels_to_hdf5(pmc_segmentation, snakemake.output)
