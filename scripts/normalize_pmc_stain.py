# Script to normalize PMC stain using Intensify3D
#
# Takes as input any microscopy image file, outputs an .h5 file
# to be used as input for ilastik PMC segmentation


import h5py
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology, transform


def to_hdf5(image, filename):
    f = h5py.File(filename, "w")
    dataset = f.create_dataset("image", image.shape, h5py.h5t.NATIVE_DOUBLE, data=image)
    f.close()


def clean_pmc_stain(img, size=25):
    """
    Preprocessing for PMC stains.

    Binarizes image to find isolated objects. Filters objects by set size.

    Parameters
    ----------
    img : numpy.ndarray
        PMC stain to clean.
    size : int, optional
        Size threshold for objects. Any object with volume < `size` will be
        removed. By default, 25
    -------
    np.ndarray
        PMC stain with small objects removed.
    """
    out = []
    for img_slice in img:
        equalized = exposure.equalize_adapthist(img_slice)
        fixed = morphology.closing(
            equalized > filters.threshold_otsu(equalized), morphology.disk(1)
        )
        slice_copy = img_slice.copy()
        slice_copy[~morphology.remove_small_objects(fixed, size)] = 0
        out.append(equalized)
    return np.array(out)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        snakemake = None
        img = AICSImage(snakemake.input["image"])
        channel = snakemake.input["pmc_channel"]
        z_start = snakemake.params["z_start"]
        z_stop = snakemake.params["z_end"]
        pmc = img.get_image_data("ZYX", C=channel)[z_start:z_stop]
        pmc = np.array(
            [
                exposure.rescale_intensity(x, out_range=(pmc.min(), pmc.max()))
                for x in pmc
            ]
        )
        pmc = exposure.rescale_intensity(pmc, out_range=(0, 1))
        to_hdf5(pmc, snakemake.output["out"])
