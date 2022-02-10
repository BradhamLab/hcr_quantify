# Script to normalize PMC stain using Intensify3D
#
# Takes as input any microscopy image file, outputs an .h5 file
# to be used as input for ilastik PMC segmentation

import h5py
import numpy as np
from aicsimageio import AICSImage
from skimage import exposure


def get_channel_index(channels, channel):
    channel_index = [
        i for i, x in enumerate(channels.split(";")) if x.lower() == channel.lower()
    ][0]
    return channel_index


def to_hdf5(image, filename):
    f = h5py.File(filename, "w")
    dataset = f.create_dataset("image", image.shape, h5py.h5t.NATIVE_DOUBLE, data=image)
    f.close()


def preprocess_slice(img, upper_percentile=99.99, new_min=0, new_max=1):
    """Preprocess a Z-slice by scaling and equalizing intensities.

    Parameters
    ----------
    img : np.ndarray
        Image slice to scale and equalize.
    upper_percentile : float, optional
        Upper bound to clip intensities for scaling, by default 99.99.
    new_min : int, optional
        New minimum intensity value, by default 0.
    new_max : int, optional
        New maximum intensity value, by default 1.

    Returns
    -------
    np.ndarray
        Scaled and equalize image slice.
    """
    lb, ub = np.percentile(img, (0, upper_percentile))
    out = exposure.equalize_adapthist(
        exposure.rescale_intensity(img, in_range=(lb, ub), out_range=(new_min, new_max))
    )
    return out


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        snakemake = None
    if snakemake is not None:
        print(snakemake.params)
        img = AICSImage(snakemake.input["image"])
        channel = get_channel_index(
            snakemake.params["channels"], snakemake.params["channel_name"]
        )
        z_start = snakemake.params["z_start"]
        z_stop = snakemake.params["z_end"]
        pmc = img.get_image_data("ZYX", C=channel)[z_start:z_stop]
        pmc = np.array(
            [
                preprocess_slice(x, upper_percentile=100, new_min=0, new_max=1)
                for x in pmc
            ]
        )
        to_hdf5(pmc, snakemake.output["h5"])
