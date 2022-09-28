import typer
from nd2reader import ND2Reader
import numpy as np
import pathlib
from pyometiff import OMETIFFReader, OMETIFFWriter


def convert_metadata(nd2_img):
    # https://github.com/filippocastelli/pyometiff
    meta = OMETIFFReader._get_metadata_template()
    meta["PhysicalSizeX"] = nd2_img.metadata["pixel_microns"]
    meta["PhysicalSizeXUnit"] = "µm"
    meta["PhysicalSizeY"] = nd2_img.metadata["pixel_microns"]
    meta["PhysicalSizeYUnit"] = "µm"
    meta["PhysicalSizeZ"] = int(
        np.median(np.diff(nd2_img.metadata["z_coordinates"][::-1]))
    )
    meta["PhysicalSizeZUnit"] = "µm"
    meta["SizeX"] = nd2_img.metadata["height"]
    meta["SizeY"] = nd2_img.metadata["width"]
    meta["SizeZ"] = len(nd2_img.metadata["z_levels"])
    meta["SizeC"] = len(nd2_img.metadata["channels"])
    meta["Channels"] = {x: {} for x in nd2_img.metadata["channels"]}
    return meta


def main(nd2: str, prefix: str = "", emb_start: int = 0):
    """
    Extract mutliple embryos as series from an ND2 file.

    Write each as an ome.tiff file.

    Parameters\n
    ----------\n
    nd2 : str\n
        ND2 image to separate.\n
    prefix : str, optional\n
        Prefix to append to written file names, by default ''.\n
    emb_start : int, optional\n
        Embryo number to start count from, by default 0, and 'emb0' will be the
        first file written.
    """
    images = ND2Reader(nd2)
    embryos = []
    dimension_order = "CZYX"
    for i in range(images.sizes["v"]):
        images.default_coords["v"] = i
        channels = []
        for j in range(images.sizes["c"]):
            images.default_coords["c"] = j
            images.bundle_axes = ("z", "y", "x")
            channels.append(images.get_frame(0))
        embryos.append(np.array(channels))
    embryos = np.array(embryos)
    outdir = pathlib.Path(nd2).parent
    meta_data = convert_metadata(images)
    for i in range(emb_start, len(embryos)):
        OMETIFFWriter(
            fpath=outdir.joinpath(prefix + f"emb{i}.ome.tiff"),
            dimension_order=dimension_order,
            array=embryos[i, :, :, :],
            metadata=meta_data,
            explicit_tiffdata=False,
        ).write()


if __name__ == "__main__":
    # https://github.com/tiangolo/typer
    typer.run(main)
