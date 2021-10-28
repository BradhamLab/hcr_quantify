from skimage import io
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
import napari
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import xarray as xr
import skimage
import starfish
from starfish.types import Axes, Levels

from starfish import Experiment
from starfish import ImageStack
from starfish.image import Filter
from starfish.util.plot import imshow_plane, intensity_histogram
from starfish.spots import FindSpots, DecodeSpots, AssignTargets
from starfish import Codebook
from starfish import BinaryMaskCollection
from starfish.core.types import Coordinates


img = AICSImage("/home/dakota/Data/hcr_fish/Chd_IRXA_control_emb1.nd2")
# n_bits = img.metadata.images[0].pixels.significant_bits
n_bits = 12
example_z = 26

z_coords = np.arange(0, img.dims.Z) * img.physical_pixel_sizes.Z
xy_coords = np.arange(0, img.dims.X) * img.physical_pixel_sizes.X

# with napari.gui_qt():
#     viewer = napari.Viewer()
#     for i in range(3):
#         viewer.add_image(
#             img.get_image_dask_data("ZYX", C=i),
#             contrast_limits=[0, 2 ** n_bits - 1],
#             scale=[
#                 # img.physical_pixel_sizes.Z / img.physical_pixel_sizes.X,
#                 1,
#                 1,
#                 1,
#             ],
#         )
#     labels = viewer.add_labels(np.zeros(img.shape[2:], dtype=int))
# print(np.unique(labels.data))

experiment = ImageStack.from_numpy(
    skimage.img_as_float32(
        exposure.rescale_intensity(
            img.get_image_data()[:, 1:, :, :],
            in_range=(0, 2 ** (n_bits) - 1),
            out_range=(0, 1),
        )
    ),
    coordinates={
        Coordinates.X: xy_coords,
        Coordinates.Y: xy_coords,
        Coordinates.Z: z_coords,
    }
    # index_labels=["r", "c", "z", "x", "y"],
)

clip_97 = Filter.Clip(p_min=97)
clipped: starfish.ImageStack = clip_97.run(experiment)

masking_radius = 5
filt = Filter.WhiteTophat(masking_radius, is_volume=True)
filtered = filt.run(experiment, verbose=True, in_place=False)


def plot_comparisons(image1, image2, title1="", title2="", z=example_z):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8), dpi=260)
    for (i, row_axes) in zip(range(2), axes):
        orig_plot: xr.DataArray = image1.sel(
            {Axes.CH: i, Axes.ROUND: 0, Axes.ZPLANE: z}
        ).xarray.squeeze()

        comp_plot: xr.DataArray = image2.sel(
            {Axes.CH: i, Axes.ROUND: 0, Axes.ZPLANE: z}
        ).xarray.squeeze()
        row_axes[0].imshow(orig_plot, cmap="magma")
        row_axes[0].set_title(f"{title1} - channel {i + 1}")
        row_axes[1].imshow(comp_plot, cmap="magma")
        row_axes[1].set_title(f"{title2} - channel {i + 1}")
        row_axes[0].axis("off")
        row_axes[1].axis("off")
    plt.tight_layout()
    return fig, axes


plot_comparisons(experiment, filtered, "raw", "white_hat filtered")
plt.savefig("white_hat.png")


def plot_intensity_histograms(stack: starfish.ImageStack, z: int):
    fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=150, sharey=True, sharex=True)
    intensity_histogram(
        stack,
        sel={Axes.ROUND: 0, Axes.CH: 0, Axes.ZPLANE: z},
        log=True,
        bins=50,
        ax=ax1,
    )
    intensity_histogram(
        stack,
        sel={Axes.ROUND: 0, Axes.CH: 1, Axes.ZPLANE: z},
        log=True,
        bins=50,
        ax=ax2,
    )
    ax1.set_title("channel=1")
    ax2.set_title("channel=2")
    fig.tight_layout()


# ClipPercentileToZero values below 80% and above 99.999% and scale
cptz_2 = Filter.ClipPercentileToZero(
    p_min=80, p_max=99.999, is_volume=True, level_method=Levels.SCALE_BY_CHUNK
)
normed = cptz_2.run(filtered, in_place=False)
plot_intensity_histograms(filtered, example_z)
plot_intensity_histograms(normed, example_z)

plot_comparisons(filtered, normed, "filtered", "normalized")

# p = FindSpots.BlobDetector(
#     min_sigma=1,
#     max_sigma=10,
#     num_sigma=10,
#     threshold=np.percentile(np.ravel(filtered.xarray.values), 90),
#     measurement_type="mean",
#     detector_method="blob_dog",
#     is_volume=True,
# )
# intensities = p.run(filtered)

p = FindSpots.LocalMaxPeakFinder(
    min_distance=3, stringency=0, min_obj_area=4, max_obj_area=400, is_volume=True
)
intensities = p.run(normed, n_processes=6)

codebook = Codebook.synthetic_one_hot_codebook(n_round=1, n_channel=1, n_codes=2)
decoder = DecodeSpots.SimpleLookupDecoder(codebook=codebook)
decoded_intensities = decoder.run(spots=intensities)


label_img = io.imread("/home/dakota/Code/scratch/label_test.tif")
binary_masks = [label_img == i for i in np.unique(label_img) if i != 0]
labels = BinaryMaskCollection.from_binary_arrays_and_ticks(
    binary_masks,
    # pixel_ticks={
    #     Coordinates.X: range(img.dims.X),
    #     Coordinates.Y: range(img.dims.Y),
    #     Coordinates.Z: range(img.dims.Z),
    # },
    pixel_ticks=None,
    physical_ticks={
        Coordinates.X: xy_coords,
        Coordinates.Y: xy_coords,
        Coordinates.Z: z_coords,
    },
    log=experiment.log,
)

al = AssignTargets.Label()
labeled = al.run(labels, decoded_intensities)
labeled_filtered = labeled[labeled.cell_id != "nan"]
mat = labeled_filtered.to_expression_matrix()
