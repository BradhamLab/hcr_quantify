# hcr-quantify
Quantify signal in HCR FISH images

This workflow specifically quantifies signal found in primary mesenchyme cells (PMCs) using 3D confocal images of sea urchin embryos. 

The workflow preprocesses images to normalize PMC stains for Z-depth and increasing contrast. After preprocessing, we identify PMCs using a random forest classifier trained using Ilastik. We quantify signal levels using FISHQuant to count puncta, as well as calculating the average signal intensity. The final output of the pipeline is a summarized table including signal measurements as well as physical locations and size of each cell in each embryo:


## Final Output
|   label | embryo                                      |   area |   diameter |        Z |       Y |       X |   sm50_spots |   sm50_intensity |   pks2_spots |   pks2_intensity | treatment   |
|--------:|:--------------------------------------------|-------:|-----------:|---------:|--------:|--------:|-------------:|-----------------:|-------------:|-----------------:|:------------|
|       1 | MK886_MK-2-0_replicate3_18hpf-MK2uM-R3-emb4 |    399 |    9.13394 | 1.88471  | 281.09  | 232.043 |            5 |          1.60824 |           18 |          2.26741 | MK886       |
|       2 | MK886_MK-2-0_replicate3_18hpf-MK2uM-R3-emb4 |    359 |    8.8179  | 1.32869  | 287.46  | 219.78  |            2 |          1.08333 |           20 |          2.60488 | MK886       |
|       3 | MK886_MK-2-0_replicate3_18hpf-MK2uM-R3-emb4 |    105 |    5.85325 | 0.790476 | 297.343 | 211.829 |            0 |          1.64723 |            2 |          1.16409 | MK886       |
|       4 | MK886_MK-2-0_replicate3_18hpf-MK2uM-R3-emb4 |    293 |    8.24055 | 2.0785   | 296.072 | 228.212 |            4 |          1.47955 |           22 |          3.77743 | MK886       |
|       5 | MK886_MK-2-0_replicate3_18hpf-MK2uM-R3-emb4 |    557 |   10.2083  | 2.2711   | 313.194 | 210.686 |            5 |          1.52303 |           30 |          2.27475 | MK886       |

## Additional Output
Along the way, processed images, identified labels, and PMC isolated expression patterns are also generated as `.h5` and `.nc` files.

# Installation

To install the pipeline, clone the repository and install [snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html). Navigate to the repository in your terminal, and issue the command:

`snakemake -j{number_of_jobs} --use-conda --conda-frontend="{frontend_of_choice}"`

You will have to update the configuration file (`files/config.yaml`) to specifically point to your data of interest as well as a log file containing necessary meta information for each of the confocal image files (e.g. z start and stop, channel order, any treatments applied to the embryo, etc.)
