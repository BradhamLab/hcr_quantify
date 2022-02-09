import os

import pandas as pd

if __name__ == "__main__":
    try:
        snakemake
    except NameErorr:
        snakemake = None
    if snakemake is not None:
        logfile = pd.read_csv(snakemake.input["csv"])
        datadir = snakemake.params["datadir"]
        outdir = snakemake.params["outdir"]
        logfile["new_name"] = logfile.apply(
            lambda x: x.file.replace("_", "-").replace("/", "_"), axis=1
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        for idx in logfile.index:
            src = logfile.loc[idx, "file"]
            dst = logfile.loc[idx, "new_name"]
            os.symlink(os.path.join(datadir, src), os.path.join(outdir, dst))
