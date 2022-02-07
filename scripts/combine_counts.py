import pandas as pd

if __name__ == '__main__':
    try:
        snakmake
    except NameError:
        snakemake = None
    if snakemake is not None:
        embryo_counts = pd.concat[pd.read_csv(x) for x in snakemake.input]
        embryo_counts.to_csv(snakemake.output)