import os
import pandas as pd

configfile: "files/config.yaml"
# shell.prefix("conda activate napari; ")


OUTDIR = config['output']['dir']


def file_to_wc(filename):
    return os.path.splitext(filename)[0].replace('_', '-').replace('/', '_')

embryo_log = pd.read_csv(config['input']['logfile'])
embryo_log['wildcard'] = embryo_log.apply(lambda x: file_to_wc(x.file), axis=1)
embryo_log.set_index('wildcard', inplace=True)


rule all:
    input:
        os.path.join(OUTDIR, 'final', 'counts.csv')


rule debug_conda:
    output:
        "conda_debug.out"
    shell:
        "python -c 'import sys; print(sys.prefix); print(sys.path); import pandas as pd;' > {output};"


checkpoint symlink_input_files:
    input:
        csv=config['input']['logfile'],
    params:
        datadir=config['input']['datadir'],
        OUTDIR=config['output']['dir']
    output:
        directory(os.path.join(OUTDIR, "raw"))
    script:
        'scripts/symlink_input_files.py'

def get_embryo_param(wc, col):
    return embryo_log.at[wc.embryo, col]

rule normalize_pmc_stains:
    input:
        image=os.path.join(OUTDIR, "raw", '{embryo}.nd2')
    params:
        channel_name='pmc',
        channels=lambda wc: get_embryo_param(wc, 'channel_order'),
        z_start=lambda wc: get_embryo_param(wc, 'z-start'),
        z_end=lambda wc: get_embryo_param(wc, 'z-end')
    output:
        h5=os.path.join(OUTDIR, "pmc_norm", '{embryo}.h5')
    conda:
        "envs/preprocess.yaml"
    script:
        "scripts/normalize_pmc_stain.py"


rule predict_pmcs:
    input:
        image=os.path.join(OUTDIR, "pmc_norm", "{embryo}.h5"),
        model=config['ilastik']['model']
    params:
        ilastik_loc = config['ilastik']['loc']
    output:
        os.path.join(OUTDIR, "pmc_probs", "{embryo}.h5")
    shell:
        "{params.ilastik_loc} --headless "
        "--project={input.model} "
        "--output_format=hdf5 "
        "--output_filename_format={output} "
        "{input.image}"


rule label_pmcs:
    input:
        stain=os.path.join(OUTDIR, "pmc_norm", "{embryo}.h5"),
        probs=os.path.join(OUTDIR, "pmc_probs", "{embryo}.h5")
    output:
        labels=os.path.join(OUTDIR, "labels", "{embryo}_pmc_labels.h5")
    conda:
        "envs/preprocess.yaml"
    script:
        "scripts/label_pmcs.py"


rule quantify_expression:
    input:
        image=os.path.join(OUTDIR, 'raw', '{embryo}.nd2'),
        labels=os.path.join(OUTDIR, 'labels', '{embryo}_pmc_labels.h5')
    params:
        gene_params=config['quant']['genes'],
        channels=lambda wc: get_embryo_param(wc, 'channel_order'),
        z_start=lambda wc: get_embryo_param(wc, 'z-start'),
        z_end=lambda wc: get_embryo_param(wc, 'z-end')
    output:
        csv=os.path.join(OUTDIR, 'counts', '{embryo}.csv')
    conda:
        "envs/quant.yaml"
    script:
        'scripts/count_spots.py'


def aggregate_checkpoing_output(wildcards):
    checkpoint_output = checkpoints.symlink_input_files.get(**wildcards).output[0]
    print(checkpoint_output)
    file_names = expand(
        os.path.join(OUTDIR, 'counts', '{embryo}.csv'), 
        embryo=glob_wildcards(
            os.path.join(checkpoint_output, "{embryo}.nd2")
        ).embryo
    )
    return file_names


rule combine_counts:
    input:
        aggregate_checkpoing_output
    output:
        os.path.join(OUTDIR, 'final', 'counts.csv')
    script:
        'scripts/combine_counts.py'