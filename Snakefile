import os


outdir = 'data'

rule move_files:
    input:
        ['a.nd2', 'b.nd2']
    output:
        os.path.join(outdir, "raw", "{embryo}.nd2")
    script:
        'scripts/symlink_input_files.py'

rule normalize_pmc_stains:
    input:
        os.path.join(outdir, "raw", '{embyro}.nd2')
    output:
        os.path.join(outdir, "pmc_norm", '{embyro}.h5')
    script:
        "scripts/normalize_pmc_stain.py"

rule predict_pmcs:
    input:
        image=os.path.join(outdir, "pmc_norm", "{embryo}.h5"),
        model=config['ilastik']['model']
    params:
        ilastik_loc = '/home/dakota/Downloads/ilastik-1.3.3post3-Linux/run_ilastik.sh'
        ilastik_loc = config['ilastik']['loc']
    output:
        os.path.join(outdir, "pmc_probs", "{embryo}.h5")
    shell:
        "{params.ilastik_loc} --headless "
        "project=MyProject.ilp "
        "output_format=hdf5 "
        "output_filename_format={snakemake.output} "
        "{input.image}"

rule label_pmcs:
    input:
        stain=os.path.join(outdir, "pmc_norm", "{embryo}.h5"),
        probs=os.path.join(outdir, "pmc_probs", "{embryo}-image_Probabilities.h5")
    output:
        os.path.join(outdir, "labels", "{embryo}_pmc_labels.h5")
    script:
        "scripts/label_pmcs.py"

rule count_spots:
    input:
        image=os.path.join(outdir, 'raw', '{embryo}.nd2'),
        labels=os.path.join(outdir, 'labels', '{embryo}_pmc_labels.h5')
    output:
        os.path.join(outdir, 'counts', '{embryo}.csv')
    script:
        'scripts/count_spots.py'


rule combine_counts:
    input:
        os.path.join(outdir, 'counts', '{embryo}.csv')
    output:
        os.path.join(outdir, 'final', 'counts.csv')
    script:
        'scripts/combine_counts.py'