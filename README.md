# Workflow for the paper Ultrafast computational screening of molecules with inverted singlet-triplet energy gaps using semi-empirical quantum chemistry

To reproduce the calculations of the paper:

1. [Install Quarto](https://quarto.org/docs/get-started/)
2. Install snakemake in a fresh conda environmen with `conda install snakemake`
3. Run the snakemake workflow with `snakemake manuscript --cores <number of cpus> --use-conda`