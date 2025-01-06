#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=PDBBdock
#SBATCH --array=0-19442


module load Anaconda3/2024.02-1

source $EBROOTANACONDA3/etc/profile.d/conda.sh

conda activate /work/mai13hsq-PDBBindDock/pdbbind_dock_env

python /work/mai13hsq-PDBBindDock/pdbbind_dock/main.py \
    --n_relax 10 \
    --n_relax_ligaway 10 \
    --n_dock 150 \
    --zarr_store /work/mai13hsq-PDBBindDock/results.zarr \
    --pdbbind /work/mai13hsq-PDBBindDock/pdbbind_2020 \
    --pdb_file /work/mai13hsq-PDBBindDock/pdbbind_2020/index/INDEX_structure.2020 \
    --pdb_index $SLURM_ARRAY_TASK_ID \
    --protocols \
        /work/mai13hsq-PDBBindDock/pdbbind_dock/xml_protocols/docking_perturb.xml \
        /work/mai13hsq-PDBBindDock/pdbbind_dock/xml_protocols/docking_std.xml
