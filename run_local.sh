#!/bin/bash
set -e

source /home/iwe20/Projects/pdbbind_dock/.venv/bin/activate        

run_docking() {
    ulimit -v 6000000

    python /home/iwe20/Projects/pdbbind_dock/main.py \
        --n_relax 0 \
        --n_relax_ligaway 0 \
        --n_dock 1 \
        --zarr_store /home/iwe20/Projects/pdbbind_dock/full_test.zarr \
        --pdbbind /home/iwe20/Projects/pdbbind_dock/pdbbind_cleaned \
        --pdb_file /home/iwe20/Projects/pdbbind_dock/pdbbind_cleaned/index.txt \
        --pdb_index "$1" \
        --protocols /home/iwe20/Projects/pdbbind_dock/xml_protocols/docking_perturb.xml
}

export -f run_docking

# Run indices 0 to 11517 using 10 parallel jobs
# only start new jobs if 7gb of memory are free
seq 0 11517 | parallel -j 10 --memfree 7G run_docking