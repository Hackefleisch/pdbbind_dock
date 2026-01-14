#!/bin/bash
set -e

#source /media/iwe20/DataSSD/pdbbind_dock/.venv/bin/activate

run_docking() {
    ulimit -v 6000000

    uv run /media/iwe20/DataSSD/pdbbind_dock/main.py \
        --n_relax 0 \
        --n_relax_ligaway 0 \
        --n_dock 1 \
        --outdir /media/iwe20/DataSSD/pdbbind_dock/h5_test \
        --pdbbind /media/iwe20/DataSSD/pdbbind_dock/pdbbind_cleaned \
        --pdb_file /media/iwe20/DataSSD/pdbbind_dock/pdbbind_cleaned/index.txt \
        --pdb_index "$1" \
        --protocols /media/iwe20/DataSSD/pdbbind_dock/xml_protocols/docking_perturb.xml
}

export -f run_docking

# Run indices 0 to 11459 using 10 parallel jobs
# only start new jobs if 7gb of memory are free
seq 0 11459 | parallel -j 10 --memfree 7G run_docking