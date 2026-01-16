#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=12:00:00
#SBATCH --job-name=Astdock
#SBATCH --array=0-84
#SBATCH --output=astex_logs/%x_%A_%a.out
#SBATCH --error=astex_logs/%x_%A_%a.err

WORKSPACE="/data/horse/ws/paei790f-pdbbind_dock"
VENV_PYTHON="$WORKSPACE/pdbbind_dock/.venv/bin/python"
DATA_DIR="$WORKSPACE/posebusters_paper_data/astex_diverse_set"

cd "$WORKSPACE" || exit 1

mkdir -p astex_logs

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not accessible on node $(hostname)"
    exit 1
fi

echo "Running Task ID: $SLURM_ARRAY_TASK_ID on node $(hostname)"

srun  "$VENV_PYTHON" "$WORKSPACE/pdbbind_dock/main.py" \
    --n_relax 10 \
    --n_relax_ligaway 10 \
    --n_dock 300 \
    --outdir "$WORKSPACE/astex_h5" \
    --pdbbind "$DATA_DIR" \
    --pdb_file "$DATA_DIR/astex_diverse_set_ids.txt" \
    --pdb_index "$SLURM_ARRAY_TASK_ID" \
    --protocols  \
        "$WORKSPACE/pdbbind_dock/xml_protocols/docking_perturb.xml" \
        "$WORKSPACE/pdbbind_dock/xml_protocols/docking_std.xml"
