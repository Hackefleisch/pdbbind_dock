#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=PDBBdock
#SBATCH --array=0-11459
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

WORKSPACE="/data/horse/ws/paei790f-pdbbind_dock"
VENV_PYTHON="$WORKSPACE/pdbbind_dock/.venv/bin/python"
DATA_DIR="$WORKSPACE/data/pdbbind_cleaned"

# cd to the repo root so `-m docking.main` resolves the package
cd "$WORKSPACE/pdbbind_dock" || exit 1

mkdir -p "$WORKSPACE/logs"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not accessible on node $(hostname)"
    exit 1
fi

echo "Running Task ID: $SLURM_ARRAY_TASK_ID on node $(hostname)"

srun  "$VENV_PYTHON" -m docking.main \
    --n_relax 10 \
    --n_relax_ligaway 10 \
    --n_dock 300 \
    --outdir "$WORKSPACE/data/pdbbind_h5" \
    --pdbbind "$DATA_DIR" \
    --pdb_file "$DATA_DIR/index.txt" \
    --pdb_index "$SLURM_ARRAY_TASK_ID" \
    --protocols  \
        "$WORKSPACE/pdbbind_dock/docking/xml_protocols/docking_perturb.xml" \
        "$WORKSPACE/pdbbind_dock/docking/xml_protocols/docking_std.xml"
