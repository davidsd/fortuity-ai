#!/bin/bash
#SBATCH --job-name=fort-su2
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=48:00:00
#SBATCH --array=1-64
#SBATCH --output=log/su2_%A_%a.out
# T-027: SU(2) BMN complete SU(3)_R highest-weight scan, j <= 44. One line of tasks_su2_j44.txt per array task.
# Resumable: run_sectors skips sectors already in the task's JSONL; re-sbatch after a timeout.
cd $SCRATCH/fortuity-torch
source $SCRATCH/fortuity-venv/bin/activate
python3 -c "import fortuity.hwcoh.sector" || { echo "VENV BROKEN -- run setup.sbatch"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# traces of length <= 3 generate all invariants of 2x2 matrices; every sector is certified (rank = exact multiplicity)
export MAXLEN=3 MAXLEN_UP=3 GEN_NMAX=2 SEED=1
TASKS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" tasks_su2_j44.txt)
echo "task $SLURM_ARRAY_TASK_ID: $TASKS"
python3 -u -m fortuity.hwcoh.run_sectors 2 dims_nf_su2_44.pkl results/su2_t${SLURM_ARRAY_TASK_ID}.jsonl $TASKS
