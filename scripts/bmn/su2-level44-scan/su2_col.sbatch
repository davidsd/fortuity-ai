#!/bin/bash
#SBATCH --job-name=fort-su2c
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --output=log/su2c_%A_%a.out
# T-027 phase 4: one SU(2) column per array task. Column list file and resources are given at submission:
#   sbatch --array=1-K --cpus-per-task=C --mem=M --export=ALL,COLS=cols_heavy.txt,HWPAR_WORKERS=W su2_col.sbatch
# Results go to results/su2_c_<q1>_<q2>_<q3>.jsonl, pre-seeded with the column's finished sectors (resumable).
cd $SCRATCH/fortuity-torch
source $SCRATCH/fortuity-venv/bin/activate
python3 -c "import fortuity.hwcoh.sector" || { echo "VENV BROKEN"; exit 1; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MAXLEN=3 MAXLEN_UP=3 GEN_NMAX=2 SEED=1 ENGINE=par
COL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $COLS)
OUT=results/su2_c_$(echo $COL | tr ',' '_').jsonl
echo "column $COL -> $OUT (workers ${HWPAR_WORKERS:-auto}, cpus $SLURM_CPUS_PER_TASK)"
python3 -u -m fortuity.hwcoh.run_sectors 2 dims_nf_su2_44.pkl $OUT $COL
