#!/bin/bash
#SBATCH --job-name=r007-prep
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/prep_%x_%j.out
# usage: sbatch --cpus-per-task=C --mem=M r007_prep_seed.sbatch CASE SEED   -> prep/CASE_sSEED (independent functionals)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
CASE=$1; SEED=$2
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK, case $CASE seed $SEED"
rm -rf prep/${CASE}_s${SEED}
python3 fullpar.py $CASE --seed $SEED --workers $SLURM_CPUS_PER_TASK --xcap 12e9 --workdir $TMPDIR --prep-dir prep/${CASE}_s${SEED}
test -s prep/${CASE}_s${SEED}/meta.json
echo "PREP DONE $(date)"
