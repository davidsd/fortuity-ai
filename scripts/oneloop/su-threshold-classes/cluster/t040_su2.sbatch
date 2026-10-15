#!/bin/bash
#SBATCH --job-name=t040-su2
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=${REPO}/log/t040_%A_%a.out
# T-040: one-loop fate of one SU(2) BMN sector with non-graviton classes (line SLURM_ARRAY_TASK_ID of t040_cases.txt):
# classes (oneloop_test reps), Coulomb test, Q_1 of the classes, controls, then the parallel exactness test (fullpar)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
CASE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" t040_cases.txt)
echo "node $(hostname) case $CASE"
for st in reps coulomb q1; do python3 -u oneloop_test.py $CASE $st; done
python3 -u make_ctrl.py $CASE
python3 -u fullpar.py $CASE --workers $SLURM_CPUS_PER_TASK --seed 41 --workdir $TMPDIR --span
python3 -u fullpar.py $CASE --workers $SLURM_CPUS_PER_TASK --seed 43 --workdir $TMPDIR
echo "T040 DONE $(date)"
