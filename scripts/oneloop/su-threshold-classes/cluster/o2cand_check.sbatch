#!/bin/bash
#SBATCH --job-name=o2c-check
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=${REPO}/log/o2c_check_%j.out
# R-007 order 2: fresh-point checks of the candidate (t != 0, t = 0 on the Cartan, Q_0 t = 0, Q_0 D t = 0)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u build_cand.py --stage check --nvec 2 --workers $SLURM_CPUS_PER_TASK
echo "CHECK DONE $(date)"
