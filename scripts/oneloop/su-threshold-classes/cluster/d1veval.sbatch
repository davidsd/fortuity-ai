#!/bin/bash
#SBATCH --job-name=d1-veval
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=${REPO}/log/d1veval_%j.out
# V rows [Q_1 Q_0 u; Q_1 t_k] at the functionals of the stored order-3 outbound prep (Q_0 into U_2 = (3,3;14,8,8), Y=9)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u veval.py prep/r007_o3_out_codd cache/fullpar_hw_r007_o3_out_codd.pkl classes/r007_d1/vecs_Q1.pkl prep/r007_d1_U2 \
    --log ${REPO}/log/oprep_18905656.out --workers $SLURM_CPUS_PER_TASK --slice-words 100000
echo "D1 VEVAL DONE $(date)"
