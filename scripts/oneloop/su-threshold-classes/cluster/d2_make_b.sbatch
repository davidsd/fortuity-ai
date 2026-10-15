#!/bin/bash
#SBATCH --job-name=d2-makeB
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/d2_makeB_%j.out
# R-007 order 2, d_2: y with Q_0 y = -Q_1 x, z = K_2 U x + Q_1 y, the closedness checks at fresh points, and the test
# rows [z, Q_1 y'_j for 24 random closed y' (rank of d_1 into T_2), the 14 cocycles t_k of classes/r007_d1 (Coulomb LW
# classes and the candidate), z + noise] with exact controls
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK"
python3 -u d2_make.py --case su3_j30 --N 3 --prep ../oneloop/prep/su3_j30_b --cache ../oneloop/cache/fullpar_hw_su3_j30.pkl \
    --x xker_aug.npy --z zker_aug.npy --nclosed 24 --q1-stored ../oneloop/classes/su3_j30/q1.pkl \
    --in-cache cache/fullpar_hw_r007_o2_in_codd.pkl --tdir classes/r007_d1 --out classes/r007_d2b --npts 4 \
    --workers $SLURM_CPUS_PER_TASK
echo "D2 MAKE DONE $(date)"
