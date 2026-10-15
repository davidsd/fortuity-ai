#!/bin/bash
#SBATCH --job-name=d2-kernelB
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=280G
#SBATCH --time=12:00:00
#SBATCH --output=${REPO}/log/d2_kernelB_%j.out
# R-007 order 2, d_2: left-kernel vectors (x, z) of [B; Q_1 x] for the one-loop prep (all rows of B: --frac 1.0),
# i.e. c = x / z with c B = -Q_1 x.  Expected rank 76781 + 0 (Q_1 x is exact).  26 vectors: one solution and 25
# differences of solutions = random Q_0-closed vectors of (5,5;14,8,8), Y = 9 (for the rank of d_1 into T_2).
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -c "from fortuity.hwcoh import ffblas; assert ffblas.HAVE_FFBLAS; print('ffblas (MKL) loaded')"
# second, independent copy (different random kernel vectors, 32 threads): its own directory with links to the inputs
mkdir -p ../oneloop/prep/su3_j30_b
for f in B.i32 BT.i32 VC.i32 VC.json meta.json; do ln -sf ../su3_j30/$f ../oneloop/prep/su3_j30_b/$f; done
echo "node $(hostname), cpus $SLURM_CPUS_PER_TASK"
python3 -u kernel_vec_aug.py ../oneloop/prep/su3_j30_b --frac 1.0 --k 26 --seed 2028
python3 -c "
import numpy as np
z = np.load('../oneloop/prep/su3_j30_b/zker_aug.npy'); print('z', z.tolist()); assert z.any(), 'no kernel vector with z != 0'"
echo "D2 KERNEL DONE $(date)"
