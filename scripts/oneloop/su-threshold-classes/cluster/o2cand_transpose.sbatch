#!/bin/bash
#SBATCH --job-name=o2c-transp
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=${REPO}/log/o2c_transp_%j.out
# R-007 order 2: transpose the T_2 'out' prep (B | Cartan, 218976 x 173227) for kernel_vec.py
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
python3 -u transpose_prep.py prep/r007_o2_out_codd
echo "TRANSPOSE DONE $(date)"
