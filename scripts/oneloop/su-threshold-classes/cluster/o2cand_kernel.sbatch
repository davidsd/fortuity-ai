#!/bin/bash
#SBATCH --job-name=o2c-kernel
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --output=${REPO}/log/o2c_kernel_%j.out
# R-007 order 2: random left-kernel vectors of the T_2 'out' prep on one H200 (rank must be 105277)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
NVLIBS=$(ls -d ${VENV}/lib*/python*/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "import cupy; cupy.arange(8).sum(); print('GPU smoke test ok')" || { echo "GPU STACK BROKEN"; exit 1; }
python3 -u kernel_vec.py prep/r007_o2_out_codd --k 4 --blk 2048
python3 -c "import json; d = json.load(open('prep/r007_o2_out_codd/xker.json')); assert d['rank'] == 105277, d['rank']; print('rank OK')"
echo "KERNEL DONE $(date)"
