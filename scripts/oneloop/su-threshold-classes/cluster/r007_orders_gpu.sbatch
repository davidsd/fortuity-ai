#!/bin/bash
#SBATCH --job-name=r007-ogpu
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --output=${REPO}/log/ogpu_%j.out
# usage: sbatch --dependency=afterok:<prep> r007_orders_gpu.sbatch PREPDIR   (exact rank mod p on one H200)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
cd ${REPO}/scripts/oneloop/su-threshold-classes_orders
NVLIBS=$(ls -d ${VENV}/lib*/python*/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "import cupy; cupy.arange(8).sum(); print('GPU smoke test ok')" || { echo "GPU STACK BROKEN"; exit 1; }
python3 gpu_rank.py prep/$1 --blk 2048
cat prep/$1/rank_gpu.json
echo "GPU DONE $(date)"
