#!/bin/bash
#SBATCH --job-name=r007-gpu
#SBATCH --account=${ACCOUNT}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --output=${REPO}/log/gpu_%j.out
# usage: sbatch --dependency=afterok:<prep job> r007_gpu.sbatch CASE [CASE ...]   (exact rank mod p on one H200)
set -eo pipefail
source ${REPO}/scripts/oneloop/su-threshold-classes/cluster/r007_env.sh
NVLIBS=$(ls -d ${VENV}/lib*/python*/site-packages/nvidia/*/lib 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$NVLIBS$LD_LIBRARY_PATH
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "import cupy; cupy.arange(8).sum(); print('GPU smoke test ok, cupy', cupy.__version__)" || { echo "GPU STACK BROKEN"; exit 1; }
for CASE in "$@"; do
  echo "=== $CASE $(date)"
  python3 gpu_rank.py prep/$CASE --blk 2048
done
echo "GPU DONE $(date)"
