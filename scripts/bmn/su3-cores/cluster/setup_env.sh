#!/bin/bash
# One-time setup on FASRC, on any compute node (the kernels are built for generic AVX2, so shared/test/yin all work):
#   salloc -p shared -c 4 --mem=16G -t 01:00:00      (or -p test; then, inside the allocation)
#   bash ~/bootpod/fortuity/scripts/bmn/su3-cores/cluster/setup_env.sh
set -euo pipefail
module load python
module load gcc
FORT_ROOT=${FORT_ROOT:-$HOME/bootpod/fortuity}
FORT_ENV=${FORT_ENV:-$HOME/bootpod/fortenv}
if [ ! -d "$FORT_ENV" ]; then
  mamba create -y -p "$FORT_ENV" -c conda-forge python=3.11 numpy sympy pytest libflint mkl-devel
fi
source activate "$FORT_ENV" 2>/dev/null || { eval "$(conda shell.bash hook)"; conda activate "$FORT_ENV"; }
pip install --quiet -e "$FORT_ROOT/code"
cd "$FORT_ROOT/code/fortuity/hwcoh/flint"
# fast finite-field kernels (optional; without them the code falls back to numpy, ~10x slower on big sectors)
FLINT_PREFIX=$CONDA_PREFIX bash build.sh || echo "WARNING: FLINT kernel (libffla.so) not built"
MKLROOT=$CONDA_PREFIX MARCH=x86-64-v3 bash build_ffblas_mkl.sh \
  || MKLROOT=$CONDA_PREFIX MARCH=haswell bash build_ffblas_mkl.sh \
  || echo "WARNING: BLAS kernel (libffblas.so) not built"      # generic AVX2 build: runs on shared and yin nodes
cd "$FORT_ROOT/scripts/bmn/su3-cores/cluster"
source env.sh
python3 check_env.py
