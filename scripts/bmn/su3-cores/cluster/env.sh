# Source me on the FASRC cluster (login or compute node):  source env.sh
# Environment for the T-032 runs: conda env with numpy/sympy/FLINT/MKL, repo code on PYTHONPATH, single-threaded BLAS.
module load python >/dev/null 2>&1        # Miniforge (mamba/conda). `module avail python` if you want a specific one.
export FORT_ROOT=${FORT_ROOT:-$HOME/bootpod/fortuity}
export FORT_ENV=${FORT_ENV:-$HOME/bootpod/fortenv}
source activate "$FORT_ENV" 2>/dev/null || { eval "$(conda shell.bash hook)"; conda activate "$FORT_ENV"; }
export PYTHONPATH=$FORT_ROOT/code${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLINT_THREADS=1
