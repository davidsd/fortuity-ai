# sourced by the T-031 sbatch scripts
R=${REPO}
source ${VENV}/bin/activate
export PYTHONPATH=$R/code
export FORTUITY_FFBLAS=$R/lib/libffblas.so
export FF_THREADS=${SLURM_CPUS_PER_TASK:-1} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1
cd $R/scripts/oneloop/su-threshold-classes
