# T-042 environment on Torch: repo code copy, MKL ffblas kernel, SU(3) BMN hw cohomology at j=32 (settings of R-007's j=30 scan)
source ${VENV}/bin/activate
export PYTHONPATH=${SCRATCH}/t042/code
export FORTUITY_FFBLAS=${REPO}/lib/libffblas.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ENGINE=par GROUP=SU GEN_NMAX=3
export MAXLEN=6 MAXLEN_UP=6 HW_CACHE=${SCRATCH}/t042/hwcache
cd ${SCRATCH}/t042
