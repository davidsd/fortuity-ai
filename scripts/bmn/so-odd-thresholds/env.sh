# T-025 environment on Torch: repo code copy (commit 809059f), MKL ffblas kernel (same sources as fortuity-r007), shared hw cache
source ${VENV}/bin/activate
export PYTHONPATH=${REPO}/code
export FORTUITY_FFBLAS=${REPO}/lib/libffblas.so
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 ENGINE=par GROUP=SO_odd
export HW_CACHE=${REPO}/hwcache
cd ${REPO}
