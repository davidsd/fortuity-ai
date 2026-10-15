#!/bin/bash
#SBATCH --job-name=r007-setup
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=log/setup_%j.out
# T-031 environment, separate from T-027's fortuity-venv: r007-venv (python 3.12, numpy, sympy, MKL, CuPy) and
# libffblas.so built against pip-installed MKL (repo recipe: code/fortuity/hwcoh/flint/build_ffblas_mkl.sh).
set -e
R=${REPO}
cd $R
mkdir -p log lib
PY=$(command -v python3.12 || command -v python3)
echo "python: $PY ($($PY --version))"; echo "gcc: $(command -v gcc) $(gcc --version | head -1)"
V=${VENV}
[ -x $V/bin/python3 ] || $PY -m venv $V
source $V/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy sympy mkl mkl-include mkl-devel cupy-cuda12x nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "site-packages: $SP"
ls $V/lib | grep -i mkl | head -20
ls $V/include | grep -i mkl_cblas || true
# build libffblas.so against MKL (LP64, GNU OpenMP threading layer)
mkdir -p $R/lib/inc/blis
echo '#include <mkl_cblas.h>' > $R/lib/inc/blis/cblas.h
lib() { ls $V/lib/lib$1.so* 2>/dev/null | head -1; }
LP=$(lib mkl_intel_lp64); TH=$(lib mkl_gnu_thread); CO=$(lib mkl_core)
echo "MKL libs: $LP $TH $CO"
gcc -O3 -march=x86-64-v3 -fopenmp -fPIC -shared -o $R/lib/libffblas.so code/fortuity/hwcoh/flint/ffblas.c \
  -I$R/lib/inc -I$V/include -Wl,--no-as-needed $LP $TH $CO -lgomp -lpthread -lm -ldl -Wl,-rpath,$V/lib
echo "built $R/lib/libffblas.so"
# smoke test: BLIS/MKL kernel vs the numpy fallback (matmul and ranks on random matrices mod p)
export FORTUITY_FFBLAS=$R/lib/libffblas.so FF_THREADS=4
cd $R/code && python3 - <<'PYEOF'
import numpy as np
from fortuity.hwcoh import ffblas
from fortuity.bmn.modp import P, matmul_mod, RREFBasis
assert ffblas.HAVE_FFBLAS, "libffblas.so not loaded"
rng = np.random.default_rng(1)
A = rng.integers(0, P, (300, 700), dtype=np.int64); B = rng.integers(0, P, (700, 200), dtype=np.int64)
assert (ffblas.matmul(A, B) == matmul_mod(A, B)).all(), "matmul mismatch"
# a rank-deficient pair: rank 150 and 170
L = rng.integers(0, P, (400, 150), dtype=np.int64); Rm = rng.integers(0, P, (150, 900), dtype=np.int64)
M1 = matmul_mod(L, Rm); M2 = rng.integers(0, P, (20, 900), dtype=np.int64)
r1, r12 = ffblas.rank_pair32(M1.astype(np.int32), M2.astype(np.int32))
b = RREFBasis(900); b.add_batched(M1 % P); rb = b.rank
print("ffblas rank_pair", r1, r12, "numpy rank", rb)
assert (r1, r12) == (150, 170) and rb == 150
SB = ffblas.StreamBasis(900); SB.add(M1); SB.add(M2[:5]); assert SB.rank == 155
print("ffblas (MKL) smoke test OK")
PYEOF
python3 -c "import cupy; print('cupy', cupy.__version__, '(no GPU on this node, import only)')" || true
echo SETUP DONE
