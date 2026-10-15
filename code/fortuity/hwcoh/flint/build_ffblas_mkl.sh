#!/bin/bash
# Build libffblas.so against Intel MKL instead of BLIS (for machines with oneAPI but no BLIS, e.g. PI Symmetry).
# ffblas.c includes <blis/cblas.h>; a one-line shim header redirects it to MKL's CBLAS. MKL's GNU-OpenMP threading
# layer honours omp_set_num_threads, so FF_THREADS / ff_set_threads work as with BLIS.
# usage: MKLROOT=/opt/intel/oneapi/mkl/latest OUT=<dir> ./build_ffblas_mkl.sh   then  export FORTUITY_FFBLAS=<dir>/libffblas.so
# (OUT defaults to this directory). Check with the random-matrix comparison against the numpy fallback before use.
set -e
cd "$(dirname "$0")"
: "${MKLROOT:=/opt/intel/oneapi/mkl/latest}"
OUT=${OUT:-$(pwd)}
mkdir -p "$OUT/inc/blis"
echo '#include <mkl_cblas.h>' > "$OUT/inc/blis/cblas.h"
gcc -O3 -march=${MARCH:-native} -fopenmp -fPIC -shared -o "$OUT/libffblas.so" ffblas.c -I"$OUT/inc" -I"$MKLROOT/include" \
  -L"$MKLROOT/lib" -Wl,--no-as-needed -lmkl_gnu_thread -lmkl_intel_lp64 -lmkl_core -lgomp -lpthread -lm -ldl \
  -Wl,-rpath,"$MKLROOT/lib"
echo built "$OUT/libffblas.so"
