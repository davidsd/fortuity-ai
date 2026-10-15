#!/bin/bash
# Build libffblas.so (BLIS dgemm based mod-p matmul and blocked rank).  Needs a CBLAS with 64-bit ints and OpenMP.
# On Expanse (python/3.11 module stack): AMDBLIS_ROOT is set by the amdblis module; compiler: aocc clang.
set -e
cd "$(dirname "$0")"
: "${AMDBLIS_ROOT:?set AMDBLIS_ROOT (or adapt the include/lib paths)}"
${CC:-clang} -O3 -march=native -fopenmp -fPIC -shared -o libffblas.so.new.$$ ffblas.c -I$AMDBLIS_ROOT/include \
  -L$AMDBLIS_ROOT/lib -lblis-mt -Wl,-rpath,$AMDBLIS_ROOT/lib -lm
mv -f libffblas.so.new.$$ libffblas.so   # atomic rename: processes that have the old library mapped keep it (an in-place
                                     # rebuild crashed running jobs with SIGBUS on 2026-09-28)
echo built $(pwd)/libffblas.so
