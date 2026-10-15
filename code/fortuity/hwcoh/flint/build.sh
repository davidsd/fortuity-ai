#!/bin/bash
# Build libffla.so against a FLINT 3 installation.  On Expanse (a site build of FLINT; see scripts/README.md):
#   module load cpu/0.15.4 gcc/10.2.0 openmpi/4.0.4 gmp/6.1.2 mpfr/4.0.2 slurm; FLINT_PREFIX=<path to a FLINT 3 installation> ./build.sh
# Elsewhere: FLINT_PREFIX=<prefix of a FLINT 3 install> ./build.sh.  Without the library, hwcoh falls back to numpy.
set -e
cd "$(dirname "$0")"
: "${FLINT_PREFIX:?set FLINT_PREFIX}"
DIRS=$(ldd $FLINT_PREFIX/lib/libflint.so | awk '/=> \//{print $3}' | xargs -n1 dirname | sort -u | sed 's/^/-Wl,-rpath,/' | tr '\n' ' ')
gcc -O2 -fPIC -shared -o libffla.so.new.$$ ffla.c -I$FLINT_PREFIX/include -L$FLINT_PREFIX/lib -lflint -Wl,-rpath,$FLINT_PREFIX/lib $DIRS
mv -f libffla.so.new.$$ libffla.so   # atomic rename: processes that have the old library mapped keep it (an in-place
                                     # rebuild crashed running jobs with SIGBUS on 2026-09-28)
echo built $(pwd)/libffla.so
