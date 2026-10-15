#!/bin/bash
# Reproduce the independent checks of the fixes for review items C19 (R-017) and C8 (R-016).
# Usage: REPO=/path/to/fortuity [PY=python3] [SKIP_NET=1] bash run_all.sh
# Needs numpy and sympy. c19/table1_cmp.py downloads the arXiv HTML of arXiv:2312.16443v2 unless SKIP_NET=1.
# Runs in a fresh temporary directory; nothing is written into the repo.
set -u
: "${REPO:?set REPO to the repo checkout}"
HERE=$(cd "$(dirname "$0")" && pwd)
export REPO PYTHONPATH="$REPO/code${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python3}
WORK=$(mktemp -d); cd "$WORK" || exit 1
echo "work dir $WORK; $($PY --version 2>&1)"
run() { echo "=== $*"; t0=$(date +%s); "$PY" "$@" 2>&1 || echo "!!! exit $?"; echo "--- $(( $(date +%s) - t0 )) s"; }
# C19 (R-017)
run "$HERE/c19/indep_index.py"          # exact SU(2) index and SU(3) C-twisted index, no repo code; writes su2_index_hw.pkl
run "$HERE/c19/su2_coulomb_exact.py"    # exact SU(2) diagonal-graviton dims over Q; level sums vs CCKLL (3.4)
run "$HERE/c19/cmp_a3.py"               # the 102 per-weight values of the 17 forced SU(2) columns vs c19/out/a3_forced.txt
run "$HERE/c19/tw22.py"                 # library twisted index to |q| <= 22: (9,6,6) -4, (10,6,6) -10
run "$HERE/c19/t027_cmp.py"             # T-027 classes vs -T in the complete SU(2) columns
[ "${SKIP_NET:-0}" = 1 ] || run "$HERE/c19/table1_cmp.py"   # CCKLL Table 1 boson/fermion signs vs c19/out/a5_table1.txt
# C8 (R-016)
run "$HERE/c8/exact_checks.py"          # exact: (2.20), Q r = R at N=3 (hw components), x1^3 = -648/5 O^2_24 identically
run "$HERE/c8/exact_allcomp.py"         # exact: Q r = R on all components of the traceless tensors at N=3 (about 5 min)
run "$HERE/c8/check_modp.py"            # closed at N=3 / not at N=4 and the -648/5 ratio at prime 33554393
run "$HERE/c8/check_modp_p3.py"         # the same at prime 50331653
run "$HERE/c8/singlet_T.py"             # singlet part T with and without readings 1 and 2
run "$HERE/c8/crt_ratio.py"             # T without reading 1, by CRT: -65649/350
run "$HERE/c8/spot_qr.py"               # spot checks of trace-sensitive non-hw components
