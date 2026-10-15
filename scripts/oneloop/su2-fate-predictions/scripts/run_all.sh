#!/bin/bash
# Run the T-037 reproduction scripts (a)-(g) against a checkout of the repo; everything goes to stdout.
# Usage: REPO=/path/to/fortuity [PY=python3] [SCRIPTS="a_scope e_L2_inputs"] bash run_all.sh
# Runs in a fresh temporary directory, so the ./cache files the scripts write stay out of the repo.
set -u
: "${REPO:?set REPO to the repo checkout}"
PY=${PY:-python3}
HERE=$(cd "$(dirname "$0")" && pwd)
export REPO PYTHONDONTWRITEBYTECODE=1
WORK=$(mktemp -d)
cd "$WORK" || exit 1
echo "work dir $WORK; $($PY --version 2>&1)"
for s in ${SCRIPTS:-a_scope b_hair_model c_partner_sizes c3_degree10_allorders d_L1_qplus d2_L1_qplus_f d3_L1_full_letters e_L2_inputs e2_dimV_certificate f_labels_j28_j30 g_h_T1_QplusO0}; do
  echo "=== $s"
  t0=$(date +%s)
  "$PY" "$HERE/$s.py" 2>&1 || echo "!!! $s exited with status $?"
  echo "--- $s: $(( $(date +%s) - t0 )) s"
done
