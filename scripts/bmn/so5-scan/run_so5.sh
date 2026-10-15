#!/bin/bash
# SO(5) = Sp(2) BMN highest-weight scan (R-012, T-024): all dominant columns with j <= JMAX, all 2J, restartable.
#
# RECONSTRUCTION, 2026-10-02.  The original run_so5.sh used for the j <= 26 laptop scan of
# 2026-09-27 was never committed and is lost (review C2).  This script reproduces the command recorded in
# the data metas and in this directory's README:
#     GROUP=SO_odd GEN_NMAX=8 SEED=1 python3 -m fortuity.hwcoh.run_sectors 2 <dims> hw_SO_odd2.jsonl <columns>
# with the exact dimensions computed in place (auto:JMAX) instead of read from a pickle.  It was checked against the
# stored records: see check_so5_driver.py (same m, m_d, r_up, grav for every sector recomputed).
#
# usage: JMAX=24 [OUT=hw_SO_odd2.jsonl] ./run_so5.sh      (bash, not zsh: the column list must be word-split)
set -euo pipefail
JMAX=${JMAX:-24}
OUT=${OUT:-hw_SO_odd2.jsonl}
COLS=$(python3 - "$JMAX" <<'PY'
import sys
jmax = int(sys.argv[1])
cols = []
for j in range(2, jmax + 1, 2):
    s = j // 2
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if 0 <= q3 <= q2:
                cols.append(f"{q1},{q2},{q3}")
print(" ".join(cols))
PY
)
GROUP=SO_odd GEN_NMAX=8 SEED=${SEED:-1} python3 -m fortuity.hwcoh.run_sectors 2 "auto:$JMAX" "$OUT" $COLS
