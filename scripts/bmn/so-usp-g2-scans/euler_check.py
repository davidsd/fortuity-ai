"""Euler-characteristic check of an hw scan against the exact BMN index, for any group in fortuity.bmn.groupindex.
For every SU(3)_R column lam in the scan: sum_2J (-1)^2J h(2J, lam) must equal the multiplicity of the irrep with
highest weight lam in the exact index (h = m - r_up(2J-1) - r_up(2J), from fortuity.hwcoh.run_sectors output).
usage (from code/): python ../scripts/bmn/so-usp-g2-scans/euler_check.py GROUP N JMAX scan1.jsonl [scan2.jsonl ...]
  GROUP in SU, SO_odd, Sp, G2 (N ignored for G2). Columns must be complete (all 2J) for the check to be meaningful."""
import sys, json
from fortuity.bmn.groupindex import bmn_index
from fortuity.bmn.fullindex import su3_multiplicities

group, N, jmax = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = {}
for f in sys.argv[4:]:
    for line in open(f):
        r = json.loads(line)
        rows[(tuple(r['lam']), r['n'])] = r
chi = {}
for (lam, n), r in rows.items():
    rin = rows[(lam, n - 1)]['r_up'] if (lam, n - 1) in rows else 0
    chi[lam] = chi.get(lam, 0) + (-1) ** n * (r['m'] - rin - r['r_up'])
Z, A = bmn_index(group, N, jmax)
bad = []
for lam, c in sorted(chi.items()):
    j = 2 * sum(lam)
    if j > jmax:
        continue
    m = int(su3_multiplicities(Z[j], A).get((lam[0] - lam[1], lam[1] - lam[2]), 0))
    if c != m:
        bad.append((lam, c, m))
print(f"{group}: {len(chi)} columns, all self-checks {all(r['check'] for r in rows.values())}; "
      f"Euler characteristic = index in all columns: {not bad}" + (f"; mismatches (lam, chi, index): {bad}" if bad else ""))
