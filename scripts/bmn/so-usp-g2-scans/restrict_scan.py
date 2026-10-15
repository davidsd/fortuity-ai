"""T-023: after a complete SO(7)/G2 hw scan, run hwcoh.restrict on every sector where SO(7) has non-graviton classes.
usage (from code/): python ../scripts/bmn/so-usp-g2-scans/restrict_scan.py DIMS_SO7 DIMS_G2 OUT_JSONL SO7_JSONL [...]
Uses HW_CACHE (same hw vectors as the scan) and a process pool over sectors (env FORTUITY_WORKERS)."""
import sys, os, json, pickle
import multiprocessing as mp
from fortuity.hwcoh.gravhw import GravitonAlgebra
from fortuity.hwcoh.restrict import restrict_sector

dso, dg = (pickle.load(open(f, 'rb')) for f in sys.argv[1:3])
out = sys.argv[3]
rows = {}
for f in sys.argv[4:]:
    for line in open(f):
        r = json.loads(line)
        rows[(tuple(r['lam']), r['n'])] = r
targets = []
for (lam, n), r in sorted(rows.items()):
    rin = rows[(lam, n - 1)]['r_up'] if (lam, n - 1) in rows else 0
    h = r['m'] - rin - r['r_up']
    if h - r.get('grav', 0) > 0:
        targets.append((lam, n))
print(f"SO(7) sectors with non-graviton classes: {targets}", flush=True)
GA = GravitonAlgebra(8)


def work(t):
    lam, n = t
    return restrict_sector(lam, n, (3, 'SO_odd', dso), (0, 'G2', dg), GA)


W = int(os.environ.get('FORTUITY_WORKERS', '1'))
with mp.get_context('fork').Pool(max(1, min(W, len(targets) or 1))) as pool:
    res = pool.map(work, targets)
with open(out, 'w') as fh:
    for r in res:
        fh.write(json.dumps(r) + "\n")
        print("RESTRICT", json.dumps(r), flush=True)
