"""T-027 phase 4: one job per unfinished SU(2) column. Reads every results file (su2_t*.jsonl, su2_c*.jsonl), finds
columns not complete, writes cols_heavy.txt / cols_light.txt (one column per line, heaviest first) and pre-seeds
results/su2_c_<q1>_<q2>_<q3>.jsonl with the sectors of that column already done, so run_sectors skips them."""
import json, glob, pickle, os
from collections import defaultdict
from fortuity.hwcoh.dims import collapse
from fortuity.hwcoh.column import hw_mult
from fortuity.bmn.words import contents_of_sector
dims = collapse(pickle.load(open('dims_nf_su2_44.pkl', 'rb')))
def cost(lam):
    s = sum(lam)
    return sum(dims.get((n,) + lam, 0) * max(hw_mult(dims, n, lam), 1) for n in range(0, 3 * s + 1))
lines = defaultdict(dict)
for f in sorted(glob.glob('results/su2_t*.jsonl')) + sorted(glob.glob('results/su2_c_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        if r.get('check'):
            lines[tuple(r['lam'])][r['n']] = l if l.endswith('\n') else l + '\n'
allcols = [tuple(int(x) for x in c.split(',')) for l in open('tasks_su2_j44.txt') for c in l.split()]
rem = []
for lam in allcols:
    ns = [n for n in range(0, sum(lam) + 1) if contents_of_sector((n,) + lam)]
    if not all(n in lines[lam] for n in ns):
        rem.append((cost(lam), lam, len(ns) - sum(1 for n in ns if n in lines[lam])))
rem.sort(reverse=True)
NHEAVY = int(os.environ.get('NHEAVY', '20'))
with open('cols_heavy.txt', 'w') as fh, open('cols_light.txt', 'w') as fl:
    for i, (c, lam, left) in enumerate(rem):
        (fh if i < NHEAVY else fl).write(','.join(map(str, lam)) + '\n')
        out = 'results/su2_c_%d_%d_%d.jsonl' % lam
        with open(out, 'w') as g:
            for n in sorted(lines[lam]):
                g.write(lines[lam][n])
print('complete columns: %d of %d; unfinished: %d (heavy %d, light %d)' % (len(allcols) - len(rem), len(allcols), len(rem), min(NHEAVY, len(rem)), max(0, len(rem) - NHEAVY)))
for c, lam, left in rem[:6]: print('   heaviest: %s, %d sectors left, cost %.2e' % (lam, left, c))
