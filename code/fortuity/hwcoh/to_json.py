"""Collect per-sector JSONL results into one data file: per (j, lam, 2J) multiplicities of the SU(3)_R irrep lam
in cohomology (h), multi-graviton part (grav), non-graviton part (BH), plus the exact-index Euler check per column."""
import sys, json, glob
from collections import defaultdict
from fortuity.bmn.fullindex import full_index, su3_multiplicities
N = int(sys.argv[2]); files = [f for pat in sys.argv[1].split(',') for f in sorted(glob.glob(pat))]
col = defaultdict(dict)
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r['N'] == N:
            col[tuple(r['lam'])][r['n']] = r
jmax = max(2 * sum(l) for l in col)
Z, A = full_index(N, jmax, verbose=False)
out, bad = [], 0
for lam, secs in sorted(col.items(), key=lambda x: (2 * sum(x[0]), x[0])):
    j = 2 * sum(lam); rep = [lam[0] - lam[1], lam[1] - lam[2]]
    rows, chi, ok = [], 0, True
    for n in sorted(secs):
        r = secs[n]
        rn = secs[n - 1]['r_up'] if (n - 1) in secs else 0
        h = r['m'] - r['r_up'] - rn
        ok &= bool(r['check'])
        chi += (-1) ** n * h
        rows.append(dict(twoJ=n, E=j / 2 - n / 2, m=r['m'], h=h, grav=r['grav'], BH=h - r['grav'], check=r['check']))
    idx = su3_multiplicities(Z[j], A).get(tuple(rep), 0)
    bad += (not ok) or (chi != idx)
    out.append(dict(j=j, lam=list(lam), rep=rep, sectors=rows, euler=chi, index=idx, all_checks=ok))
json.dump(dict(N=N, columns=out), open(sys.argv[3], 'w'), indent=1)
print(len(out), "columns;", "problems:", bad, "; nonzero BH:",
      [(c['j'], c['rep'], s['twoJ'], s['BH']) for c in out for s in c['sectors'] if s['BH']])
