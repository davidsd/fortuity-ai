"""T-025 scan audit: completeness (every nonempty 2J of every column up to JMAX), per-sector self-check, cohomology
h = m - r_up(2J-1) - r_up(2J) and non-graviton count h - grav per sector.
usage (PYTHONPATH=<repo>/code): python analyze.py N JMAX results/soX_*.jsonl"""
import sys, json
from fortuity.bmn.words import contents_of_sector
from make_tasks import cols

N, jmax, files = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
rows = {}
for f in files:
    for line in open(f):
        r = json.loads(line)
        if r['N'] == N and r.get('group') == 'SO_odd':
            rows[(tuple(r['lam']), r['n'])] = r
missing, bad, neg, nongrav, nsec, hsum, gsum = [], [], [], [], 0, 0, 0
for s in range(1, jmax // 2 + 1):
    for lam in cols(s):
        for n in range(0, s + 1):
            if not contents_of_sector((n,) + lam):
                continue
            r = rows.get((lam, n))
            if r is None:
                missing.append((lam, n)); continue
            nsec += 1
            if not r['check']:
                bad.append((lam, n))
            rin = rows[(lam, n - 1)]['r_up'] if n > 0 and (lam, n - 1) in rows else 0
            h = r['m'] - rin - r['r_up']
            hsum += h; gsum += r['grav']
            if h < 0 or r['grav'] > h:
                neg.append((lam, n, h, r['grav']))
            if h > r['grav']:
                nongrav.append((lam, n, h, r['grav']))
print(f"SO({2*N+1}) j<={jmax}: {nsec} sectors in {sum(len(cols(s)) for s in range(1, jmax//2+1))} columns; "
      f"missing {missing or 'none'}; failed self-checks {bad or 'none'}; inconsistent (h<0 or grav>h) {neg or 'none'}")
print(f"  total h = {hsum}, of which graviton {gsum}; sectors with non-graviton classes (lam, 2J, h, grav): {nongrav or 'none'}")
