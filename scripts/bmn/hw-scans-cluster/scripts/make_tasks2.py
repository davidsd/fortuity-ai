"""Task lists for hw_par.sbatch: per-sector tasks for heavy columns, packed columns for light ones.
usage: make_tasks2.py DIMS_PKL QSUM1[,QSUM2..] OUT_PREFIX [--mmax M] [--heavy-frac F] [--nlight K]
Writes OUT_PREFIX_heavy.txt (one 'lam:n' per line, m <= mmax), OUT_PREFIX_light.txt (packed columns),
OUT_PREFIX_big.txt (sectors with m > mmax, not scheduled)."""
import sys, pickle, argparse
from fortuity.hwcoh.dims import collapse
from fortuity.hwcoh.column import hw_mult
ap = argparse.ArgumentParser()
ap.add_argument('dims'); ap.add_argument('qsums'); ap.add_argument('out')
ap.add_argument('--mmax', type=int, default=10**9); ap.add_argument('--heavy-frac', type=float, default=0.85)
ap.add_argument('--nlight', type=int, default=24)
a = ap.parse_args()
d = collapse(pickle.load(open(a.dims, 'rb')))
cols = []
for s in [int(x) for x in a.qsums.split(',')]:
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if q3 > q2 or q3 < 0:
                continue
            lam = (q1, q2, q3)
            per = {n: (d.get((n,) + lam, 0), hw_mult(d, n, lam)) for n in range(0, 3 * s + 1) if d.get((n,) + lam, 0)}
            if per:
                cols.append((sum(v * max(m, 1) for v, m in per.values()), lam, per))
cols.sort(reverse=True)
tot = sum(c for c, _, _ in cols)
L = lambda l: ",".join(map(str, l))
heavy, big, light, acc = [], [], [], 0
for c, lam, per in cols:
    if acc < a.heavy_frac * tot or any(m > a.mmax for v, m in per.values()):
        for n, (v, m) in sorted(per.items()):
            (big if m > a.mmax else heavy).append((m, f"{L(lam)}:{n}"))
    else:
        light.append((c, L(lam)))
    acc += c
heavy.sort(reverse=True)
bins = [[0, []] for _ in range(a.nlight)]
for c, l in light:
    b = min(bins, key=lambda x: x[0]); b[0] += c; b[1].append(l)
open(a.out + '_heavy.txt', 'w').write("".join(t + "\n" for _, t in heavy))
open(a.out + '_light.txt', 'w').write("".join(" ".join(b[1]) + "\n" for b in bins if b[1]))
open(a.out + '_big.txt', 'w').write("".join(f"{t} m={m}\n" for m, t in sorted(big, reverse=True)))
print(f"{len(cols)} columns: {len(heavy)} heavy sector tasks, {sum(1 for b in bins if b[1])} light tasks, "
      f"{len(big)} big sectors (m > {a.mmax}) deferred")
