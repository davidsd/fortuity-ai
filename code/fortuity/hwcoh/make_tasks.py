"""Write a task file: one line per array task, columns (dominant lam with 2*sum(lam) = j) balanced by estimated cost.
usage: python -m fortuity.hwcoh.make_tasks DIMS_NF_PKL J1,J2,... NTASKS OUT"""
import sys, pickle
from fortuity.hwcoh.dims import collapse
from fortuity.hwcoh.column import hw_mult
dims = collapse(pickle.load(open(sys.argv[1], 'rb')))
js = [int(x) for x in sys.argv[2].split(',')]; nt = int(sys.argv[3])
cols = []
for j in js:
    s = j // 2
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if q3 > q2 or q3 < 0:
                continue
            lam = (q1, q2, q3)
            cost = sum(dims.get((n,) + lam, 0) * max(hw_mult(dims, n, lam), 1) for n in range(0, 3 * s + 1))
            if cost:
                cols.append((cost, lam))
cols.sort(reverse=True)
bins = [[0, []] for _ in range(nt)]
for cost, lam in cols:
    b = min(bins, key=lambda x: x[0]); b[0] += cost; b[1].append(lam)
with open(sys.argv[4], 'w') as f:
    for c, lams in bins:
        f.write(" ".join(",".join(map(str, l)) for l in lams) + "\n")
print(len(cols), "columns in", nt, "tasks; max bin cost %.2e, total %.2e" % (max(b[0] for b in bins), sum(b[0] for b in bins)))
