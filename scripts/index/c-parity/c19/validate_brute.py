import sys, time, json
from molien import molien
from brute import Space
N, Q = int(sys.argv[1]), int(sys.argv[2])
M = molien(N, Q)
S = Space(N)
t0 = time.time(); n = nz = bad = 0
mism = []
for s in range(Q + 1):
    for q1 in range(s + 1):
        for q2 in range(s + 1 - q1):
            q = (q1, q2, s - q1 - q2)
            for tj in range(0, s + 1):
                b = S.invariant_dim(tj, q)
                m = M.get((tj,) + q, 0)
                n += 1; nz += (b > 0)
                if b != m:
                    bad += 1; mism.append(((tj,) + q, b, m))
print(f'N={N} |q|<={Q}: {n} weights checked ({nz} nonzero, total dim {sum(M.get(k,0) for k in M)}), mismatches {bad} {mism[:10]}, {time.time()-t0:.1f}s')
