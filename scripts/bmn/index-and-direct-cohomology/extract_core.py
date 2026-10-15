"""Divide Z - Z_grav by descendant factor prod_I (1 - t^2 x_I) and hair factor prod_I 1/(1 - t^8 x_I)."""
import sys, pickle, numpy as np
from fortuity.bmn.fullindex import SeriesBatch, su3_multiplicities
f = sys.argv[1]
hair = sys.argv[2] if len(sys.argv) > 2 else "w2"
d = pickle.load(open(f, "rb"))
D = (d['Z'] - d['Zg']).astype(np.complex128); A = d['A']; jmax = d['jmax']
sb = SeriesBatch(1, jmax, A); sb.S[0] = D
X = [(1, 0), (0, -1), (-1, 1)]   # (a,b) of x_1, x_2, x_3
one = np.ones(1, dtype=np.complex128)
# divide by prod_I (1 - t^2 x_I): multiply by geometric series  (bosonic-type factor)
for (a, b) in X:
    sb.mul_factor(one, 2, a, b, fermionic=False)
if hair == "w2":
    # divide by prod_I 1/(1 - t^8 x_I): multiply by (1 - t^8 x_I)
    for (a, b) in X:
        sb.mul_factor(one, 8, a, b, fermionic=True)
core = np.rint(sb.S[0].real).astype(np.int64)
print("Z_core unrefined:", list(core.sum(axis=(1, 2))))
for j in range(0, jmax + 1, 2):
    m = su3_multiplicities(core[j], A)
    if m: print(f"j={j}: ", {f"[{int(p)},{int(q)}]": int(c) for (p, q), c in sorted(m.items())})
