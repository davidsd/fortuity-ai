"""Sum of Q-cohomology dimensions over all full-letter sectors at level n for SU(2), vs Chang-Lin Table 3 d_SU(2)(n)."""
import sys, time
from fortuity.full.words import sectors_at_level
from fortuity.full.cohom import FullDirect
CL = {4: 6, 5: 6, 6: 9, 7: 18, 8: 30, 9: 40, 10: 66, 11: 120, 12: 198, 13: 324, 14: 537}
nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D = FullDirect(N=2, maxlen=3, verbose=False)
for n in range(4, nmax + 1):
    t = time.time(); tot = 0
    secs = sectors_at_level(n)
    for s in secs:
        tot += D.dimH(s)
    print(f"n={n}: sectors {len(secs)}, total cohomology {tot}, Chang-Lin {CL.get(n)}  ({time.time()-t:.0f}s)", flush=True)
