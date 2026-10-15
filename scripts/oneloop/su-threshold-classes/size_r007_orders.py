"""Sizes of the [3,0] highest-weight complexes around R-007's possible order-n target sectors (n = 2..5):
target (4,4;14,8,8) at Y = 11-n, its Q_0-preimage (5,5;14,8,8) at Y = 10-n and its Q_0-image (3,3;14,8,8) at Y = 12-n
(all with n derivative units, sum |S| = 15 - 3n)."""
import sys, time
from fastcontents import prime_cache, contents_at
from fortuity.full.sector import hw_mult_by_degree

for n in [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5]:
    for lab, sec, Y in [('pre', (5, 5, 14, 8, 8), 10 - n), ('target', (4, 4, 14, 8, 8), 11 - n), ('image', (3, 3, 14, 8, 8), 12 - n)]:
        t = time.time()
        prime_cache(sec, Y)
        nc = len(contents_at(sec, Y))
        m = hw_mult_by_degree(3, sec, Y)
        print(f"n={n} {lab:6s} sec={sec} Y={Y}: contents {nc}, [3,0] hw multiplicity {sum(m.values())} {dict(sorted(m.items()))} ({time.time()-t:.0f}s)", flush=True)
