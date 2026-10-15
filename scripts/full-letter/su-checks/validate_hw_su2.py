"""SU(2) full letters: sum over dominant weights lam of h_hw(lam) dim[lam] at level n must equal Chang-Lin's d(n)."""
import sys, time
from collections import defaultdict
from fortuity.full.words import sectors_at_level, contents_of_sector
from fortuity.full.sector import Sector
CL = {4: 6, 5: 6, 6: 9, 7: 18, 8: 30, 9: 40, 10: 66, 11: 120, 12: 198}
nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 9
maxlen = 3
def dim_irrep(lam2):
    p, q = (lam2[0] - lam2[1]) // 2, (lam2[1] - lam2[2]) // 2
    return (p + 1) * (q + 1) * (p + q + 2) // 2
for n in range(4, nmax + 1):
    t = time.time()
    secs = [s for s in sectors_at_level(n) if s[2] >= s[3] >= s[4]]
    # group by (2J1, 2J2, lam): all Y
    r_up = {}   # (sec, Y) -> rank Q from (sec+, Y-1)
    m = {}
    total = 0
    for s in secs:
        Ys = sorted({len(c) for c in contents_of_sector(s)})
        for Y in Ys:
            job = Sector(2, s, Y, maxlen, log=lambda *a: None)
            res = job.run()
            assert res['check'], (s, Y, res)
            r_up[(s, Y)] = res['r_up']; m[(s, Y)] = res['m']
    for (s, Y), mm in m.items():
        low = (s[0] - 1, s[1] - 1) + s[2:]
        r_s = r_up.get((low, Y + 1), 0)
        h = mm - r_s - r_up[(s, Y)]
        assert h >= 0, (s, Y, mm, r_s, r_up[(s, Y)])
        total += h * dim_irrep(s[2:])
    print(f"n={n}: sum_lam h_hw dim[lam] = {total}, Chang-Lin {CL.get(n)} ({time.time()-t:.0f}s)", flush=True)
