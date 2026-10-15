import sys, json, time
from fortuity.bmn.symdirect import SymDirect
N = int(sys.argv[1]); q = int(sys.argv[2])
twoJs = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(range(0, 3*q+1))
D = SymDirect(N=N, gen_nmax=N)
res = []
for twoJ in twoJs:
    from fortuity.bmn.words import contents_of_sector
    sec = (twoJ, q, q, q)
    if not contents_of_sector(sec): continue
    res.append(D.analyse(sec))
    json.dump(res, open(f"results/sym_N{N}_q{q}.json", "w"), indent=1)
