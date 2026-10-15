"""SU(4) threshold energy: direct cohomology in the [2,0] highest-weight sectors (2J; 6,4,4) at j=28, from high J down."""
import sys, json
from fortuity.bmn.symdirect import SymDirect
from fortuity.bmn.words import contents_of_sector
twoJs = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [8, 7, 6, 5]
D = SymDirect(N=4, symmetrize=False, maxlen=20)
res = []
for twoJ in twoJs:
    sec = (twoJ, 6, 4, 4)
    if not contents_of_sector(sec): continue
    res.append(D.analyse(sec))
    json.dump(res, open('results/su4_energy_j28.json', 'w'), indent=1)
