from fortuity.bmn.fullindex import partition_dims
from fortuity.bmn.direct import Direct
pd = partition_dims(6, 14)
d = Direct(6, maxlen=100, verbose=False)
for s in [(2, 2, 2, 2), (2, 1, 2, 4), (2, 1, 3, 3), (3, 2, 2, 3)]:
    pts = d.points(s)
    print(s, "exact dim", pd.get(s), "nwords", len(d.words(s)), "Direct.dimV", d.dimV(s),
          "npts", pts.P, "k", pts.k, "cols per degree", {dd: pts.ncomp[dd] * pts.P for dd in pts.degrees})
r = d.analyse((2, 2, 2, 2)); print({k: r[k] for k in ('dimV', 'rQ', 'rQup', 'coh', 'grav', 'nongrav')})
d2 = Direct(6, maxlen=100, verbose=False, margin=2000)
r = d2.analyse((2, 2, 2, 2)); print("many points:", {k: r[k] for k in ('dimV', 'rQ', 'rQup', 'coh', 'grav', 'nongrav')})
