"""Level-22 graviton/Coulomb analysis without the full level-22 generator build: products landing at level 22 use
generators of level <= 18 (two or more factors; the smallest generator has level 4, so levels 19-21 cannot occur and
level 20 cannot pair with anything) plus the level-22 single-trace classes of the target sectors themselves."""
import time, json, pickle
from fortuity.full.gravitons import graviton_count, single_trace_cohomology
import fortuity.full.gdims as dims_group
t0=time.time()
gens=[g for g in pickle.load(open("runs/gens_j20_Y8.pkl","rb")) if g[0][2]+g[0][3]+g[0][4] <= 18]
print(f"generators level<=18: {len(gens)}", flush=True)
C=json.load(open("../../../data/fullso/census_j22.json"))
feasible=sorted([d for d in C["differences"] if dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8) <= 14000], key=lambda d: dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8))
print("feasible sectors:", len(feasible), "of", len(C["differences"]), flush=True)
# level-22 single-trace classes in the target sectors (they are dominant weights)
extra=[]
for d in feasible:
    sec=tuple(d["sec"]); words,reps=single_trace_cohomology(sec,8)
    for row in reps:
        import numpy as np
        extra.append((sec,8,{words[i]:int(row[i]) for i in np.flatnonzero(row)}))
    print(f"  single-trace classes at {sec} Y=8: {reps.shape[0]} ({time.time()-t0:.0f}s)", flush=True)
gens=gens+extra
out=[]
for d in feasible:
    sec=tuple(d["sec"])
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,8,gens,cartan=True,log=lambda s: print(s, flush=True)); r["irrep"]=d["irrep"]; out.append(r)
        json.dump(out, open("../../../data/fullso/graviton_grav_j22.json","w"), indent=1)
print("DONE", flush=True)
