"""Level-22 graviton/Coulomb analysis on the feasible differing sectors, one of each mirror pair, generators of level <= 18
(the only ones that can enter a product at level 22).  Single-trace classes of the target sectors themselves: computed
zero for nine of the twelve sectors (runs/grav_j22b.log); for (-1,3;8,8,6), (3,-1;8,8,6), (1,1;8,8,6) the step is skipped
(ASSUMED zero, to be confirmed later); a missed single-graviton class would lower the graviton count by one."""
import time, json, pickle
from fortuity.full.gravitons import graviton_count
import fortuity.full.gdims as dims_group
t0=time.time()
gens=[g for g in pickle.load(open("runs/gens_j20_Y8.pkl","rb")) if g[0][2]+g[0][3]+g[0][4] <= 18]
print(f"generators level<=18: {len(gens)}", flush=True)
C=json.load(open("../../../data/fullso/census_j22.json"))
feasible=[d for d in C["differences"] if dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8) <= 14000 and d["sec"][0] <= d["sec"][1]]
feasible.sort(key=lambda d: dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8))
print("sectors to run (one per mirror pair):", [tuple(d["sec"]) for d in feasible], flush=True)
out=[]
for d in feasible:
    sec=tuple(d["sec"])
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,8,gens,cartan=True,log=lambda s: print(s, flush=True)); r["irrep"]=d["irrep"]; out.append(r)
        json.dump(out, open("../../../data/fullso/graviton_grav_j22.json","w"), indent=1)
print("DONE", flush=True)
