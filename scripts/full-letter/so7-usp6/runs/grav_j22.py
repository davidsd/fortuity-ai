import time, json, pickle, os
from fortuity.full.gravitons import graviton_count
while not os.path.exists("runs/gens_j22_Y8.pkl"): time.sleep(60)
time.sleep(5)
gens=pickle.load(open("runs/gens_j22_Y8.pkl","rb")); print(f"generators level<=22: {len(gens)}", flush=True)
C=json.load(open("../../../data/fullso/census_j22.json"))
import fortuity.full.gdims as dims_group
out=[]
feasible=[d for d in C["differences"] if dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8) <= 14000]
print("feasible sectors (weight-space dim <= 14000):", len(feasible), "of", len(C["differences"]), flush=True)
for d in sorted(feasible, key=lambda d: dims_group.sector_dim("SO_odd",3,tuple(d["sec"]),8)):
    sec=tuple(d["sec"])
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,8,gens,cartan=True,log=lambda s: print(s, flush=True)); r["irrep"]=d["irrep"]; out.append(r)
        json.dump(out, open("../../../data/fullso/graviton_grav_j22.json","w"), indent=1)
print("DONE", flush=True)
