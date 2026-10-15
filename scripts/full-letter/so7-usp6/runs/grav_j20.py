import time, json, pickle, os
from fortuity.full.gravitons import generators_up_to_level, graviton_count
t0=time.time()
fn="runs/gens_j20_Y8.pkl"
if os.path.exists(fn): gens=pickle.load(open(fn,"rb"))
else:
    gens=generators_up_to_level(20, 8, log=lambda s: print(s, flush=True)); pickle.dump(gens, open(fn,"wb"))
print(f"generators level<=20, Y<=8: {len(gens)} ({time.time()-t0:.0f}s)", flush=True)
# re-validation of Table 1 with the Weyl-transported generators
for group,sec in [("SO_odd",(1,1,6,6,6)),("SO_odd",(0,0,6,6,6)),("Sp",(1,1,6,6,6)),("Sp",(0,0,6,6,6))]:
    graviton_count(group,3,sec,8,gens,log=lambda s: print("  [table1]"+s, flush=True))
C=json.load(open("../../../data/fullso/census_j20.json"))
out=[]
for d in C["differences"]:
    sec=tuple(d["sec"]); Y=d["Y"]
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,Y,gens,log=lambda s: print(s, flush=True)); r["irrep"]=d["irrep"]; out.append(r)
        json.dump(out, open("../../../data/fullso/graviton_grav_j20.json","w"), indent=1)
print("DONE", flush=True)
