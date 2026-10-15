import json, pickle
from fortuity.full.gravitons import graviton_count
gens=pickle.load(open("runs/gens_j20_Y8.pkl","rb"))
out=[]
# validation: BMN (1,1;6,6,6) Y=8: all 123 gravitons Coulomb-type for both groups (R-015: SO(7)/Sp(3) Coulomb counts equal; no non-Coulomb at j=18)
for group in ["SO_odd","Sp"]:
    out.append(graviton_count(group,3,(1,1,6,6,6),8,gens,cartan=True,log=lambda s: print("  [table1]"+s, flush=True)))
    out.append(graviton_count(group,3,(0,0,6,6,6),8,gens,cartan=True,log=lambda s: print("  [table1]"+s, flush=True)))
for d in json.load(open("../../../data/fullso/census_j20.json"))["differences"]:
    sec=tuple(d["sec"])
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,8,gens,cartan=True,log=lambda s: print(s, flush=True)); r["irrep"]=d["irrep"]; out.append(r)
        json.dump(out, open("../../../data/fullso/graviton_coulomb_j20.json","w"), indent=1)
print("DONE", flush=True)
