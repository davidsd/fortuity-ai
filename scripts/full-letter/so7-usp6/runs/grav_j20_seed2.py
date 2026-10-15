import json, pickle, time
from fortuity.full.gravitons import graviton_count
gens=pickle.load(open("runs/gens_j20_Y8.pkl","rb"))
prev={ (tuple(r['sec']),r['group']): r for r in json.load(open("../../../data/fullso/graviton_grav_j20.json")) }
out=[]; bad=0
for d in json.load(open("../../../data/fullso/census_j20.json"))["differences"]:
    sec=tuple(d["sec"])
    for group in ["SO_odd","Sp"]:
        r=graviton_count(group,3,sec,8,gens,seed=2,margin=96,log=lambda s: print(s, flush=True)); out.append(r)
        p=prev[(sec,group)]
        same=(r['exact'],r['rank_out'],r['h'],r['graviton'])==(p['exact'],p['rank_out'],p['h'],p['graviton'])
        if not same: bad+=1; print("   MISMATCH vs seed 1:", (p['exact'],p['rank_out'],p['h'],p['graviton']), flush=True)
json.dump(out, open("../../../data/fullso/graviton_grav_j20_seed2.json","w"), indent=1)
print(f"DONE seed-2 recheck: {len(out)} runs, {bad} mismatches", flush=True)
