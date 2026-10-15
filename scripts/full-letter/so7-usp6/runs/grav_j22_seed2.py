import json, pickle
from fortuity.full.gravitons import graviton_count
gens=[g for g in pickle.load(open("runs/gens_j20_Y8.pkl","rb")) if g[0][2]+g[0][3]+g[0][4] <= 18]
prev={(tuple(r['sec']),r['group']): r for r in json.load(open("../../../data/fullso/graviton_grav_j22.json"))}
bad=0; n=0
for (sec,group),p in sorted(prev.items(), key=lambda kv: kv[1]['dim']):
    r=graviton_count(group,3,sec,8,gens,seed=2,margin=96,cartan=True,log=lambda s: print(s, flush=True)); n+=1
    same=(r['exact'],r['rank_out'],r['h'],r['graviton'],r['coulomb_graviton'])==(p['exact'],p['rank_out'],p['h'],p['graviton'],p['coulomb_graviton'])
    if not same: bad+=1; print("   MISMATCH vs seed 1:", (p['exact'],p['rank_out'],p['h'],p['graviton'],p['coulomb_graviton']), flush=True)
print(f"DONE seed-2 recheck level 22: {n} runs, {bad} mismatches", flush=True)
