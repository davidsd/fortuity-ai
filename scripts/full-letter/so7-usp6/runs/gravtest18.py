import time, json
from fortuity.full.gravitons import generators_up_to_level, graviton_count
t0=time.time()
gens=generators_up_to_level(18, 8)
print(f"generators with level <= 18, Y <= 8: {len(gens)} ({time.time()-t0:.0f}s)", flush=True)
from collections import Counter
print("by level:", sorted(Counter(g[0][2]+g[0][3]+g[0][4] for g in gens).items()), flush=True)
out=[]
for group in ["SO_odd","Sp"]:
    for sec,Y in [((1,1,6,6,6),8),((0,0,6,6,6),8)]:
        r=graviton_count(group,3,sec,Y,gens,log=lambda s: print(s, flush=True)); out.append(r)
json.dump(out, open("../../../data/fullso/graviton_gravtest18.json","w"), indent=1)
print("expected (Table 1): SO(7) (1,1;6,6,6): 903/559/h124/grav123; USp(6): 903/559/123/123; (0,0;6,6,6): SO(7) 826/741/85/85, USp(6) 825/741/84/84")
