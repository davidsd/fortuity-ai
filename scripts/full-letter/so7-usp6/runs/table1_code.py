import time, json
from collections import Counter
from fortuity.full.gravitons import generators_up_to_level, graviton_count
t0 = time.time()
gens = generators_up_to_level(18, 8, log=lambda s: print(s, flush=True))
print(f"generators with level <= 18, Y <= 8: {len(gens)} ({time.time()-t0:.0f}s)", flush=True)
print("by level:", sorted(Counter(g[0][2] + g[0][3] + g[0][4] for g in gens).items()), flush=True)
out = []
for group in ["SO_odd", "Sp"]:
    for sec, Y in [((1, 1, 6, 6, 6), 8), ((0, 0, 6, 6, 6), 8)]:
        out.append(graviton_count(group, 3, sec, Y, gens, log=lambda s: print(s, flush=True), cartan=True))
json.dump(out, open("../data/fullso/graviton_table1_code.json", "w"), indent=1)
print(f"done ({time.time()-t0:.0f}s)")
