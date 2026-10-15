import time, pickle
from fortuity.full.gravitons import generators_up_to_level
t0=time.time()
gens=generators_up_to_level(22, 8, workers=10, log=lambda s: print(s, flush=True))
pickle.dump(gens, open("runs/gens_j22_Y8.pkl","wb"))
print(f"generators level<=22, Y<=8: {len(gens)} ({time.time()-t0:.0f}s) DONE", flush=True)
