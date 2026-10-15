"""SU(3), j=30, weight (7,4,4) BMN column via the full-letter hw complex: sectors (2J,2J;14,8,8) at Y = 15 - 2J."""
import json, sys, time, resource
from fortuity.full.sector import Sector
maxlen = int(sys.argv[1]) if len(sys.argv) > 1 else 6
twoJs = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [4, 5, 6]
out = open(f'bmn_check_ml{maxlen}.jsonl', 'a')
for n in twoJs:
    sec = (n, n, 14, 8, 8); Y = 15 - n
    print(f"== sector {sec} Y={Y}", flush=True)
    res = Sector(3, sec, Y, maxlen, log=lambda *a: print(*a, flush=True)).run()
    res['maxrss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    print(json.dumps(res), flush=True)
    out.write(json.dumps(res) + '\n'); out.flush()
