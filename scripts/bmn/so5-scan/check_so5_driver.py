"""Compare records produced by the reconstructed run_so5.sh with the stored scan hw_SO_odd2.jsonl.
usage: python3 check_so5_driver.py NEW.jsonl [STORED.jsonl]"""
import json, sys
new = [json.loads(l) for l in open(sys.argv[1])]
stored = {(tuple(r['lam']), r['n']): r for r in map(json.loads, open(sys.argv[2] if len(sys.argv) > 2 else 'hw_SO_odd2.jsonl'))}
FIELDS = ('m', 'm_d', 'r_up', 'grav', 'check')
bad = missing = 0
for r in new:
    s = stored.get((tuple(r['lam']), r['n']))
    if s is None:
        missing += 1
        continue
    for f in FIELDS:
        if r.get(f) != s.get(f):
            bad += 1
            print('DIFF', r['lam'], r['n'], f, r.get(f), s.get(f))
jmax = max(2 * sum(r['lam']) for r in new)
print(f"{len(new)} recomputed sectors (j <= {jmax}): {bad} differences in {FIELDS}, {missing} not in the stored file")
sys.exit(1 if bad else 0)
