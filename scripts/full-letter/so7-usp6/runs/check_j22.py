import json, collections, glob, sys
sys.path.insert(0,".")
from fortuity.full.levels import cohomology
from fortuity.full.levels import load as load_many   # same function, renamed
from sector_group import SectorG
def bmn_h(path, j):
    rows=[json.loads(l) for l in open(path) if l.strip()]
    rows=[r for r in rows if 2*sum(r['lam'])==j]
    by=collections.defaultdict(dict)
    for r in rows: by[tuple(r['lam'])][r['n']]=r
    return {(lam,n): r['m']-r['r_up']-(d[n-1]['r_up'] if (n-1) in d else 0) for lam,d in by.items() for n,r in d.items() if r['m']-r['r_up']-(d[n-1]['r_up'] if (n-1) in d else 0)}
for g,bmn in [("SO_odd","../../../data/g2/so7_hw_cohomology_j22_j24.jsonl"),("Sp","../../../data/g2/usp6_hw_cohomology_j20_j24.jsonl")]:
    R=load_many(sorted([{"SO_odd": "../../../data/fullso/so7_full_j22.jsonl", "Sp": "../../../data/fullso/usp6_full_j22.jsonl"}[g]])); H=cohomology(R)
    fb={(tuple(v//2 for v in sec[2:]),sec[0]):h for (sec,Y),h in H.items() if 22-2*Y-(sec[0]+sec[1])==0 and sec[0]==sec[1]}
    B=bmn_h(bmn,22); bad=[(k,fb.get(k,0),B.get(k,0)) for k in set(fb)|set(B) if fb.get(k,0)!=B.get(k,0)]
    print(f"{g} level 22 BMN cross-check: full-scan D=0 {sum(fb.values())} in {len(fb)} sectors; BMN engine {sum(B.values())} in {len(B)}; mismatches {len(bad)} {bad[:4]}", flush=True)
C=json.load(open("../../../data/fullso/census_j22.json"))
for g in ["SO_odd","Sp"]:
    R=load_many(sorted([{"SO_odd": "../../../data/fullso/so7_full_j22.jsonl", "Sp": "../../../data/fullso/usp6_full_j22.jsonl"}[g]])); bad=0; n=0
    for d in C["differences"]:
        sec=tuple(d["sec"]); Y=d["Y"]
        for s,y in [(sec,Y),((sec[0]-1,sec[1]-1)+sec[2:],Y+1)]:
            if (s,y) not in R: continue
            r0=R[(s,y)]; r=SectorG(g,3,s,y,11,seed=2,log=lambda *a:None).run(); n+=1
            if (r['m'],r['r_up'],r['check'])!=(r0['m'],r0['r_up'],True): bad+=1; print("MISMATCH",g,s,y,(r0['m'],r0['r_up']),(r['m'],r['r_up'],r['check']),flush=True)
    print(f"{g} level 22 seed-2 recheck of differing sectors and the sectors below them: {n} sectors, {bad} mismatches", flush=True)
