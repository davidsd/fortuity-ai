"""Fast enumeration of full-letter sectors at a level: multisets of letters (a, b, S) with sum of 2q = j, grouped by
(2J1, 2J2, dominant 2q).  usage: python enum_level.py J [J ...]  -> prints sizes and writes pairs_J.json"""
import sys, json, collections, time
from fortuity.full.letters import admissible, charges2, THETA
from itertools import combinations
def types(jmax):
    out=[]
    for a in range(jmax//3+1):
        for b in range(jmax//3+1-a):
            for k in range(4):
                for S in combinations(THETA,k):
                    if admissible(a,b,S):
                        c=charges2((a,b,S)); lvl=sum(c[2:])
                        if 0<lvl<=jmax: out.append(((a,b,S),c,lvl))
    return out
def enumerate_level(j):
    T=sorted(types(j), key=lambda t:-t[2])
    sectors=collections.defaultdict(lambda: collections.Counter())   # (J1,J2,dom q) -> Counter of Y
    def rec(i, rem, J1, J2, q, Y):
        if rem==0:
            dom=tuple(sorted(q, reverse=True)); sectors[(J1,J2,dom)][Y]+=1; return
        for k in range(i, len(T)):
            L,c,lvl=T[k]
            if lvl>rem: continue
            rec(k, rem-lvl, J1+c[0], J2+c[1], (q[0]+c[2],q[1]+c[3],q[2]+c[4]), Y+1)
    rec(0, j, 0, 0, (0,0,0), 0)
    return sectors
for j in [int(x) for x in sys.argv[1:]]:
    t0=time.time(); S=enumerate_level(j)
    pairs=sorted({(k[0],k[1]) for k in S})
    nsec=sum(len(v) for v in S.values()); ncont=sum(sum(v.values()) for v in S.values()); maxY=max(y for v in S.values() for y in v)
    print(f"j={j}: columns {len(S)}, (column,Y) sectors {nsec}, contents {ncont}, max Y {maxY}, (2J1,2J2) pairs {len(pairs)} ({time.time()-t0:.1f}s)")
    print("   pairs:", pairs)
    json.dump({"pairs":pairs, "columns":[[list(k),dict(v)] for k,v in S.items()]}, open(f"pairs_{j}.json","w"))
