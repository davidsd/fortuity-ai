import pickle, json
from hwgrav import hw_graviton_count
gens=pickle.load(open("runs/gens_j20_Y8.pkl","rb"))
R={}
for g in ["SO_odd","Sp"]:
    for l in open({"SO_odd": "../../../data/fullso/so7_full_j20.jsonl", "Sp": "../../../data/fullso/usp6_full_j20.jsonl"}[g]):
        r=json.loads(l); R[(g,tuple(r['sec']),r['Y'])]=r
# expected hw non-graviton from the weight-space reading: (1,1;8,8,4) [0,2]: SO(7) 1; (1,1;8,6,6) [1,0]: SO(7) 1 of 2 extra; (2,2;8,6,6): 1; (-1,3;8,6,6): 1; (0,0;8,8,4): 0
for sec in [(1,1,8,8,4),(2,2,8,6,6),(0,0,8,8,4),(-1,3,8,6,6),(1,1,8,6,6)]:
    for g in ["SO_odd","Sp"]:
        res=hw_graviton_count(g,3,sec,8,gens,log=lambda s: print(s, flush=True))
        r=R[(g,sec,8)]; below=R.get((g,(sec[0]-1,sec[1]-1)+sec[2:],9))
        h=r['m']-r['r_up']-(below['r_up'] if below else 0)
        print(f"==> {g} {sec}: hw h = {h}, hw graviton = {res['graviton']}, hw non-graviton = {h-res['graviton']}  (exact check: {res['exact']} vs scan {r['r_up']})", flush=True)
print("DONE", flush=True)
