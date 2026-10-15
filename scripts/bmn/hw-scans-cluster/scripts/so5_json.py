"""Write data/so5/bmn_hw_cohomology_j<JMAX>.json in the format of data/so5 from run_sectors JSONL files."""
import sys, json, glob
jmax = int(sys.argv[1]); out = sys.argv[2]; files = [f for pat in sys.argv[3:] for f in glob.glob(pat)]
rows = {}
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r.get('group') not in (None, 'SO_odd') or r['N'] != 2:
            continue
        if 2 * sum(r['lam']) <= jmax:
            rows[(tuple(r['lam']), r['n'])] = r
cols, per_level, fails = [], {}, 0
for (lam, n) in sorted(rows, key=lambda k: (2 * sum(k[0]), k[0], k[1])):
    r = rows[(lam, n)]
    rin = rows[(lam, n - 1)]['r_up'] if (lam, n - 1) in rows else 0
    h = r['m'] - r['r_up'] - rin
    j = 2 * sum(lam)
    ng = h - r['grav']
    fails += not r['check']
    cols.append(dict(j=j, lam=list(lam), rep=[lam[0] - lam[1], lam[1] - lam[2]], twoJ=n, E=j / 2 - n / 2, m=r['m'],
                     r_up=r['r_up'], h=h, grav=r['grav'], nongrav=ng, check=r['check'], maxlen=r.get('maxlen')))
    pl = per_level.setdefault(str(j), dict(hw_classes=0, graviton=0, nongraviton=0))
    pl['hw_classes'] += h; pl['graviton'] += r['grav']; pl['nongraviton'] += ng
json.dump(dict(group="SO(5) = Sp(2) (GROUP=SO_odd, N=2)", jmax=jmax, sectors=len(cols), checks_failed=fails,
               per_level=per_level, columns=cols), open(out, 'w'), indent=1)
print(len(cols), "sectors; checks failed", fails, "; per level:", {k: v for k, v in per_level.items() if int(k) >= 24})
