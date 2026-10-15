import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
# Check R-017's new sentence: T-027 found #BH = -T, all with odd Y, per SU(2) column (read-only).
import json
R = REPO + ''
d = json.load(open(R + 'data/su2/bmn_hw_cohomology_j44.json'))
T = {tuple(r['lam']): r['T'] for r in json.load(open(R + 'data/su2/kernel_parity_j44.json'))['rows']}
good = bad = 0; tot = 0; inc = []
for c in d['columns']:
    lam = tuple(c['lam'])
    if c['euler'] != c['index']:
        inc.append((c['j'], lam, [s['twoJ'] for s in c['sectors']], c['euler'], c['index'])); continue
    bh = sum(s['BH'] for s in c['sectors'])
    even = sum(s['BH'] for s in c['sectors'] if (sum(lam) - s['twoJ']) % 2 == 0)
    if bh == -T.get(lam, 0) and even == 0: good += 1; tot += bh
    else: bad += 1; print('MISMATCH', c['j'], lam, bh, T.get(lam, 0), even)
print('columns with euler == index:', good + bad, 'agree:', good, 'disagree:', bad, 'total BH:', tot)
print('incomplete (euler != index):', inc)
