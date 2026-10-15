"""SU(4) C-twisted BMN index beyond j = 30: where does the index alone force black holes?

(-1)^|lam| I_C(lam) = #P+ - #P- in the cohomology of the SU(3)_R column lam (R-017). Gravitons are P-even, so a
negative value forces at least that many P-odd, hence non-graviton, classes, with no Coulomb-branch input (as for
SU(3) at j = 42, 44).

The twisted Haar integral (fortuity.bmn.twistedindex.integrate, 2-dim torus for SU(4)) is done twice, on grids of
Mg = 4Q + 12 and Mg + 6 points per direction. The two must give the same integers (no aliasing, rounding under control).
Columns with j <= 30 are compared with the direct cohomology (data/su4, R-008, R-014), where there are no P-odd classes,
so the signed twisted index must equal the total cohomology.

usage: PYTHONPATH=code python3 su4_twisted_scan.py QMAX        (j <= 2 QMAX; output ../runs/su4_twisted_scan_j<2QMAX>.json)
"""
import json
import os
import sys
import time

from fortuity.bmn.twistedindex import SUN, integrate, to_int, hw_multiplicities

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
Q = int(sys.argv[1])
t0 = time.time()
G = SUN(4)
res = []
for Mg in (4 * Q + 12, 4 * Q + 18):
    tw, nrm = integrate(G, Q, True, Mg=Mg)
    tw, err = to_int(tw)
    res.append((Mg, hw_multiplicities(tw, Q), err))
    print(f'Mg = {Mg}: max rounding error {err:.2e} ({time.time() - t0:.0f}s)', flush=True)
(m1, h1, e1), (m2, h2, e2) = res
assert h1 == h2, 'the two grids disagree'
signed = {lam: (-1) ** sum(lam) * v for lam, v in h1.items()}
# validation against the direct SU(4) cohomology, j <= 30 (no P-odd classes there)
ref = {}
for f in ('data/su4/bmn_hw_cohomology_j28.json', 'data/su4/bmn_hw_cohomology_j30.json'):
    for c in json.load(open(os.path.join(REPO, f)))['columns']:
        ref[tuple(c['lam'])] = sum(s['h'] for s in c['sectors'])
cmp = [l for l in ref if sum(l) <= Q]
bad = [(l, signed.get(l, 0), ref[l]) for l in cmp if signed.get(l, 0) != ref[l]]
print(f'j <= 30 validation: {len(cmp)} columns, mismatches {bad}', flush=True)
neg = sorted([(l, v) for l, v in signed.items() if v < 0], key=lambda t: (sum(t[0]), t[0]))
for l, v in neg:
    print(f'  forced: lam {l} (j = {2 * sum(l)}, [{l[0] - l[1]},{l[1] - l[2]}]): (-1)^|lam| I_C = {v} -> >= {-v} P-odd',
          flush=True)
by_j = {}
for l, v in signed.items():
    by_j.setdefault(2 * sum(l), []).append(v)
summary = {j: dict(columns=len(v), negative=sum(1 for x in v if x < 0), min=min(v)) for j, v in sorted(by_j.items())
           if j >= 26}
print(json.dumps(summary), flush=True)
out = dict(description='SU(4) C-twisted BMN index per SU(3)_R hw column, (-1)^|lam| I_C = #P+ - #P- (R-017); negative '
                       'values force P-odd (non-graviton) classes with no Coulomb input',
           qmax=Q, grids=[m1, m2], rounding_error=[e1, e2], validation_j30=dict(columns=len(cmp), mismatches=bad),
           forced=[dict(lam=list(l), j=2 * sum(l), signed_twisted=v, forced_p_odd=-v) for l, v in neg],
           summary_by_j=summary,
           columns=[dict(lam=list(l), j=2 * sum(l), signed_twisted=v) for l, v in
                    sorted(signed.items(), key=lambda t: (sum(t[0]), t[0])) if sum(l) >= 16],
           time=round(time.time() - t0, 1))
json.dump(out, open(os.path.join(HERE, '..', 'runs', f'su4_twisted_scan_j{2 * Q}.json'), 'w'), indent=1, default=str)
