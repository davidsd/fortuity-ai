"""The C-twisted index functions moved to the shared library as fortuity.bmn.twistedindex (R-017, 2026-09-29).
This file keeps the comparison with the cohomology data: python3 twisted_index.py N [QMAX=15] [PLAIN_QMAX=QMAX]."""
import os, sys, json, itertools, time
import numpy as np
from fortuity.bmn.twistedindex import (SUN, zeros, mul_h, letter_series, Series, traces, integrate, to_int,   # noqa
                                       hw_multiplicities, kernel_parity)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
RUNS = os.path.join(HERE, '..', 'runs')

DATA = {3: ['data/su3/bmn_hw_cohomology_j30.json'],
        4: ['data/su4/bmn_hw_cohomology_j28.json', 'data/su4/bmn_hw_cohomology_j30.json']}
P_ODD = {3: {(7, 4, 4): 1}, 4: {}}


if __name__ == '__main__':
    N = int(sys.argv[1])
    QMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    PQ = int(sys.argv[3]) if len(sys.argv) > 3 else QMAX
    t0 = time.time()
    G = SUN(N)
    ref_index, ref_total = {}, {}
    for f in DATA[N]:
        for c in json.load(open(os.path.join(REPO, f)))['columns']:
            ref_index[tuple(c['lam'])] = c['index']
            ref_total[tuple(c['lam'])] = sum(s['h'] for s in c['sectors'])
    out = dict(description=f'SU({N}) BMN C-twisted index per SU(3)_R hw column vs the prediction from the cohomology data '
                           'and graviton parity: (-1)^|lam| I_C = total cohomology - 2 #(P-odd classes)', N=N, qmax=QMAX)
    if PQ > 0:
        pl, err = to_int(integrate(G, PQ, False)[0])
        hw_pl = hw_multiplicities(pl, PQ)
        cmp = [l for l in ref_index if sum(l) <= PQ]
        bad = [(l, hw_pl.get(l, 0), ref_index[l]) for l in cmp if hw_pl.get(l, 0) != ref_index[l]]
        print(f'SU({N}) ordinary index (validation, j <= {2 * PQ}): max rounding error {err:.2e}; columns compared '
              f'{len(cmp)}; mismatches {len(bad)} {bad[:5]}  ({time.time() - t0:.0f}s)', flush=True)
        out['ordinary_index_check'] = dict(qmax=PQ, columns=len(cmp), mismatches=bad)
    tw, nrm = integrate(G, QMAX, True)
    tw, err = to_int(tw)
    hw_tw = hw_multiplicities(tw, QMAX)
    print(f'SU({N}) twisted index (j <= {2 * QMAX}): max rounding error {err:.2e}; mean weight {nrm.real:.4f}  '
          f'({time.time() - t0:.0f}s)', flush=True)
    rows, bad = [], []
    for lam in sorted(ref_total, key=lambda l: (sum(l), l)):
        if sum(lam) > QMAX:
            continue
        pred = ref_total[lam] - 2 * P_ODD[N].get(lam, 0)
        got = (-1) ** sum(lam) * hw_tw.get(lam, 0)
        rows.append(dict(lam=lam, j=2 * sum(lam), total_cohomology=ref_total[lam], ordinary_index=ref_index[lam],
                         twisted_index=hw_tw.get(lam, 0), predicted_signed=pred, ok=(got == pred)))
        if got != pred:
            bad.append((lam, got, pred))
    extra = [l for l in hw_tw if sum(l) <= QMAX and l not in ref_total and l != (0, 0, 0)]
    print(f'SU({N}) twisted vs prediction: {len(rows)} columns, mismatches {len(bad)} {bad[:8]}; '
          f'twisted content outside the data columns: {extra[:5]}')
    for r in rows:
        if P_ODD[N].get(r['lam']):
            print('  P-odd column:', r)
    out.update(rows=rows, mismatches=bad, outside=extra)
    json.dump(out, open(os.path.join(RUNS, f'su{N}_twisted_index_j{2 * QMAX}.json'), 'w'), indent=1, default=str)
