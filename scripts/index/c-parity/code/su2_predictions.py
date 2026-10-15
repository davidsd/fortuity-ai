"""T-029 / T-027: index-only predictions for the SU(2) black-hole content per SU(3)_R column up to j = 44, and the
checks to run on T-027's cohomology output when it arrives.

For SU(2), C: X -> -X^T is inner, so the graviton parity P = C (-1)^Y is just (-1)^Y: every multi-graviton has an even
number of letters Y, and Coulomb-branch (diagonal) states too.  Per hw column lam:

    T(lam) = (-1)^|lam| I(lam) - D(lam) = sum over ker(rho) of (-1)^Y
           = #(non-Coulomb gravitons) + #(black holes with even Y) - #(black holes with odd Y),

with I the exact BMN index and D the unsigned diagonal-graviton count.  T < 0 proves odd-Y black holes (from indices
alone).  If, as in all SU(2) data so far (j <= 28 and the (6,6,6) column at j = 36), every SU(2) black hole has odd Y
and there are no non-Coulomb gravitons, then #BH(lam) = -T(lam) exactly.  T-027 tests both.

usage:
  python3 su2_predictions.py                       -> runs/su2_predictions_j44.json (needs runs/cache/diag_dims_N2_j44.pkl
                                                       from `diag_counts.py 2 44`; index computed here)
  python3 su2_predictions.py T027_JSONL [...]      -> compare T-027 run_sectors output with the predictions:
                                                       black holes per column, their 2J (concentration), their Y parity
"""
import os, sys, json, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twisted_index import SUN, integrate, to_int, hw_multiplicities, RUNS, REPO

Q = 22


def predictions():
    idx, _ = to_int(integrate(SUN(2), Q, False)[0])
    dims = pickle.load(open(os.path.join(RUNS, 'cache', 'diag_dims_N2_j44.pkl'), 'rb'))['dims']
    uns = {}
    for sec, d in dims.items():
        q = tuple(sec[1:])
        if sum(q) <= Q and d:
            uns[q] = uns.get(q, 0) + d
    H_I, H_D = hw_multiplicities(idx, Q), hw_multiplicities(uns, Q)
    rows = []
    for lam in sorted(set(H_I) | set(H_D), key=lambda l: (sum(l), l)):
        if lam == (0, 0, 0):
            continue
        T = (-1) ** sum(lam) * H_I.get(lam, 0) - H_D.get(lam, 0)
        if T:
            rows.append(dict(j=2 * sum(lam), lam=lam, irrep=[lam[0] - lam[1], lam[1] - lam[2]], T=T,
                             predicted_BH_if_odd_Y=-T if T < 0 else None))
    return rows


def load_t027(paths):
    """run_sectors jsonl -> {lam: {n: dict(m, h, grav, BH)}} (h from m and r_up of the neighbouring sectors)."""
    recs = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r.get('check') and r.get('N') == 2:
                recs.setdefault(tuple(r['lam']), {})[r['n']] = r
    out = {}
    for lam, rs in recs.items():
        prof = {}
        for n, r in sorted(rs.items()):
            if n - 1 >= 0 and (n - 1) not in rs:
                continue
            h = r['m'] - (rs[n - 1]['r_up'] if n - 1 in rs else 0) - r['r_up']
            prof[n] = dict(m=r['m'], h=h, grav=r['grav'], BH=h - r['grav'])
        out[lam] = prof
    return out


if __name__ == '__main__':
    rows = predictions()
    if len(sys.argv) == 1:
        print('SU(2) columns with T != 0 (j <= 44):')
        for r in rows:
            print(f"  j={r['j']:2} lam={r['lam']} [{r['irrep'][0]},{r['irrep'][1]}]: T={r['T']:3}"
                  + (f"  -> {r['predicted_BH_if_odd_Y']} black hole(s) if all odd-Y" if r['T'] < 0 else '  (even-Y kernel!)'))
        json.dump(dict(description='SU(2) index-only predictions: T = (-1)^|lam| I - D (diagonal count) per hw column, '
                                   'j <= 44; T < 0 forces odd-Y black holes; = -#BH if all BH have odd Y and no '
                                   'non-Coulomb gravitons', rows=rows),
                  open(os.path.join(RUNS, 'su2_predictions_j44.json'), 'w'), indent=1, default=str)
    else:
        cols = load_t027(sys.argv[1:])
        pred = {tuple(r['lam']): r['T'] for r in rows}
        print('column: predicted -T | found BH (2J support, Y parity) | gravitons (2J) | verdict')
        agree = 0
        for lam in sorted(set(cols) | set(pred), key=lambda l: (sum(l), l)):
            if sum(lam) > Q:
                continue
            prof = cols.get(lam, {})
            bh = {n: p['BH'] for n, p in prof.items() if p['BH']}
            gr = {n: p['grav'] for n, p in prof.items() if p['grav']}
            T = sum((-1) ** (sum(lam) - n) * (p['h']) for n, p in prof.items()) if prof else None
            ys = sorted({(sum(lam) - n) % 2 for n in bh})
            nbh = sum(bh.values())
            ok = (-pred.get(lam, 0) == nbh) and ys in ([], [1])
            agree += ok
            if bh or pred.get(lam):
                print(f"  j={2 * sum(lam):2} {lam}: pred {-pred.get(lam, 0)} | BH {nbh} at 2J={sorted(bh)} "
                      f"Y-parity={['even', 'odd'][ys[0]] if len(ys) == 1 else ys} | grav 2J={sorted(gr)} | "
                      f"{'OK' if ok else 'CHECK'}{' (not all 2J present)' if lam in cols and len(prof) < len(cols[lam]) else ''}")
        print(f'{agree} columns agree with "#BH = -T, all odd Y"')
