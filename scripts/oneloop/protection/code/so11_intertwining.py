"""T-033: the one-loop intertwining test of so9_intertwining.py, for the two SO(11) classes of R-011 under the two-block
restriction SO(11) > SO(7) x SO(2) x SO(2) (R-016).  Coefficient maps (so11_block_lift.rho_pair):
    [4,0] class (1;7,3,3): coefficient of x_1^2 y_1^2;   [0,2] class (1;5,5,3): coefficient of x_1^2 y_2^2.
Delta = rho(Q_1^{SO(11)} O) - Q_1^{SO(7)}(rho O) must be Q_0-exact in SO(7) (0,0;6,6,6), Y=8, d_BMN=1 for intertwining.
Run from the repo root:  python3 scripts/oneloop/protection/code/so11_intertwining.py [40|02]
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from so9_intertwining import ROOT, residues, lin, ratrec, ratio, Q1_expr, sector_of, O18, load_rep, Q_expr, values
from so11_block_lift import rho_pair, TR11, TR22
import numpy as np

CASES = {'40': ('so11_nongraviton_rep_1733.json', TR11, TR11, '[4,0] (1;7,3,3), x1^2 y1^2'),
         '02': ('so11_nongraviton_rep_1553.json', TR11, TR22, '[0,2] (1;5,5,3), x1^2 y2^2')}

if __name__ == '__main__':
    t0 = time.time()
    for key in (sys.argv[1:] or ['40', '02']):
        fn, wx, wy, label = CASES[key]
        O, info = load_rep(os.path.join(ROOT, 'data', 'so_sp', fn))
        r0 = rho_pair(O, wx, wy)
        q0 = Q_expr(r0)
        print(f"{label}: O {len(O)} terms; rho(O) {len(r0)} terms in {set(sector_of(m) for m in r0)}; "
              f"Q_0-closed on so(7): {(not q0) or not values(q0, 3, 'SO_odd').any()}", flush=True)
        r1 = rho_pair(Q1_expr(O, 11, 'SO_odd'), wx, wy)
        Q1r0 = Q1_expr(r0, 7, 'SO_odd')
        Q1O18 = Q1_expr(O18(), 7, 'SO_odd')
        Delta = lin((1, r1), (-1, Q1r0))
        print(f"  rho(Q_1 O) {len(r1)} terms; Q_1 rho(O) {len(Q1r0)} terms; Delta {len(Delta)} terms  [{time.time() - t0:.0f}s]",
              flush=True)
        names = ['Delta', 'rho Q1 O', 'Q1 rho O', 'Q1 O18']
        for seed in (11, 12):
            for factor in (1, 2):
                inf, R = residues([Delta, r1, Q1r0, Q1O18], [], (0, 0, 6, 6, 6), 8, 2, 3, seed, factor)
                nz = [bool(np.any(x)) for x in R]
                rat = {names[i]: ratio(R[i], R[3]) for i in range(3)}
                print(f"  [mod exact] seed {seed} x{factor}: {inf}\n     nonzero residue {dict(zip(names, nz))}\n"
                      f"     ratio to Q_1 O_18: {rat}  rational: {{k: ratrec(v) if v is not None else None for k, v in rat.items()}}"
                      .replace('{k: ratrec(v) if v is not None else None for k, v in rat.items()}',
                               str({k: (ratrec(v) if v is not None else None) for k, v in rat.items()}))
                      + f"  [{time.time() - t0:.0f}s]", flush=True)
