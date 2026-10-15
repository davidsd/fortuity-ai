import os as _os, sys as _sys
REPO = (_os.environ.get('REPO') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), *['..'] * 5))).rstrip('/') + '/'
_sys.path.insert(0, REPO + 'code'); _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import re, ast, runpy, io, contextlib
# re-run the exact computation silently and grab its globals
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'su2_coulomb_exact.py'))
uns = g['uns']
terms = [(0, 0, 1), (1, 0, -1), (0, 1, -1), (2, 1, 1), (1, 2, 1), (2, 2, -1)]
txt = open(REPO + 'scripts/index/c-parity/c19/out/a3_forced.txt').read()
su2 = txt.split('diag_dims_N2_j44.pkl')[1]
n = bad = 0
for m in re.finditer(r'j=(\d+) \((\d+), (\d+), (\d+)\) T=\s*(-?\d+) D=\s*(\d+)\s+terms=(\[.*?\])\s', su2):
    lam = tuple(int(m.group(i)) for i in (2, 3, 4))
    stored = ast.literal_eval(m.group(7))
    mine = [(sg, uns.get((lam[0] + i, lam[1] - i + j, lam[2] - j), 0)) for i, j, sg in terms]
    n += 1
    if stored != mine:
        bad += 1; print('MISMATCH', lam, stored, mine)
print('SU(2) columns compared:', n, ' mismatches in the six per-weight unsigned dims:', bad)
