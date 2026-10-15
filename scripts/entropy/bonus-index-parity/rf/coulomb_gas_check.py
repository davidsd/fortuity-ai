# The maxima of the P-/C-twisted classical Cardy exponents equal the unsigned Coulomb-branch graviton gas.
# At u = -1 (x = -1) even windings vanish; an involution acting as -1 on a Cartan (C itself for SU(N): X -> -X^T is
# -1 on the diagonal Cartan, tr_adj = 1-N = -rank; for SU(2)/SO(2N+1)/USp(2N)/G2 a lift of w_0 = -1) gives
# -F = 7 zeta(3) rank, i.e. rank free abelian BPS multiplets counted without signs (sum_{n odd} 8/n^3 = 7 zeta3).
# Prediction (7 zeta3 rank/dimG)^{1/3}/S0, compared with the scan maxima of T-038/twist_classes.py (grid) and of the
# the T-038 analysis (SU(2) 0.979613; SO(7)/USp(6)/G2 0.738577).
import mpmath as mp
S0 = (4*mp.pi**3/27)**(mp.mpf(1)/3)*mp.cos(mp.pi/6)
scan = {'SU(3)': 0.88998, 'SU(4)': 0.82618, 'SU(5)': 0.77747, 'SU(6)': 0.73853, 'SU(8)': 0.67918, 'SU(2)': 0.979613, 'SO(7)': 0.738577}
groups = {'SU(2)': (1, 3), 'SU(3)': (2, 8), 'SU(4)': (3, 15), 'SU(5)': (4, 24), 'SU(6)': (5, 35), 'SU(8)': (7, 63), 'SO(7)': (3, 21), 'USp(6)': (3, 21), 'G2': (2, 14)}
for g, (r, d) in groups.items():
    pred = (7*mp.zeta(3)*r/d)**(mp.mpf(1)/3)/S0
    print("%-6s rank/dim=%2d/%-2d  (7 zeta3 rank/dim)^(1/3)/S0 = %.5f   scan max: %s" % (g, r, d, float(pred), scan.get(g, '-')))
