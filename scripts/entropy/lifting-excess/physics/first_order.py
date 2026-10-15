"""T-041 physics check 5: meaning of the first-order coefficient r1 at equal charges.

(a) The two first-order coefficients separately (S/K = L + eps C + O(eps^2), eps = (dimG/j)^{1/3} at j1 = j2 = j):
    protected C_p and classical C_c, from the formulas of classical_max.py (independent of t038 code).
(b) Exact check of the protected expansion against the exact large-N black-hole formula (CJKKKLMP 2501.17217
    eqs. sh1, entbhc; Kim-Lee hep-th/0607085), in units dimG = 1 (N^2 -> dimG): equal charges Q, J1 = J2 = J on the
    sheet; KKLP's j = J + Rbar = J + Q (arXiv:2606.27955); K = 3 j^{2/3}; eps = j^{-1/3}.
    (S_BH/K - S0)/eps -> C_p, and the remainder is O(eps^2).
Runtime: about 3 s.
"""
import time
import mpmath as mp
import classical_max as cm

mp.mp.dps = 30
PI = mp.pi

if __name__ == "__main__":
    t0 = time.time()
    lam0 = (4 * PI ** 3 / 27) ** (mp.mpf(1) / 3) * mp.expjpi(mp.mpf(1) / 6)
    xi = [mp.mpf(0)] * 3
    lam, mu = cm.protected_closed(xi, lam0)
    vp, mpr, gp = cm.saddle(2 * PI, xi, (mu[0], mu[1]))
    (vc, mc, gc), Phi = cm.classical_max(xi, mpr)
    Lp, Lc = mp.re(vp), mp.re(vc)
    Cp, Cc = mp.re(cm.first_order_C(mpr, gp)), mp.re(cm.first_order_C(mc, gc))
    print("(a) xi = 0: L_p = %s, C_p = %s;  L_c = %s, C_c = %s" % tuple(mp.nstr(x, 8) for x in (Lp, Cp, Lc, Cc)))
    print("    relative slopes: C_p/L_p = %s, C_c/L_c = %s;  r1 = %s;  absolute excess per K: (L_c-L_p) + eps (C_c-C_p)"
          " = %s + %s eps" % tuple(mp.nstr(x, 6) for x in (Cp / Lp, Cc / Lc, (Cc * Lp - Lc * Cp) / Lp ** 2, Lc - Lp,
                                                             Cc - Cp)))
    print("    closed form C_p = (2/3) Re(H/g), H = -2 pi^2/9, g = (4 pi^3 i/27)^(1/3): %s" %
          mp.nstr(mp.re(mp.mpf(2) / 3 * (-2 * PI ** 2 / 9) / lam0), 8))
    print("(b) exact black-hole entropy at equal charges, J1 = J2 (units dimG = 1):")
    print("        j            eps        S_BH/K          (S_BH/K - S0)/eps    remainder/eps^2")
    S0 = Lp
    for k in (3, 4, 5, 6, 8, 10):
        j = mp.mpf(10) ** k
        # unknowns Q, J with J + Q = j and sheet: J^2/2 + Q^3 = (3Q + 1/2)(3Q^2 - J)
        F = lambda Q: (j - Q) ** 2 / 2 + Q ** 3 - (3 * Q + mp.mpf(1) / 2) * (3 * Q ** 2 - (j - Q))
        Q = mp.findroot(F, (j / 4) ** (mp.mpf(2) / 3))
        J = j - Q
        S = 2 * PI * mp.sqrt(3 * Q ** 2 - J)
        K = 3 * j ** (mp.mpf(2) / 3)
        eps = j ** (-mp.mpf(1) / 3)
        a = (S / K - S0) / eps
        print("  %12s  %10s  %14s  %18s  %14s" % (mp.nstr(j, 3), mp.nstr(eps, 4), mp.nstr(S / K, 10), mp.nstr(a, 8),
                                               mp.nstr((S / K - S0 - Cp * eps) / eps ** 2, 5)))
    print("time %.1f s" % (time.time() - t0))
