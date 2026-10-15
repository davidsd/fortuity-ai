"""T-038 derivation, Cardy-limit numerics (each part runs in a few seconds).

Notation: KKLP arXiv:2606.27955 eqs. (4.4), (4.14): f(mu) = -[Li3(1) - 3Li3(u) + 3Li3(u^2) - Li3(u^3)], u = e^{-mu};
at mu = -i phi the entropy is S ~ 3 (dimG j1 j2)^{1/3} Re[f^{1/3}] (root with the largest real part), eq. (4.12).
More generally G(y) := sum_n a_n y^n with a_n = -(1-u^n)^3/n^3, so f = G(1), and a twisted single-letter character
chi_n multiplies a_n.

(A) KKLP: g(phi) = max_root Re f(-i phi)^{1/3}; phi_0, ratio g(phi_0)/g(2pi/3) (KKLP 4.15: 2.2247, 1.01213), and the window
    phi in (2pi/3, phi_1) where g > g(2pi/3).
(B) Roots of unity x = e^{2 pi i k/m} (Cardy: u^3 = x): Z(x) is preserved by d_n iff m | n; is its growth above S?
(C) C-twisted SU(N), trivial twisted holonomy: chi_even = N^2-1, chi_odd = 1-N; large N: F_C -> N^2 f(2mu)/8.
    P-refined trace = Z_C(u -> -u): same leading f(2mu) term (f is 2 pi i periodic).
(D) SU(3) twisted torus: h = g gbar has eigenvalues (1, e^{i gamma}, e^{-i gamma});
    chi_{2m} = (1+2cos m gamma)^2 - 1, chi_{2m+1} = 1 - Tr h^{2m+1} = -2 cos((2m+1) gamma).
"""
import sys, time
import mpmath as mp

FULL = "--full" in sys.argv   # full grids: ~60 s (run once locally on 2026-10-01, output deriv_cardy_full.out); default coarse grids < 10 s

mp.mp.dps = 20
t0 = time.time()
TWO_PI = 2 * mp.pi


def Li3(z):
    return mp.polylog(3, z)


def G(u, y):
    return -(Li3(y) - 3 * Li3(u * y) + 3 * Li3(u ** 2 * y) - Li3(u ** 3 * y))


def best_root_re(F):
    """max real part over the three cube roots of F."""
    r = mp.cbrt(abs(F)) if F != 0 else mp.mpf(0)
    th = mp.arg(F)
    return max(r * mp.cos((th + TWO_PI * k) / 3) for k in range(3))


def g(phi):
    u = mp.expj(phi)
    return best_root_re(G(u, 1))


# (A)
g_prot = g(TWO_PI / 3)
phi0 = mp.findroot(lambda t: mp.diff(g, t), 2.22)
r0 = g(phi0) / g_prot
print('(A) protected g(2pi/3) = %.10f ; check |f(2pi i/3)|^(1/3) cos(pi/6) = %.10f' %
      (g_prot, mp.cbrt(4 * mp.pi ** 3 / 27) * mp.cos(mp.pi / 6)))
print('    phi_0 = %.6f = %.6f pi ; ratio g(phi_0)/g(2pi/3) = %.6f  (KKLP 4.15: 2.2247, 1.01213)' % (phi0, phi0 / mp.pi, r0))
print('    17 pi/24 = %.6f pi (x = e^{3 i phi_0} = e^{i %.5f}, 2pi/16 = %.5f)' % (17 / 24, 3 * phi0 - TWO_PI, TWO_PI / 16))
phi1 = mp.findroot(lambda t: g(t) - g_prot, 2.5)
print('    window above protected value: phi in (2pi/3, %.6f) = (0.666667 pi, %.6f pi); x-angle 3phi-2pi in (0, %.5f)'
      % (phi1, phi1 / mp.pi, 3 * phi1 - TWO_PI))

# (B) roots of unity
print('(B) Z(x) at x = e^{2 pi i k/m}: preserved by d_n iff m | n. Growth ratio max_branch g / g_prot:')
for m in [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]:
    best = None
    for k in range(1, m):
        from math import gcd
        if gcd(k, m) != 1:
            continue
        for l in range(3):
            phi = (TWO_PI * k / m + TWO_PI * l) / 3
            val = g(phi) / g_prot
            if best is None or val > best[0]:
                best = (val, k, phi / mp.pi)
    print('    m = %2d: max ratio %.6f (k = %d, phi = %.4f pi)%s' % (m, best[0], best[1], best[2],
          '  > 1' if best[0] > 1 else ''))

# (C) C-twisted SU(N) at the trivial twisted holonomy, ratio to the untwisted S at the protected point and at max
print('(C) C-twisted SU(N), trivial twisted holonomy: S_C/S at protected point; S_C,max/S_cl,max over phi')
for N in ([3, 4, 5, 10, 100, 10 ** 4] if FULL else [3, 10 ** 4]):
    def FC(phi, N=N):
        u = mp.expj(phi)
        f1 = G(u, 1)
        f2 = G(u ** 2, 1)          # f(2 mu)
        return (N ** 2 - 1) * f2 / 8 + (1 - N) * (f1 - f2 / 8)

    def Fu(phi, N=N):
        return (N ** 2 - 1) * G(mp.expj(phi), 1)
    sp_ = best_root_re(FC(TWO_PI / 3)) / best_root_re(Fu(TWO_PI / 3))
    # classical max over phi of the twisted function, scanned
    grid = [mp.mpf(i) / (400 if FULL else 80) * mp.pi for i in range(1, 400 if FULL else 80)]
    mC = max(best_root_re(FC(t)) for t in grid)
    mU = best_root_re(Fu(phi0))
    print('    N = %6d: S_C/S (protected) = %.4f  [closed form ((N+10)/(8(N+1)))^(1/3) = %.4f];'
          ' max_phi S_C / S_cl,max = %.4f ; max_phi S_C / S = %.4f'
          % (N, sp_, mp.cbrt(mp.mpf(N + 10) / (8 * (N + 1))), mC / mU, mC / best_root_re(Fu(TWO_PI / 3))))
# P-refined = Z_C(u -> -u): leading large-N term f(2(mu + i pi)) = f(2 mu)
u = mp.expj(mp.mpf('0.7'))
print('    periodicity check f(2mu+2pi i) - f(2mu) at phi=0.7: %.2e' % abs(G(u ** 2, 1) - G((-u) ** 2, 1)))

# (D) SU(3) twisted torus scan
print('(D) SU(3) twisted torus h = (1, e^{i gamma}, e^{-i gamma}): max over gamma of Re F_C^{1/3} / Re F^{1/3}')


def FC3(phi, gam):
    u = mp.expj(phi)
    e = lambda s: mp.expj(s)
    Gy = lambda y: G(u, y)
    even2 = Gy(1) + Gy(-1)
    even4c = Gy(e(gam / 2)) + Gy(-e(gam / 2)) + Gy(e(-gam / 2)) + Gy(-e(-gam / 2))
    even2c = (Gy(e(gam)) + Gy(-e(gam)) + Gy(e(-gam)) + Gy(-e(-gam))) / 2
    odd = -(Gy(e(gam)) - Gy(-e(gam)) + Gy(e(-gam)) - Gy(-e(-gam))) / 2
    return even2 + even4c + even2c + odd


# consistency: gamma = 0 equals the closed form (5/4) f(2mu) - 2 f(mu)
uu = mp.expj(TWO_PI / 3)
print('    gamma=0 check: %.2e' % abs(FC3(TWO_PI / 3, 0) - (mp.mpf(5) / 4 * G(uu ** 2, 1) - 2 * G(uu, 1))))
Sprot = best_root_re(8 * G(uu, 1))
Smax = best_root_re(8 * G(mp.expj(phi0), 1))
gams = [mp.pi * i / (60 if FULL else 12) for i in range(61 if FULL else 13)]
prot = [(best_root_re(FC3(TWO_PI / 3, gm)), gm) for gm in gams]
bp = max(prot)
print('    protected point: gamma=0: %.4f ; max over gamma: %.4f at gamma = %.3f pi  (ratio to S)' %
      (prot[0][0] / Sprot, bp[0] / Sprot, bp[1] / mp.pi))
phis = [mp.pi * i / 60 for i in range(1, 120)] if FULL else [mp.pi * (0.9 + 0.01 * i) for i in range(0, 21)]
bc = max((best_root_re(FC3(t, gm)), t, gm) for t in (phis[::2] if FULL else phis[::2]) for gm in (gams[::3] if FULL else gams[::3]))
print('    classical (max over phi, gamma, coarse grid): %.4f S = %.4f S_cl,max at phi = %.3f pi, gamma = %.3f pi'
      % (bc[0] / Sprot, bc[0] / Smax, bc[1] / mp.pi, bc[2] / mp.pi))
print('time %.1fs' % (time.time() - t0))
