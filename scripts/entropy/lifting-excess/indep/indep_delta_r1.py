#!/usr/bin/env python3
r"""indep_delta_r1.py -- T-041, independent implementation of the charge dependence of the KKLP lifting excess.

Written from scratch for T-041 (written by a Claude agent, 2026-10-02).  t038/rf/scan_kklp_charges.py
was neither imported nor copied.  Only its docstring, its CLI and its scan() grid were read, so that the comparison
points below can sit on the production grid.  KKLP = Kim, Kim, Lee, Park, "Quantum black hole cohomologies",
arXiv:2606.27955, sec. 4, eqs. (4.1)-(4.16).  Conventions are those of the T-038 analysis (summary/fortuity-summary.pdf)

WHAT IT COMPUTES (large N, collapsed = identity-holonomy saddle, Cardy regime j1, j2 >> dim G)
  input   xi = (xi1, xi2, xi3), sum xi = 0, xi_I = (R_I - Rbar)/K, K = 3 (dim G j1 j2)^{1/3}
  delta0  = S_cl,max / S_prot - 1 at leading Cardy order        (KKLP (4.15) at xi = 0: 1.0121344 - 1)
  Phi*    bonus phase at the classical maximum: sum_I mu_I = -i Phi*  (Phi* = 3 phi_0 = 2.1244768 pi at xi = 0)
  Q_I/sum Q   black-hole charge fractions: Q_I/K = cbar + xi_I, cbar = Im(S_prot/K)/(2 pi) (the Im S = 0 condition)
  r1      delta(eps) = delta0 + r1 eps + O(eps^2), eps = (dim G)^{1/3} (j1 + j2) / (2 (j1 j2)^{2/3}), along the
          family of fixed xi and fixed j1/j2 (j_i ~ lambda^3, R_I - Rbar ~ lambda^2, eps ~ 1/lambda).  This is the
          definition behind the note's envelope formula.  Each member of the family is a fixed set of physical
          charges (j1, j2, R_I - Rbar), at which both indices are evaluated.  Along it the black-hole fractions drift
          at O(eps) (column dcbar/deps).  At fixed fractions instead, r1 would change by grad delta0 . d xi/d eps,
          which vanishes at xi = 0.
  All of these are independent of the gauge group at the collapsed saddle, since dim G only sets the scale (T-038
  note sec. 7.1 statement 3).

FREE ENERGY USED (derived and checked numerically in verify_freeenergy.py of this folder)
  log Z / dim G = [ f(m) + (w1 + w2) H(m) ] / (w1 w2) + O(log w),   mu_I = m_I + (w1 + w2)/3,   u_I = e^{-mu_I},
  f = -sum_A s_A Li3(z_A),  H = sum_A s_A (|A|/3 - 1/2) Li2(z_A),  z_A = exp(-sum_{I in A} m_I),  s_A = (-1)^{|A|},
  A over the 8 subsets of {1,2,3} (z_empty = 1).  This H equals h + (1/3) sum_I d_I f with h = -(1/2) sum_A s_A Li2.
  All Li_s are principal branches; every saddle reports its distance from the cuts (columns cut_cl, cut_pr).
  Charges: the KKLP weight p^{J1+T} q^{J2+T} prod u_I^{R_I-T} equals p^{j1} q^{j2} prod_I e^{-m_I (rho_I + b)} with
  j_i = J_i + Rbar, rho_I = R_I - Rbar = K xi_I, b = Rbar - T = q_B/3 (KKLP's bonus charge q_B = 3(R-T)), so that
  m_I is conjugate to R_I - T and nu = sum_I m_I = sum mu - w1 - w2 is the bonus potential (x = e^{-nu}).

METHOD (chosen to differ from the pilot, which solves at fixed Phi and then runs a secant in Phi)
 (A) delta0, classical side: ONE real Newton system in all variables at once.  Unknowns (Re m_I, Im m_I, beta),
     beta = b/K real; equations d_I g(m) + xi_I + beta = 0 (I = 1, 2, 3; complex) and Re sum m = 0 (KKLP (4.11):
     the maximum over real bonus charge forces Re nu = 0).  g = f^{1/3}, principal root = root of maximal real part
     (KKLP below (4.14)).  S_cl,max/K = Re[g + sum m xi].  The second-order condition d^2 Re S/d beta^2 =
     Re(d nu/d beta) < 0 (a maximum over the bonus) is computed from the Hessian and printed.
 (B) delta0, second route without cube roots: the same Legendre transform with w1, w2 kept as unknowns,
       S/K = (1/3)[f + s (w1+w2) H]/(w1 w2) + (kap w1 + w2/kap)/3 + sum_I m_I (xi_I + beta),        (*)
     at s = 0, as a real Newton system in (w1, w2, m_I, beta) (11 real unknowns).  The cube-root branch is replaced
     by the choice of the w-saddle.  Column dB-dA is the difference of the two delta0's.
 (P) Protected side (x = 1, sum m = -2 pi i): Newton on the surface in (m1, m2), with polylogs; checked against
     (i) the closed form f = m1 m2 m3/2 and H = (m1m2+m2m3+m3m1)/6 on the surface (column cubic), and
     (ii) the black-hole formula Re S_prot/K = 2 pi sqrt(q1q2+q2q3+q3q1) with the charge constraint
     (q1+q2+q3)(q1q2+q2q3+q3q1) - q1q2q3 = 1/54, q_I = cbar + xi_I (column BH0).
 (r1) Finite-eps Legendre transform of the truncated free energy (*): in (*) the physical charges are fixed through
     xi, kap = sqrt(j1/j2) and s = (dim G/sqrt(j1 j2))^{1/3}, so eps = s (kap + 1/kap)/2.  Both sides are solved at
     eps in {-2h, -h, 0, h, 2h, 3h, 4h} (h = 0.002 by default) by continuation in s from s = 0, giving
     delta(eps) = Re S_cl,max(eps)/Re S_prot(eps) - 1.  r1 is the Richardson-extrapolated symmetric difference
     (column r1); a one-sided 5-point difference over eps >= 0 is column r1_os.  The envelope value
     r1_env = d/d eps of Re S_cl/Re S_prot from (2/3) Re[H/g] at the leading saddles (T-038 note sec. 7.1, sec. 9.3)
     is printed only for comparison.  Two further checks of the protected side at finite eps: the same solve with
     the exact large-N cubic dim G mu1 mu2 mu3/(2 w1 w2) in place of the truncated (f, H) (column r1cub-r1), and,
     for that exact cubic, the finite-charge black-hole formula Re S/K = 2 pi sqrt(e2(q) - eps/9 + cbar s^2/3) with
     the constraint p0 = p1 p2 (column BHfin).  The O(eps^2) term of the truncated model, c2_trunc, is printed for
     scale only.  It is not physical, since the O(w^0) part of log Z is not included.
 Continuation: every point is reached from xi = 0 (KKLP's saddle) in steps of at most 0.03 in |xi|, with a secant
 predictor, so the saddle is the one continuously connected to KKLP's.  Column ncross counts crossings of a
 polylog cut along the path; it must be 0.

USAGE
  python indep_delta_r1.py XI1 XI2 XI3            one point (fractions such as -1/8 are accepted)
  python indep_delta_r1.py --frac F1 F2 F3        one point given by black-hole fractions Q_I/sum Q
  python indep_delta_r1.py --polar THETA_DEG R    one point on the grid of scan_kklp_charges.py scan:
                                                  xi = R (cos th e1 + sin th e2), e1 = (2,-1,-1)/sqrt6, e2 = (0,1,-1)/sqrt2
  python indep_delta_r1.py --file points.txt      one point per line: "XI1 XI2 XI3", "frac F1 F2 F3" or "polar TH R";
                                                  '#' starts a comment
  python indep_delta_r1.py selftest               29 built-in checks, about 11 s on one core (M1 Max)
  python indep_delta_r1.py --grid 12 [--rmax0 1.0] [--chunk k/K]
                                                  the whole grid of scan_kklp_charges.py scan 12 (13 rays,
                                                  R = 0.02 j while min Q > 0), with the TH = 0 ray, which has no
                                                  edge, cut at R <= rmax0; --chunk k/K keeps points i = k mod K
  options: --dps N (default 25), --h H (default 0.002), --kappa K (sqrt(j1/j2), default 1), --no-r1
  Output: one tab-separated row per point.  The first 13 columns have the same names and order as the output of
  scan_kklp_charges.py scan, for direct comparison (compare_scan.py in this folder).

EXPECTED OUTPUT (from local runs, 2026-10-02; dps 25; 2-4 s per point, up to 9 s for R >= 1 on the TH = 0 ray)
  xi = (0, 0, 0)               delta0 = 0.01213443354  Phi*/pi = 2.124476770  Q/sumQ = 1/3 each   r1 = 0.05507117330
  xi = (0.1, -0.1, 0)          delta0 = 0.01048536398  Phi*/pi = 2.115618057  Q/sumQ = (0.5739, 0.0928, 0.3333)
                               r1 = 0.0526715891   [pilot selftest: 0.010485, 0.052672]
  xi = (0.25, -0.125, -0.125)  delta0 = 0.005364944962 Phi*/pi = 2.072800945  Q/sumQ = (0.8281, 0.0859, 0.0859)
                               r1 = 0.0423899161   [pilot selftest: 0.005365, 0.042390]
  polar 0 1.5 (corner side)    delta0 = 1.78894e-5     Phi*/pi = 2.001212938  Q/sumQ = (0.99704, 0.00148, 0.00148)
                               r1 = 0.00310213
  In the bulk of the wedge r1 - r1_env is about 1e-12, dB-dA at most 1e-25, and BH0, cubic and BHfin are at the
  working precision; ncross = 0 everywhere.  On the TH = 0 ray the saddle approaches the branch point
  u1 u2 u3 = 1 (Phi* -> 2 pi) and the Li3(u1) cut (cut_cl -> 0) only asymptotically, as R -> infinity.  There
  |r1 - r1_env|/r1 grows to about 1e-5 at R = 1.5, because the eps steps are no longer small on that scale.
"""
import sys
import math
import time
from fractions import Fraction

import mpmath as mp

SUBSETS = ((), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
E3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))          # coordinate directions in m-space
V2 = ((1, 0, -1), (0, 1, -1))                   # tangent directions of the protected surface sum m = -2 pi i
SQ6, SQ2 = None, None                           # set in set_precision


def set_precision(dps):
    global SQ6, SQ2, TPI, TOL
    mp.mp.dps = dps
    SQ6, SQ2 = mp.sqrt(6), mp.sqrt(2)
    TPI = 2j * mp.pi
    TOL = mp.mpf(10) ** (-(dps * 4) // 5)


def Li(s, z):
    """Principal-branch polylogarithm, with the elementary closed forms for s = 1, 0."""
    if s >= 2:
        return mp.polylog(s, z)
    if s == 1:
        return -mp.log(1 - z)
    if s == 0:
        return z / (1 - z)
    raise ValueError(s)


def fH(m, dirs, surface=False):
    """f(m), H(m) and their first and second directional derivatives along the integer vectors in dirs.

    f = -sum_A s_A Li3(z_A),  H = sum_A s_A (|A|/3 - 1/2) Li2(z_A),  z_A = exp(-sum_{I in A} m_I).
    d_v Li_s(z_A) = -c_A(v) Li_{s-1}(z_A) with c_A(v) = sum_{I in A} v_I.
    surface=True: sum m = -2 pi i exactly, so z_{123} = 1 is constant; its derivative weights must vanish."""
    nd = len(dirs)
    z0 = mp.mpc(0)
    f, H = z0, z0
    df, dH = [z0] * nd, [z0] * nd
    d2f = [[z0] * nd for _ in range(nd)]
    d2H = [[z0] * nd for _ in range(nd)]
    for A in SUBSETS:
        sg = -1 if len(A) % 2 else 1
        hc = mp.mpf(len(A)) / 3 - mp.mpf(1) / 2
        c = [sum(v[I] for I in A) for v in dirs]
        if len(A) == 0 or (surface and len(A) == 3):
            if any(c):
                raise ValueError("derivative along a direction that leaves the surface")
            f += -sg * mp.zeta(3)
            H += sg * hc * mp.zeta(2)
            continue
        z = mp.exp(-sum(m[I] for I in A))
        L3, L2 = Li(3, z), Li(2, z)
        f += -sg * L3
        H += sg * hc * L2
        if any(c):
            L1, L0 = Li(1, z), Li(0, z)
            for a in range(nd):
                df[a] += sg * c[a] * L2
                dH[a] += -sg * hc * c[a] * L1
                for b in range(nd):
                    d2f[a][b] += -sg * c[a] * c[b] * L1
                    d2H[a][b] += sg * hc * c[a] * c[b] * L0
    return f, df, d2f, H, dH, d2H


def cbrt_max_re(z):
    """Principal cube root = the root of maximal real part for arg z in (-pi, pi) (KKLP below (4.14))."""
    return mp.exp(mp.log(z) / 3)


def g_derivs(f, df, d2f):
    g = cbrt_max_re(f)
    n = len(df)
    dg = [g * df[a] / (3 * f) for a in range(n)]
    d2g = [[g / (3 * f) * (d2f[a][b] - mp.mpf(2) / 3 * df[a] * df[b] / f) for b in range(n)] for a in range(n)]
    return g, dg, d2g


def newton(fun, x, maxit=60):
    """Damped Newton; fun(x) -> (residual column, Jacobian).  Works for real or complex mp.matrix."""
    r, J = fun(x)
    nr = mp.mnorm(r, 'inf')
    for _ in range(maxit):
        if nr < TOL:
            return x
        dx = mp.lu_solve(J, -r)
        lam = mp.mpf(1)
        while True:
            xn = x + lam * dx
            rn, Jn = fun(xn)
            nrn = mp.mnorm(rn, 'inf')
            if nrn < (1 - lam / 4) * nr or lam < mp.mpf('1e-6'):
                break
            lam /= 2
        x, r, J, nr = xn, rn, Jn, nrn
    raise RuntimeError("Newton did not converge (|res| = %s)" % mp.nstr(nr, 3))


def cplx_block(J, i, j, c):
    """Real 2x2 block of a holomorphic derivative c at rows (2i, 2i+1), columns (2j, 2j+1)."""
    J[2 * i, 2 * j], J[2 * i, 2 * j + 1] = mp.re(c), -mp.im(c)
    J[2 * i + 1, 2 * j], J[2 * i + 1, 2 * j + 1] = mp.im(c), mp.re(c)


# ---------------------------------------------------------------- (A) leading order, cube-root form
def clA_fun(xi):
    def fun(X):
        m = [mp.mpc(X[0], X[1]), mp.mpc(X[2], X[3]), mp.mpc(X[4], X[5])]
        f, df, d2f = fH(m, E3)[:3]
        g, dg, d2g = g_derivs(f, df, d2f)
        r, J = mp.matrix(7, 1), mp.matrix(7, 7)
        for I in range(3):
            E = dg[I] + xi[I] + X[6]
            r[2 * I], r[2 * I + 1] = mp.re(E), mp.im(E)
            for K in range(3):
                cplx_block(J, I, K, d2g[I][K])
            J[2 * I, 6] = 1
        r[6] = mp.re(m[0] + m[1] + m[2])
        for K in range(3):
            J[6, 2 * K] = 1
        return r, J
    return fun


def clA_info(X, xi):
    m = [mp.mpc(X[0], X[1]), mp.mpc(X[2], X[3]), mp.mpc(X[4], X[5])]
    f, df, d2f, H = fH(m, E3)[:4]
    g, dg, d2g = g_derivs(f, df, d2f)
    Hm = mp.matrix(3, 3)
    for a in range(3):
        for b in range(3):
            Hm[a, b] = d2g[a][b]
    dnu = -sum(mp.lu_solve(Hm, mp.matrix([1, 1, 1])))      # d nu/d beta along the saddle family (fixed xi)
    return dict(m=m, f=f, g=g, H=H, S=g + sum(m[I] * xi[I] for I in range(3)), beta=X[6], dnu=dnu)


def prA_fun(xi):
    def fun(Z):
        m = [Z[0], Z[1], -TPI - Z[0] - Z[1]]
        f, df, d2f = fH(m, V2, surface=True)[:3]
        g, dg, d2g = g_derivs(f, df, d2f)
        r, J = mp.matrix(2, 1), mp.matrix(2, 2)
        for a in range(2):
            r[a] = dg[a] + xi[a] - xi[2]
            for b in range(2):
                J[a, b] = d2g[a][b]
        return r, J
    return fun


def prA_info(Z, xi):
    m = [Z[0], Z[1], -TPI - Z[0] - Z[1]]
    f, H = fH(m, [], surface=True)[0], fH(m, [], surface=True)[3]
    g = cbrt_max_re(f)
    return dict(m=m, f=f, g=g, H=H, S=g + sum(m[I] * xi[I] for I in range(3)))


# ---------------------------------------------------------------- (B) explicit-omega truncated model (*)
def F_parts(w1, w2, m, dirs, s, kap, surface):
    """Value (without the linear charge terms), gradient and Hessian of
       (1/3)[f + s (w1+w2) H]/(w1 w2) + (kap w1 + w2/kap)/3   in the variables (w1, w2, m along dirs)."""
    f, df, d2f, H, dH, d2H = fH(m, dirs, surface=surface)
    n = len(dirs)
    P, iw = w1 * w2, 1 / w1 + 1 / w2
    val = (f / P + s * H * iw) / 3 + (kap * w1 + w2 / kap) / 3
    gr = [(-f / (w1 ** 2 * w2) - s * H / w1 ** 2) / 3 + kap / 3,
          (-f / (w1 * w2 ** 2) - s * H / w2 ** 2) / 3 + 1 / (3 * kap)]
    gr += [(df[a] / P + s * dH[a] * iw) / 3 for a in range(n)]
    Hs = [[None] * (2 + n) for _ in range(2 + n)]
    Hs[0][0] = (2 * f / (w1 ** 3 * w2) + 2 * s * H / w1 ** 3) / 3
    Hs[1][1] = (2 * f / (w1 * w2 ** 3) + 2 * s * H / w2 ** 3) / 3
    Hs[0][1] = Hs[1][0] = f / (w1 ** 2 * w2 ** 2) / 3
    for a in range(n):
        Hs[0][2 + a] = Hs[2 + a][0] = (-df[a] / (w1 ** 2 * w2) - s * dH[a] / w1 ** 2) / 3
        Hs[1][2 + a] = Hs[2 + a][1] = (-df[a] / (w1 * w2 ** 2) - s * dH[a] / w2 ** 2) / 3
        for b in range(n):
            Hs[2 + a][2 + b] = (d2f[a][b] / P + s * d2H[a][b] * iw) / 3
    return val, gr, Hs, f, H


def clB_fun(xi, s, kap):
    """Classical side of (*): X = Re/Im of (w1, w2, m1, m2, m3) and beta; 11 real equations."""
    def fun(X):
        z = [mp.mpc(X[2 * k], X[2 * k + 1]) for k in range(5)]
        _, gr, Hs, _, _ = F_parts(z[0], z[1], z[2:], E3, s, kap, False)
        E = [gr[0], gr[1]] + [gr[2 + I] + xi[I] + X[10] for I in range(3)]
        r, J = mp.matrix(11, 1), mp.matrix(11, 11)
        for k in range(5):
            r[2 * k], r[2 * k + 1] = mp.re(E[k]), mp.im(E[k])
            for l in range(5):
                cplx_block(J, k, l, Hs[k][l])
            if k >= 2:
                J[2 * k, 10] = 1
        r[10] = mp.re(z[2] + z[3] + z[4])
        for l in (2, 3, 4):
            J[10, 2 * l] = 1
        return r, J
    return fun


def clB_value(X, xi, s, kap):
    z = [mp.mpc(X[2 * k], X[2 * k + 1]) for k in range(5)]
    val = F_parts(z[0], z[1], z[2:], [], s, kap, False)[0]
    return val + sum(z[2 + I] * (xi[I] + X[10]) for I in range(3)), z


def prB_fun(xi, s, kap):
    """Protected side of (*): Z = (w1, w2, m1, m2), m3 = -2 pi i - m1 - m2; holomorphic 4x4 Newton."""
    def fun(Z):
        m = [Z[2], Z[3], -TPI - Z[2] - Z[3]]
        _, gr, Hs, _, _ = F_parts(Z[0], Z[1], m, V2, s, kap, True)
        E = [gr[0], gr[1], gr[2] + xi[0] - xi[2], gr[3] + xi[1] - xi[2]]
        r, J = mp.matrix(4, 1), mp.matrix(4, 4)
        for k in range(4):
            r[k] = E[k]
            for l in range(4):
                J[k, l] = Hs[k][l]
        return r, J
    return fun


def prB_value(Z, xi, s, kap):
    m = [Z[2], Z[3], -TPI - Z[2] - Z[3]]
    return F_parts(Z[0], Z[1], m, [], s, kap, True)[0] + sum(m[I] * xi[I] for I in range(3)), m


# ---------------------------------------------------------------- exact large-N cubic on the protected surface
def cubic_parts(Z, xi, s, kap):
    """F = mu1 mu2 mu3/(6 w1 w2) + (kap w1 + w2/kap)/3 + sum m xi, mu_I = m_I + s (w1+w2)/3, m3 = -2 pi i - m1 - m2.
    Returns value, gradient, Hessian in Z = (w1, w2, m1, m2)."""
    w1, w2 = Z[0], Z[1]
    m = [Z[2], Z[3], -TPI - Z[2] - Z[3]]
    T = s * (w1 + w2) / 3
    mu = [mi + T for mi in m]
    A = [[s / 3, s / 3, 1, 0], [s / 3, s / 3, 0, 1], [s / 3, s / 3, -1, -1]]
    P3 = mu[0] * mu[1] * mu[2]
    p = [mu[1] * mu[2], mu[0] * mu[2], mu[0] * mu[1]]
    M = [[0, mu[2], mu[1]], [mu[2], 0, mu[0]], [mu[1], mu[0], 0]]
    Qa = [sum(p[I] * A[I][a] for I in range(3)) for a in range(4)]
    Qab = [[sum(A[I][a] * A[J][b] * M[I][J] for I in range(3) for J in range(3)) for b in range(4)] for a in range(4)]
    R = 1 / (6 * w1 * w2)
    Ra = [-1 / (6 * w1 ** 2 * w2), -1 / (6 * w1 * w2 ** 2), 0, 0]
    Rab = [[1 / (3 * w1 ** 3 * w2), 1 / (6 * w1 ** 2 * w2 ** 2), 0, 0],
           [1 / (6 * w1 ** 2 * w2 ** 2), 1 / (3 * w1 * w2 ** 3), 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    lin = [kap / 3, 1 / (3 * kap), xi[0] - xi[2], xi[1] - xi[2]]
    val = P3 * R + (kap * w1 + w2 / kap) / 3 + sum(m[I] * xi[I] for I in range(3))
    gr = [Qa[a] * R + P3 * Ra[a] + lin[a] for a in range(4)]
    Hs = [[Qab[a][b] * R + Qa[a] * Ra[b] + Qa[b] * Ra[a] + P3 * Rab[a][b] for b in range(4)] for a in range(4)]
    return val, gr, Hs


def prC_fun(xi, s, kap):
    def fun(Z):
        _, gr, Hs = cubic_parts(Z, xi, s, kap)
        r, J = mp.matrix(4, 1), mp.matrix(4, 4)
        for k in range(4):
            r[k] = gr[k]
            for l in range(4):
                J[k, l] = Hs[k][l]
        return r, J
    return fun


def bh_finite_check(Sval, xi, s, kap):
    """Exact-cubic protected side: Re S/K = 2 pi sqrt(p1) and p0 = p1 p2 with cbar = Im(S/K)/(2 pi),
    p1 = e2(q) - eps/9 + cbar s^2/3, p2 = e1(q) + s^2/6, p0 = e3(q) + 1/54 - cbar eps/9 + cbar^2 s^2/6."""
    eps = s * (kap + 1 / kap) / 2
    cb = mp.im(Sval) / (2 * mp.pi)
    q = [x + cb for x in xi]
    e1, e2, e3 = sum(q), q[0] * q[1] + q[1] * q[2] + q[2] * q[0], q[0] * q[1] * q[2]
    p1 = e2 - eps / 9 + cb * s ** 2 / 3
    p2 = e1 + s ** 2 / 6
    p0 = e3 + mp.mpf(1) / 54 - cb * eps / 9 + cb ** 2 * s ** 2 / 6
    return max(abs(mp.re(Sval) - 2 * mp.pi * mp.sqrt(p1)), abs(p0 - p1 * p2))


# ---------------------------------------------------------------- branch-cut diagnostics
def cut_distance(m, skip_full):
    """min over subsets A with |z_A| > 1 of the angular distance of z_A = exp(-sum_A m) from the cut (1, inf)."""
    best = mp.inf
    for A in SUBSETS[1:]:
        if skip_full and len(A) == 3:
            continue
        x = sum(m[I] for I in A)
        if mp.re(x) < -mp.mpf('1e-12'):
            d = abs(mp.im(x) - 2 * mp.pi * mp.nint(mp.im(x) / (2 * mp.pi)))
            best = min(best, d)
    return best


def cut_crossings(m_old, m_new, skip_full):
    n = 0
    for A in SUBSETS[1:]:
        if skip_full and len(A) == 3:
            continue
        xo, xn = sum(m_old[I] for I in A), sum(m_new[I] for I in A)
        if mp.re(xo) < 0 and mp.re(xn) < 0:
            if mp.floor(mp.im(xo) / (2 * mp.pi)) != mp.floor(mp.im(xn) / (2 * mp.pi)):
                n += 1
    return n


# ---------------------------------------------------------------- KKLP point, independently
def kklp_phi0():
    """phi_0 = argmax_phi Re f(-i phi (1,1,1))^{1/3} (KKLP (4.14)-(4.15)), by golden section then a root of the
    derivative; independent of the Newton systems above."""
    G = lambda p: mp.re(cbrt_max_re(fH([-1j * p] * 3, [])[0]))
    a, b = mp.mpf(2), mp.mpf('2.5')
    gr = (mp.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = b - gr * (b - a), a + gr * (b - a)
        if G(c) > G(d):
            b = d
        else:
            a = c
    phi0 = mp.findroot(lambda p: mp.diff(G, p), (a + b) / 2)
    Sp = mp.re(cbrt_max_re(fH([-TPI / 3] * 3, [], surface=True)[0]))
    return phi0, G(phi0) / Sp


# ---------------------------------------------------------------- charges
def to_mp(sv):
    fr = Fraction(sv)
    return mp.mpf(fr.numerator) / fr.denominator


def frac_to_xi(F):
    """Black-hole fractions -> xi.  Q_I/K = t F_I with t^3 (e2(F) - e3(F)) = 1/54 (sum F = 1); xi_I = t (F_I - 1/3)."""
    F = [mp.mpf(x) for x in F]
    sF = sum(F)
    F = [x / sF for x in F]
    e2, e3 = F[0] * F[1] + F[1] * F[2] + F[2] * F[0], F[0] * F[1] * F[2]
    t = (54 * (e2 - e3)) ** (-mp.mpf(1) / 3)
    return [t * (x - mp.mpf(1) / 3) for x in F]


def polar_to_xi(th_deg, r):
    th = mp.pi * mp.mpf(th_deg) / 180
    e1 = [2 / SQ6, -1 / SQ6, -1 / SQ6]
    e2 = [0, 1 / SQ2, -1 / SQ2]
    return [mp.mpf(r) * (mp.cos(th) * a + mp.sin(th) * b) for a, b in zip(e1, e2)]


def ray_edge(th_deg):
    """R at which the ray TH of the scan grid meets the edge Q3 = 0 (closed form; infinite for TH = 0).
    On the edge the fractions are (phi, 1 - phi, 0) with tan TH = sqrt3 (1 - phi)/(3 phi - 1), so
    phi = (sqrt3 + tan TH)/(sqrt3 + 3 tan TH) and xi = t (phi - 1/3, 2/3 - phi, -1/3), t = (54 phi (1-phi))^{-1/3}."""
    tn = mp.tan(mp.pi * mp.mpf(th_deg) / 180)
    if tn == 0:
        return mp.inf
    phi = (mp.sqrt(3) + tn) / (mp.sqrt(3) + 3 * tn)
    t = (54 * phi * (1 - phi)) ** (-mp.mpf(1) / 3)
    return t * mp.sqrt((phi - mp.mpf(1) / 3) ** 2 + (mp.mpf(2) / 3 - phi) ** 2 + mp.mpf(1) / 9)


def grid_points(n, rmax0):
    """The grid of scan_kklp_charges.py scan n: TH = 60 k/n deg, R = 0.02 j while min Q > 0; the TH = 0 ray, which
    never meets an edge, is cut at R <= rmax0.  The R = 0 point is listed once."""
    pts = [("polar 0 0", [mp.mpf(0)] * 3)]
    for k in range(n + 1):
        th = mp.mpf(60) * k / n
        redge = ray_edge(th)
        j = 1
        while True:
            r = mp.mpf(j) / 50
            if (redge == mp.inf and r > mp.mpf(rmax0) + mp.mpf('1e-12')) or (redge != mp.inf and r >= redge):
                break
            pts.append(("polar %s %s" % (mp.nstr(th, 8), mp.nstr(r, 4)), polar_to_xi(th, r)))
            j += 1
    return pts


def xi_to_polar(xi):
    e1 = [2 / SQ6, -1 / SQ6, -1 / SQ6]
    e2 = [0, 1 / SQ2, -1 / SQ2]
    x, y = sum(a * b for a, b in zip(xi, e1)), sum(a * b for a, b in zip(xi, e2))
    return mp.degrees(mp.atan2(y, x)), mp.sqrt(x * x + y * y)


# ---------------------------------------------------------------- one point
def leading(xi, phi0):
    """Continuation in xi from KKLP's saddle; returns (X_cl, Z_pr, number of cut crossings)."""
    m0 = [-1j * phi0] * 3
    f, df, d2f = fH(m0, E3)[:3]
    _, dg, _ = g_derivs(f, df, d2f)
    X = mp.matrix([0, -phi0, 0, -phi0, 0, -phi0, -mp.re(dg[0])])
    Z = mp.matrix([-TPI / 3, -TPI / 3])
    X = newton(clA_fun([0, 0, 0]), X)
    Z = newton(prA_fun([0, 0, 0]), Z)
    r = mp.sqrt(sum(x * x for x in xi))
    n = max(2, int(mp.ceil(r / mp.mpf('0.03'))))
    hist = [(X, Z)]
    ncross = 0
    k, tau, dtau = 0, mp.mpf(0), mp.mpf(1) / n
    while tau < 1:
        step = min(dtau, 1 - tau)
        tn = tau + step
        xs = [tn * c for c in xi]
        if len(hist) >= 2 and step == dtau:            # secant predictor (equal steps)
            Xg = 2 * hist[-1][0] - hist[-2][0]
            Zg = 2 * hist[-1][1] - hist[-2][1]
        else:
            Xg, Zg = hist[-1]
        try:
            Xn = newton(clA_fun(xs), Xg)
            Zn = newton(prA_fun(xs), Zg)
        except RuntimeError:
            if dtau < mp.mpf('1e-4'):
                raise
            dtau /= 2
            hist = hist[-1:]
            continue
        mo = [mp.mpc(hist[-1][0][2 * i], hist[-1][0][2 * i + 1]) for i in range(3)]
        mn = [mp.mpc(Xn[2 * i], Xn[2 * i + 1]) for i in range(3)]
        po = [hist[-1][1][0], hist[-1][1][1], -TPI - hist[-1][1][0] - hist[-1][1][1]]
        pn = [Zn[0], Zn[1], -TPI - Zn[0] - Zn[1]]
        ncross += cut_crossings(mo, mn, True) + cut_crossings(po, pn, True)
        hist.append((Xn, Zn))
        hist = hist[-2:]
        tau = tn
        k += 1
    return hist[-1][0], hist[-1][1], ncross


def solve_point(xi, phi0, kap=1, h=mp.mpf('0.002'), want_r1=True):
    t0 = time.time()
    xi = [mp.mpf(x) for x in xi]
    Xc, Zp, ncross = leading(xi, phi0)
    cl, pr = clA_info(Xc, xi), prA_info(Zp, xi)
    Scl, Spr = mp.re(cl['S']), mp.re(pr['S'])
    d0A = Scl / Spr - 1
    Phi = -mp.im(sum(cl['m']))
    cbar = mp.im(pr['S']) / (2 * mp.pi)
    q = [cbar + x for x in xi]
    fr = [x / sum(q) for x in q]
    e2q = q[0] * q[1] + q[1] * q[2] + q[2] * q[0]
    chk_bh0 = max(abs(Spr - 2 * mp.pi * mp.sqrt(e2q)), abs(sum(q) * e2q - q[0] * q[1] * q[2] - mp.mpf(1) / 54))
    mp_ = pr['m']
    chk_cub = max(abs(pr['f'] - mp_[0] * mp_[1] * mp_[2] / 2),
                  abs(pr['H'] - (mp_[0] * mp_[1] + mp_[1] * mp_[2] + mp_[2] * mp_[0]) / 6))
    # envelope value, for comparison only (T-038 note sec. 9.3)
    a, b = Scl, mp.re(2 * cl['H'] / (3 * cl['g']))
    c, d = Spr, mp.re(2 * pr['H'] / (3 * pr['g']))
    r1_env = (b * c - a * d) / c ** 2
    out = dict(xi=xi, fr=fr, Spr=Spr, Scl=Scl, Phi=Phi, d0=d0A, r1_env=r1_env, beta=cl['beta'],
               d2=mp.re(cl['dnu']), chk_bh0=chk_bh0, chk_cub=chk_cub, ncross=ncross,
               cut_cl=cut_distance(cl['m'], True), cut_pr=cut_distance(pr['m'], True),
               dPhi=abs(Phi - 2 * mp.pi * mp.nint(Phi / (2 * mp.pi))), argf=mp.arg(cl['f']) / mp.pi)
    # (B) explicit-omega formulation at s = 0
    X = mp.matrix(11, 1)
    wv = [cl['g'] / kap, cl['g'] * kap] + cl['m']
    for k in range(5):
        X[2 * k], X[2 * k + 1] = mp.re(wv[k]), mp.im(wv[k])
    X[10] = cl['beta']
    Z = mp.matrix([pr['g'] / kap, pr['g'] * kap, pr['m'][0], pr['m'][1]])

    def solve_s(s, X, Z):
        Xs = newton(clB_fun(xi, s, kap), X)
        Zs = newton(prB_fun(xi, s, kap), Z)
        Sc, _ = clB_value(Xs, xi, s, kap)
        Sp, _ = prB_value(Zs, xi, s, kap)
        return Xs, Zs, Sc, Sp

    X0, Z0, Sc0, Sp0 = solve_s(0, X, Z)
    out['dBA'] = (mp.re(Sc0) / mp.re(Sp0) - 1) - d0A
    if want_r1:
        e2s = 2 / (kap + 1 / kap)                       # s = e2s * eps
        dl, cb, dlc, bhfin = {0: mp.re(Sc0) / mp.re(Sp0) - 1}, {0: mp.im(Sp0) / (2 * mp.pi)}, {}, mp.mpf(0)
        ZC0 = newton(prC_fun(xi, 0, kap), Z0)
        Sc_c0 = cubic_parts(ZC0, xi, 0, kap)[0]
        dlc[0] = mp.re(Sc0) / mp.re(Sc_c0) - 1
        bhfin = max(bhfin, bh_finite_check(Sc_c0, xi, 0, kap))
        for sgn in (1, -1):
            Xp, Zp_, Zc = X0, Z0, ZC0
            for k in ((1, 2, 3, 4) if sgn > 0 else (1, 2)):
                s = sgn * k * h * e2s
                Xp, Zp_, Sc, Sp = solve_s(s, Xp, Zp_)
                Zc = newton(prC_fun(xi, s, kap), Zc)
                Spc = cubic_parts(Zc, xi, s, kap)[0]
                dl[sgn * k] = mp.re(Sc) / mp.re(Sp) - 1
                cb[sgn * k] = mp.im(Sp) / (2 * mp.pi)
                dlc[sgn * k] = mp.re(Sc) / mp.re(Spc) - 1
                bhfin = max(bhfin, bh_finite_check(Spc, xi, s, kap))
        D1, D2 = (dl[1] - dl[-1]) / (2 * h), (dl[2] - dl[-2]) / (4 * h)
        r1 = (4 * D1 - D2) / 3
        r1_os = (-25 * dl[0] + 48 * dl[1] - 36 * dl[2] + 16 * dl[3] - 3 * dl[4]) / (12 * h)
        C1, C2 = (dlc[1] - dlc[-1]) / (2 * h), (dlc[2] - dlc[-2]) / (4 * h)
        out.update(r1=r1, r1_os=r1_os, r1cub=(4 * C1 - C2) / 3,
                   c2=(16 * (dl[1] + dl[-1] - 2 * dl[0]) - (dl[2] + dl[-2] - 2 * dl[0])) / (12 * h * h) / 2,
                   dcbar=((8 * (cb[1] - cb[-1]) - (cb[2] - cb[-2])) / (12 * h)), bhfin=bhfin)
    out['time'] = time.time() - t0
    return out


COLS = ("theta_deg", "r", "xi1", "xi2", "xi3", "Q1/sQ", "Q2/sQ", "Q3/sQ", "Sprot/K", "Scl/K", "Phi/pi", "delta0",
        "r1", "r1_env", "r1-r1_env", "r1_os-r1_env", "r1cub-r1", "dB-dA", "c2_trunc", "beta*", "d2ReS/dbeta2",
        "dcbar/deps", "BH0", "cubic", "BHfin", "cut_cl", "cut_pr", "|Phi-2pi k|", "argf/pi", "ncross", "time_s")


def row(o):
    th, r = xi_to_polar(o['xi'])
    g = lambda x, n=10: mp.nstr(x, n, min_fixed=-5, max_fixed=6)
    e = lambda x: "%.1e" % float(x)
    vals = [g(th, 8), g(r, 8)] + [g(x, 10) for x in o['xi']] + [g(x, 8) for x in o['fr']]
    vals += [g(o['Spr'], 12), g(o['Scl'], 12), g(o['Phi'] / mp.pi, 10), g(o['d0'], 11)]
    if 'r1' in o:
        vals += [g(o['r1'], 11), g(o['r1_env'], 11), e(o['r1'] - o['r1_env']), e(o['r1_os'] - o['r1_env']),
                 e(o['r1cub'] - o['r1']), e(o['dBA']), g(o['c2'], 6)]
    else:
        vals += ["-", g(o['r1_env'], 11), "-", "-", "-", e(o['dBA']), "-"]
    vals += [g(o['beta'], 10), g(o['d2'], 6), g(o['dcbar'], 6) if 'dcbar' in o else "-", e(o['chk_bh0']),
             e(o['chk_cub']), e(o['bhfin']) if 'bhfin' in o else "-", g(o['cut_cl'], 5), g(o['cut_pr'], 5),
             g(o['dPhi'], 5), g(o['argf'], 5), str(o['ncross']), "%.1f" % o['time']]
    return "\t".join(vals)


def parse_line(tok):
    if tok[0] == 'frac':
        return frac_to_xi([to_mp(x) for x in tok[1:4]])
    if tok[0] == 'polar':
        return polar_to_xi(to_mp(tok[1]), to_mp(tok[2]))
    xi = [to_mp(x) for x in tok[:3]]
    s = sum(xi)
    if abs(s) > mp.mpf('1e-12'):
        print("# warning: sum xi = %s; projected onto sum xi = 0" % mp.nstr(s, 5), flush=True)
        xi = [x - s / 3 for x in xi]
    return xi


def selftest(opts):
    t0 = time.time()
    ok = True

    def check(name, val, ref, tol):
        nonlocal ok
        good = abs(val - ref) <= tol
        ok = ok and good
        print("%-62s %-22s ref %-14s %s" % (name, mp.nstr(val, 14), mp.nstr(ref, 10), "PASS" if good else "FAIL"),
              flush=True)

    phi0, ratio = kklp_phi0()
    check("KKLP (4.15): phi_0 (1D golden section, independent)", phi0, mp.mpf('2.2247'), mp.mpf('1e-4'))
    check("KKLP (4.15): ratio at phi_0", ratio, mp.mpf('1.01213'), mp.mpf('1e-5'))
    o = solve_point([0, 0, 0], phi0, h=opts['h'])
    check("xi=0: delta0 (Newton A) = 1D ratio - 1", o['d0'], ratio - 1, mp.mpf('1e-18'))
    check("xi=0: Phi*/pi = 3 phi_0/pi", o['Phi'] / mp.pi, 3 * phi0 / mp.pi, mp.mpf('1e-15'))
    check("xi=0: r1 (finite-eps slope) vs note sec. 7.1 (0.0551)", o['r1'], mp.mpf('0.0551'), mp.mpf('1e-4'))
    check("xi=0: r1 - r1_env", o['r1'] - o['r1_env'], 0, mp.mpf('1e-9'))
    check("xi=0: q_B*/(dim G j1 j2)^{1/3} = 9 beta* (note sec. 5.2: 3.1624)", 9 * o['beta'], mp.mpf('3.1624'),
          mp.mpf('1e-4'))
    o2 = solve_point([0, 0, 0], phi0, kap=2, h=opts['h'])
    check("xi=0, j1/j2 = 4: r1 equals the j1 = j2 value", o2['r1'], o['r1'], mp.mpf('1e-9'))
    for xi, d0ref, r1ref in (([mp.mpf('0.1'), mp.mpf('-0.1'), 0], '0.010485', '0.052672'),
                             ([mp.mpf('0.25'), mp.mpf('-0.125'), mp.mpf('-0.125')], '0.005365', '0.042390')):
        o = solve_point(xi, phi0, h=opts['h'])
        lab = "xi=(%s): " % ",".join(mp.nstr(x, 3) for x in xi)
        check(lab + "delta0 vs pilot selftest", o['d0'], mp.mpf(d0ref), mp.mpf('6e-7'))
        check(lab + "r1 vs pilot selftest", o['r1'], mp.mpf(r1ref), mp.mpf('6e-7'))
        check(lab + "r1 - r1_env", o['r1'] - o['r1_env'], 0, mp.mpf('1e-9'))
        check(lab + "explicit-omega delta0 - cube-root delta0", o['dBA'], 0, mp.mpf('1e-15'))
        check(lab + "protected: BH formula + constraint", o['chk_bh0'], 0, mp.mpf('1e-15'))
        check(lab + "protected: f, H = cubic forms", o['chk_cub'], 0, mp.mpf('1e-15'))
        check(lab + "protected, finite eps: exact-cubic BH formula", o['bhfin'], 0, mp.mpf('1e-15'))
        check(lab + "r1 with exact-cubic protected side - r1", o['r1cub'] - o['r1'], 0, mp.mpf('1e-9'))
        check(lab + "maximum over the bonus: d2 Re S/d beta2 < 0", min(o['d2'], 0), o['d2'], 0)
        check(lab + "polylog cut crossings along the path", o['ncross'], 0, 0)
    xi = [mp.mpf('-0.1'), 0, mp.mpf('0.1')]
    o = solve_point(xi, phi0, want_r1=False)
    check("permuted xi=(-0.1,0,0.1): delta0 equals the (0.1,-0.1,0) value", o['d0'], mp.mpf('0.0104853639801'),
          mp.mpf('1e-12'))
    print("selftest %s, %.1f s" % ("PASSED" if ok else "FAILED", time.time() - t0))
    return ok


def main(argv):
    opts = dict(dps=25, h=mp.mpf('0.002'), kappa=1, r1=True, rmax0='1.0', chunk=(0, 1))
    args, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a == '--dps':
            opts['dps'] = int(argv[i + 1]); i += 2
        elif a == '--h':
            opts['h'] = argv[i + 1]; i += 2
        elif a == '--kappa':
            opts['kappa'] = argv[i + 1]; i += 2
        elif a == '--no-r1':
            opts['r1'] = False; i += 1
        elif a == '--rmax0':
            opts['rmax0'] = argv[i + 1]; i += 2
        elif a == '--chunk':
            opts['chunk'] = tuple(int(x) for x in argv[i + 1].split('/')); i += 2
        else:
            args.append(a); i += 1
    set_precision(opts['dps'])
    opts['h'] = mp.mpf(opts['h'])
    kap = to_mp(opts['kappa'])
    if args and args[0] == 'selftest':
        return 0 if selftest(opts) else 1
    points = []
    if args and args[0] == '--file':
        with open(args[1]) as fh:
            for line in fh:
                tok = line.split('#')[0].split()
                if tok:
                    points.append((" ".join(tok), parse_line(tok)))
    elif args and args[0] == '--grid':
        points = grid_points(int(args[1]), to_mp(opts['rmax0']))
    elif args and args[0] in ('--frac', '--polar'):
        points.append((" ".join(args), parse_line([args[0][2:]] + args[1:])))
    elif len(args) == 3:
        points.append((" ".join(args), parse_line(args)))
    else:
        print(__doc__)
        return 2
    k, K = opts['chunk']
    points = [pt for i, pt in enumerate(points) if i % K == k]
    t0 = time.time()
    phi0, ratio = kklp_phi0()
    print("# indep_delta_r1.py  dps=%d  h=%s  kappa=sqrt(j1/j2)=%s  KKLP phi0/pi=%s ratio=%s" %
          (opts['dps'], mp.nstr(opts['h'], 4), mp.nstr(kap, 6), mp.nstr(phi0 / mp.pi, 15), mp.nstr(ratio, 15)))
    print("\t".join(COLS), flush=True)
    for label, xi in points:
        try:
            print(row(solve_point(xi, phi0, kap=kap, h=opts['h'], want_r1=opts['r1'])), flush=True)
        except Exception as exc:  # noqa: BLE001 -- report and continue with the next point
            print("# fail [%s]: %s" % (label, exc), flush=True)
    print("# total %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
