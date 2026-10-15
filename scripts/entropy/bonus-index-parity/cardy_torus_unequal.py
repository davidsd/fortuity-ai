#!/usr/bin/env python3
"""Identity-holonomy dominance at UNEQUAL R-charges (open point of the T-038 analysis (summary/fortuity-summary.pdf)).

Charges: xi_I = (R_I - Rbar)/K, sum xi = 0, K = 3 (dimG j1 j2)^(1/3), leading Cardy order (as note/check_unequal.py).
Step 1 (mpmath): the identity (collapsed) saddle m_I, u_I = e^{-m_I}, of S/K = g(m) + sum m xi, g = (-G3)^(1/3),
  for the protected index (sum m = -2 pi i) and for KKLP's classical maximum over the bonus (sum m = -i Phi*,
  bonus charge real), by continuation in s from xi = 0 (check_unequal.py's equations).
Step 2 (numpy): on a grid of the maximal torus, the Cardy coefficient per dim G at holonomy theta,
  f_theta(m) = -[rank G3z(0, m) + sum_roots G3z(alpha.theta, m)]/dimG,
  G3z(a, m) = Li3(z) - sum_I Li3(z u_I) + sum_{I<J} Li3(z u_I u_J) - Li3(z u1 u2 u3),  z = e^{i a},
  with Li3(e^w) summed from its series in w (|w| < 2 pi; checked against mpmath in `selftest`).
Two dominance ratios, both = 1 at the identity (and at central elements); the identity dominates iff max <= 1:
  R1 (fixed potentials): Re[f_theta/r^2]/Re[r], r = f_identity^(1/3) (principal root = maximal real part). This is
     Re of the exponent dimG f/(w1 w2) at the identity's own saddle, where w1 w2 = (dimG f_id)^(2/3) (j1 j2)^(-1/3).
  R2 (KKLP level, as cardy_torus_groups.py): Legendre in omega at each holonomy with the R-charge potentials held at
     the identity's values: Re[f_theta(m')^(1/3)] + Re[sum m xi]; protected: m' = m; classical: m' = m + i delta/3,
     maximized over the bonus shift delta (full period), and divided by the identity's own maximum over delta.
     At xi = 0 R2 reproduces cardy_torus_groups.py / cardy_su2_torus.py (checked in `selftest`).
This is a necessary-condition (self-consistency) check of the collapsed saddle at the level of rigour of the note;
complex holonomies off the real torus are not examined.

Usage:  python cardy_torus_unequal.py selftest                         (~6 s locally)
        python cardy_torus_unequal.py GROUP POINT [dth] [ddel]          GROUP: su2 su3 su4 so7 usp6 g2
        POINT: eq, r1a, r1b, r1c, r1d, r2a, r2b, r3a, r3b (the nine points of the note's sec. 7.1), all, or x1,x2,x3
        dth, ddel: torus and bonus-shift steps as fractions of pi (defaults 1/16 and 1/60).
Runtime (one core of an M1 Max; identity saddle 2-14 s per point in mpmath, the rest is the grid and the refinement):
  measured: selftest 2.8 s; su3 r1a at (1/8, 1/30) 6.6 s; so7 r3a at (1/4, 1/30) 14.1 s (cardy_torus_unequal_smoke.out);
  identity saddles alone at r1d (4 continuation steps) 13.7 s, max |Re m_I| = 0.86, so |w| < 3.3 in li3e (radius 2 pi).
  Estimated at the defaults (1/16, 1/60): su2 ~5 s, su3 and g2 ~15 s, so7 and usp6 ~40 s per point; 'all' (9 points)
  ~6 min for so7 or usp6. Total for {su2, su3, g2, so7, usp6} x all: ~15-20 min on one core, under 1 GB
  (5 parallel jobs: ~6 min). Halving dth multiplies the grid cost by 2^rank.
"""
import sys
import time
import itertools
import numpy as np
import mpmath as mp
from cardy_lib import roots

# ------------------------------------------------------------------ vectorized Li3(e^w)
_M = 48
_ZC = np.array([complex(mp.zeta(3 - 2 * k) / mp.factorial(2 * k)) for k in range(2, _M)])   # coefficients of w^{2k}
Z3, Z2 = float(mp.zeta(3)), float(mp.zeta(2))


def li3e(w):
    """Principal Li3(e^w) for complex arrays w: Im w is reduced to [-pi, pi]; needs |w| < 2 pi after that."""
    w = np.asarray(w, dtype=complex)
    w = w.real + 1j * (w.imag - 2 * np.pi * np.round(w.imag / (2 * np.pi)))
    if w.size and np.max(np.abs(w)) > 5.8:
        raise ValueError("li3e: |w| = %.3f too close to 2 pi" % np.max(np.abs(w)))
    w2 = w * w
    nz = w != 0
    lg = np.zeros_like(w)
    lg[nz] = np.log(-w[nz])
    ser = np.zeros_like(w)
    for c in _ZC[::-1]:
        ser = ser * w2 + c
    return Z3 + Z2 * w + 0.5 * w2 * (1.5 - lg) - w * w2 / 12 + ser * w2 * w2


def G3z(a, m):
    ia = 1j * np.asarray(a, float)
    m1, m2, m3 = m
    return (li3e(ia) - li3e(ia - m1) - li3e(ia - m2) - li3e(ia - m3)
            + li3e(ia - m1 - m2) + li3e(ia - m1 - m3) + li3e(ia - m2 - m3) - li3e(ia - m1 - m2 - m3))


def croot(z):
    z = np.asarray(z, complex)
    return np.abs(z) ** (1 / 3) * np.exp(1j * np.angle(z) / 3)


# ------------------------------------------------------------------ groups (as cardy_torus_groups.py)
def setup(grp):
    if grp in ('su2', 'su3', 'su4'):
        N = int(grp[2])
        R = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    v = np.zeros(N - 1)
                    if i < N - 1: v[i] += 1
                    if j < N - 1: v[j] -= 1
                    R.append(v)                       # theta_N = 0 gauge
        return np.array(R), N - 1, N - 1, True, 2 * np.pi
    if grp == 'g2':
        R = np.array([[r_[0], r_[1]] for r_ in roots('g2')[0]])   # theta = (x, y, 0)
        return R, 2, 2, True, 2 * np.pi
    R, rank, nd = roots(grp)                          # so7, usp6: cube [0, pi]^3 (Weyl reflections)
    return R, rank, nd, False, np.pi


def f_theta(TH, m, R, rank):
    Ru, Rm = np.unique(R, axis=0, return_counts=True)
    F = rank * G3z(np.zeros(len(TH)), m)
    for row, k in zip(Ru, Rm):
        F = F + k * G3z(TH @ row, m)
    return -F / (rank + len(R))


# ------------------------------------------------------------------ identity saddle (mpmath; check_unequal.py)
mp.mp.dps = 20
L2 = lambda z: mp.polylog(2, z)
L3 = lambda z: mp.polylog(3, z)


def G3(m):
    u = [mp.exp(-x) for x in m]
    return L3(1) - sum(L3(x) for x in u) + L3(u[0] * u[1]) + L3(u[0] * u[2]) + L3(u[1] * u[2]) - L3(u[0] * u[1] * u[2])


def dG3(m):
    u = [mp.exp(-x) for x in m]; P = u[0] * u[1] * u[2]
    return [L2(u[I]) - sum(L2(u[I] * u[J]) for J in range(3) if J != I) + L2(P) for I in range(3)]


def stationary(nu, xi, start):
    def eqs(m1, m2):
        m = [m1, m2, nu - m1 - m2]; g = (-G3(m)) ** (mp.mpf(1) / 3); d = dG3(m)
        dg = [-d[I] / (3 * g ** 2) for I in range(3)]
        return [dg[0] - dg[2] + xi[0] - xi[2], dg[1] - dg[2] + xi[1] - xi[2]]
    m1, m2 = mp.findroot(eqs, start, tol=1e-26, maxsteps=40, verify=False)
    m = [m1, m2, nu - m1 - m2]; g = (-G3(m)) ** (mp.mpf(1) / 3); d = dG3(m)
    return g + sum(m[I] * xi[I] for I in range(3)), -d[2] / (3 * g ** 2) + xi[2], (m1, m2), m


def identity_saddles(xi, nsteps=None):
    xi = [mp.mpf(x) for x in xi]
    Phi = 3 * mp.mpf('2.22474687153909')
    cl = (-1j * Phi / 3, -1j * Phi / 3); pr = (-2j * mp.pi / 3, -2j * mp.pi / 3)
    nsteps = nsteps or max(1, int(mp.ceil(max(abs(x) for x in xi) / mp.mpf('0.07'))))   # continuation steps
    steps = [mp.mpf(k) / nsteps for k in range(1, nsteps + 1)]
    for s in steps:
        x = [s * v for v in xi]
        Sp, _, pr, mpr = stationary(-2j * mp.pi, x, pr)
        box = {}

        def imk(P):
            S, k, st, mm = stationary(-1j * P, x, box.get('st', cl))
            box['st'] = st
            return mp.im(k), S, mm
        P0, P1 = Phi - mp.mpf('0.01'), Phi + mp.mpf('0.01')
        f0, _, _ = imk(P0); f1, S1, mcl = imk(P1)
        for _ in range(14):
            if abs(f1) < 1e-13: break
            P2 = P1 - f1 * (P1 - P0) / (f1 - f0); P0, f0 = P1, f1; P1 = P2; f1, S1, mcl = imk(P1)
        Phi, cl = P1, box['st']
    tocx = lambda mm: tuple(complex(v) for v in mm)
    smx = lambda mm: complex(sum(mm[I] * xi[I] for I in range(3)))
    return {'protected': (tocx(mpr), smx(mpr), float(mp.re(Sp))),
            'classical': (tocx(mcl), smx(mcl), float(mp.re(S1)), float(Phi / mp.pi))}


# ------------------------------------------------------------------ grid maxima and refinement
def local_maxima(A, nd, periodic):
    Ap = A
    for ax in range(A.ndim):
        mode = 'wrap' if (ax < nd and periodic) or ax >= nd else 'reflect'   # delta axis is periodic too
        pw = [(0, 0)] * A.ndim; pw[ax] = (1, 1)
        Ap = np.pad(Ap, pw, mode=mode)
    ok = np.ones(A.shape, bool)
    for sh in itertools.product((-1, 0, 1), repeat=A.ndim):
        if any(sh):
            sl = tuple(slice(1 + s_, Ap.shape[i] - 1 + s_) for i, s_ in enumerate(sh))
            ok &= A >= Ap[sl]
    return np.argwhere(ok)


def zoom(fun, x0, w0, m=5, tol=1e-9, maxit=200):
    x = np.array(x0, float); w = float(w0); d = len(x)
    offs = np.array(list(itertools.product(np.linspace(-1, 1, m), repeat=d)))
    edge = np.any(np.abs(offs) == 1, axis=1)
    best = None
    for _ in range(maxit):
        X = x + w * offs
        v = fun(X)
        k = int(np.argmax(v)); x = X[k]; best = v[k]
        if not edge[k]:
            w /= 3
        if w < tol:
            break
    return x, best


def report(label, V, grids, nd, periodic, fun, step, top=5):
    lm = local_maxima(V, nd, periodic)
    vals = np.array([V[tuple(i)] for i in lm])
    out = []
    for k in np.argsort(vals)[::-1]:
        x0 = np.array([g[i] for g, i in zip(grids, lm[k])])
        x, v = zoom(fun, x0, 2 * step)
        if not any(abs(v - w) < 1e-7 for _, w in out):
            out.append((x, v))
        if len(out) >= top:
            break
    mx = max(v for _, v in out)
    print("  %s: max %.7f -> %s" % (label, mx, "identity dominates" if mx <= 1 + 1e-7 else "IDENTITY NOT DOMINANT"))
    for x, v in out:
        loc = "theta/pi = %s" % np.round(x[:nd] / np.pi, 4)
        if len(x) > nd:
            loc += ", delta/(3 pi) = %.4f" % (x[nd] / (3 * np.pi))
        print("      %.7f at %s" % (v, loc))
    return mx


PRESETS = {'eq': (0, 0, 0), 'r1a': (0.0625, -0.03125, -0.03125), 'r1b': (0.125, -0.0625, -0.0625),
           'r1c': (0.1875, -0.09375, -0.09375), 'r1d': (0.25, -0.125, -0.125),
           'r2a': (0.0826, -0.0413, -0.0413), 'r2b': (0.1652, -0.0826, -0.0826),
           'r3a': (0.0310, 0.0310, -0.0620), 'r3b': (0.0620, 0.0620, -0.1240)}


def run(grp, xi, dth=1 / 16, ddel=1 / 60, sad=None):
    t0 = time.time()
    R, rank, nd, periodic, L = setup(grp)
    sad = sad or identity_saddles(xi)
    ts = time.time() - t0
    n1 = int(round((L / np.pi) / dth))
    th1 = np.arange(n1 if periodic else n1 + 1) * dth * np.pi
    TH = np.array(list(itertools.product(th1, repeat=nd)))
    shape = (len(th1),) * nd
    print("%s xi = %s: Phi*/pi = %.4f, Re S_prot/K = %.10f, Re S_cl/K = %.10f (delta0 = %.6f); grid %d pts; saddle %.1f s"
          % (grp, tuple(round(float(x), 5) for x in xi), sad['classical'][3], sad['protected'][2], sad['classical'][2],
             sad['classical'][2] / sad['protected'][2] - 1, len(TH), ts))
    res = {}
    for side in ('protected', 'classical'):
        m, smx = sad[side][0], sad[side][1]
        r = complex(croot(f_theta(np.zeros((1, nd)), m, R, rank))[0])
        R1fun = lambda X: (f_theta(np.atleast_2d(X), m, R, rank) / r ** 2).real / r.real
        res[side, 'R1'] = report(side + " R1 (fixed potentials)", R1fun(TH).reshape(shape), [th1] * nd, nd, periodic,
                                 R1fun, dth * np.pi)
        if side == 'protected':
            R2fun = lambda X: (croot(f_theta(np.atleast_2d(X), m, R, rank)).real + smx.real) / (r.real + smx.real)
            res[side, 'R2'] = report(side + " R2 (KKLP level)", R2fun(TH).reshape(shape), [th1] * nd, nd, periodic,
                                     R2fun, dth * np.pi)
        else:
            dl = np.arange(int(round(2 / ddel))) * ddel * np.pi * 3 - 3 * np.pi          # delta/3 in [-pi, pi)

            def R2raw(X):
                X = np.atleast_2d(X)
                vals = np.empty(len(X))
                ud = np.unique(X[:, nd])
                for dv in ud:
                    sel = X[:, nd] == dv
                    mm = tuple(mi + 1j * dv / 3 for mi in m)
                    vals[sel] = croot(f_theta(X[sel, :nd], mm, R, rank)).real + smx.real
                return vals
            idv = np.array([[0.0] * nd + [d] for d in dl])
            idvals = R2raw(idv)
            k0 = int(np.argmax(idvals))
            # Denominator: the identity holonomy's own maximum over the bonus coordinate only. All holonomy angles stay
            # exactly 0 here (zooming over every coordinate could drift to a competing holonomy and make R2 <= 1 by
            # construction; fixed after the round-1 review, finding T038-04).
            idfun = lambda D: R2raw(np.hstack([np.zeros((len(np.atleast_2d(D)), nd)), np.atleast_2d(D)]))
            xd, norm = zoom(idfun, idv[k0][nd:], 2 * ddel * np.pi * 3)
            xid = np.concatenate([np.zeros(nd), xd])
            print("  classical: identity's own max over delta at delta/(3 pi) = %.5f (0 expected), value/(r + sum m xi) = %.9f"
                  % (xid[nd] / (3 * np.pi), norm / (r.real + smx.real)))
            V = np.empty((len(TH), len(dl)))
            for j, dv in enumerate(dl):
                mm = tuple(mi + 1j * dv / 3 for mi in m)
                V[:, j] = croot(f_theta(TH, mm, R, rank)).real + smx.real
            R2fun = lambda X: R2raw(X) / norm
            res[side, 'R2'] = report(side + " R2 (KKLP level, max over bonus)", (V / norm).reshape(shape + (len(dl),)),
                                     [th1] * nd + [dl], nd, periodic, R2fun, dth * np.pi)
    print("  time %.1f s" % (time.time() - t0))
    return res


def selftest():
    t0 = time.time()
    rng = np.random.default_rng(1)
    w = rng.uniform(-1.5, 1.5, 40) + 1j * rng.uniform(-3.3, 3.3, 40)
    err = max(abs(complex(li3e(np.array([x]))[0]) - complex(mp.polylog(3, mp.exp(x)))) for x in w)
    print("li3e vs mpmath.polylog(3, e^w) on 40 random w (|Re w| <= 1.5): max error %.1e" % err)
    S0 = float((4 * mp.pi ** 3 / 27) ** (mp.mpf(1) / 3) * mp.cos(mp.pi / 6))
    sad = identity_saddles((0, 0, 0))
    print("xi = 0: Re S_cl/K / S0 = %.7f (KKLP 1.0121344), Phi*/pi = %.5f (2.12448)" % (sad['classical'][2] / S0, sad['classical'][3]))
    print("Expected at xi = 0 (cardy_su2_torus.out): theta = pi gives R2 protected 0.793701, classical 0.979613/1.0121344 = 0.967868")
    run('su2', (0, 0, 0), 1 / 16, 1 / 60, sad)
    print("selftest %.1f s" % (time.time() - t0))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] == 'selftest':
        selftest()
    else:
        grp, pt = sys.argv[1], sys.argv[2]
        dth = eval(sys.argv[3]) if len(sys.argv) > 3 else 1 / 16
        ddel = eval(sys.argv[4]) if len(sys.argv) > 4 else 1 / 60
        pts = list(PRESETS.items()) if pt == 'all' else [(pt, PRESETS[pt] if pt in PRESETS else tuple(float(v) for v in pt.split(',')))]
        for name, xi in pts:
            print("== point %s" % name)
            run(grp, xi, dth, ddel)
