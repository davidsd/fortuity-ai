#!/usr/bin/env python3
r"""verify_freeenergy.py -- T-041: independent check of the large-N classical free energy at the collapsed saddle,
through the first subleading order of the Cardy expansion.  Written for T-041 by a Claude agent,
2026-10-02.  Imports nothing from t038/.

CLAIM UNDER TEST (T-038 note sec. 7.1; KKLP arXiv:2606.27955 (4.1)-(4.4) for the leading term)
  log Z / dim G = L(w, mu) := sum_{n>=1} (1/n) i_s(p^n, q^n, u_I^n)      (identity holonomy)
                = f(m)/(w1 w2) + (w1 + w2) H(m)/(w1 w2) + O(log w),       mu_I = m_I + (w1 + w2)/3,
  with p, q = e^{-w1}, e^{-w2}, u_I = e^{-mu_I}, z_A = exp(-sum_{I in A} m_I) over the 8 subsets A of {1,2,3},
  s_A = (-1)^{|A|}:
     f = -sum_A s_A Li3(z_A)             (= -G_3; KKLP's (4.4) at u1 = u2 = u3)
     h = -(1/2) sum_A s_A Li2(z_A)       (the O(1/w) coefficient at fixed mu)
     H = h + (1/3) sum_I d f/d m_I = sum_A s_A (|A|/3 - 1/2) Li2(z_A)   (the O(1/w) coefficient at fixed m).
  On the protected surface sum m = -2 pi i (Re m_I = 0): f = m1 m2 m3/2 and H = (m1 m2 + m2 m3 + m3 m1)/6, the
  first two orders of the exact large-N cubic dim G mu1 mu2 mu3/(2 w1 w2).

PART 1  Single-letter index from the letter charges.  Letters and charges come from the repo's
  code/fortuity/full/letters.py (letters_up_to, is_ferm, charges2; via $REPO, default four directories up), with a
  local transcription of its docstring rules as a fallback.  With T = R1 + R2 + R3 - Y (Y = 1 per letter), the KKLP
  weight of a letter is (-1)^F p^{J1+T} q^{J2+T} prod_j u_j^{R_j-T}.  Checked: the exponents are the Taylor degrees
  (a, b, [j in S]); (-1)^F = (-1)^{2 J1} = (-1)^{2 J2}; the letter sum equals 1 - prod_j (1 - u_j)/((1-p)(1-q))
  exactly through p^4 q^4 (sympy) and numerically through degree 60 at random complex fugacities.
PART 2  Small-w fits at complex points.  L is evaluated exactly in double precision as
     L = sum_{A != {}} s_A log(1 - z_A(mu)) + sum_{n>=1} (1/n) A_n (1 - B_n),
     A_n = sum_A s_A z_A(mu)^n, B_n = 1/((1-p^n)(1-q^n))
  (valid for |u_I| <= 1; the second sum converges geometrically), and checked against the plain sum
  sum_n (1/n) i_s at one point.  At w = t (w1, w2), t from 0.02 down to 0.0015, the function
  Y(t) = t^2 w1 w2 L is fitted to f + (w1+w2) H t + t^2 (d log t + c) + e3 t^3 + e4 t^4 + e5 t^5, both at fixed m
  (shift included, giving f and H) and at fixed mu (giving f and h).  A Mellin-transform argument predicts
  the log coefficient: d = -(3/4 - (w1^2 + w2^2)/(12 w1 w2)) w1 w2 in general, and d = -w1 w2 on the protected
  surface, where z_{123} -> 1.  The remainder (Y - f - (w1+w2) H t)/t^2 must be a + b log t + O(t), with no 1/t or
  1/t^2 piece left (a wrong H would leave (w1+w2) dH/t, about 10^3 dH at t = 0.0015).
PART 3  Negative controls: a shift (w1+w2)/2 instead of /3, or h normalised as -G2 instead of -G2/2, must miss the
  fitted data by O(1); the fit with the shift (w1+w2)/2 must instead equal h + (1/2) sum_I d_I f.

USAGE   python verify_freeenergy.py          (about 3 s; numpy, mpmath, sympy; optional: the repo's fortuity package)
EXPECTED OUTPUT (local run 2026-10-02): Part 1 all PASS (letters a,b <= 4 symbolic; a,b <= 60 numeric, error
  ~1e-16).  Part 2 at 6 complex points: |f_fit - f| ~ 1e-12, |H_fit - H| and |h_fit - h| ~ 1e-9, |d_fit - d_pred|
  ~ 1e-6, the remainder is a + b log t + O(t) to ~1e-7, and on the protected surface H_fit = e2(m)/6 to
  ~1e-9.  Part 3: both controls miss by O(1) (1.5-2.6), and the shifted fit equals h + (1/2) sum_I d_I f to ~1e-9.
  Last line: "verify_freeenergy: ALL PASS" (about 2 s).
"""
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import mpmath as mp

SUB = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
SG = [(-1) ** len(A) for A in SUB]
ALL_OK = True


def report(name, err, tol):
    global ALL_OK
    ok = err <= tol
    ALL_OK = ALL_OK and ok
    print("  %-70s %9.2e  (tol %.0e) %s" % (name, err, tol, "PASS" if ok else "FAIL"))


# ------------------------------------------------------------------ Part 1: letters -> i_s
def load_letters():
    repo = Path(os.environ.get("REPO", Path(__file__).resolve().parents[4]))
    sys.path.insert(0, str(repo / "code"))
    try:
        from fortuity.full import letters as LT          # noqa: E402
        return LT.letters_up_to, LT.is_ferm, LT.charges2, "fortuity.full.letters (%s)" % LT.__file__
    except Exception as exc:  # noqa: BLE001
        def letters_up_to(amax, bmax):
            return [(a, b, S) for a in range(amax + 1) for b in range(bmax + 1)
                    for r in range(4) for S in combinations((0, 1, 2), r) if r >= 1 or a + b >= 1]

        def is_ferm(L):
            return len(L[2]) % 2 == 0

        def charges2(L):                                  # letters.py docstring: (2J1, 2J2, 2q1, 2q2, 2q3)
            a, b, S = L
            s = len(S)
            return (-1 + 2 * a + s, -1 + 2 * b + s) + tuple(a + b + 2 * (j in S) for j in (0, 1, 2))
        return letters_up_to, is_ferm, charges2, "local transcription of letters.py docstring (import failed: %s)" % exc


def kklp_exponents(L, charges2):
    """(J1+T, J2+T, R1-T, R2-T, R3-T) and Y from the Chang-Lin charges; T = R1+R2+R3-Y, Y = 1, R_j = q_j - J_L."""
    c = charges2(L)
    J1, J2 = mp.mpf(c[0]) / 2, mp.mpf(c[1]) / 2
    q = [mp.mpf(x) / 2 for x in c[2:]]
    JL = (J1 + J2) / 2
    R = [x - JL for x in q]
    Y = 1
    T = sum(R) - Y
    return (J1 + T, J2 + T, R[0] - T, R[1] - T, R[2] - T), (J1, J2)


def part1():
    print("PART 1: single-letter index from the letter charges")
    letters_up_to, is_ferm, charges2, src = load_letters()
    print("  letters from:", src)
    import sympy as sp
    p, q, u1, u2, u3 = sp.symbols('p q u1 u2 u3')
    M = 4
    Ls = letters_up_to(M, M)
    bad = 0
    poly = 0
    for L in Ls:
        ex, (J1, J2) = kklp_exponents(L, charges2)
        a, b, S = L
        target = (a, b) + tuple(1 if j in S else 0 for j in (0, 1, 2))
        if any(abs(e - t) > 0 for e, t in zip(ex, target)):
            bad += 1
        sgnF = -1 if is_ferm(L) else 1
        if sgnF != (-1) ** int(2 * J1) or sgnF != (-1) ** int(2 * J2):
            bad += 1
        e = [int(x) for x in ex]
        poly += sgnF * p ** e[0] * q ** e[1] * u1 ** e[2] * u2 ** e[3] * u3 ** e[4]
    report("%d letters (a,b<=%d): exponents = (a,b,[j in S]), spin-statistics" % (len(Ls), M), bad, 0)
    closed = 1 - (1 - u1) * (1 - u2) * (1 - u3) * sum(p ** k for k in range(M + 1)) * sum(q ** k for k in range(M + 1))
    diff = sp.expand(poly - closed)
    report("sum over letters = 1 - prod(1-u)/((1-p)(1-q)) through p^%d q^%d (sympy, exact)" % (M, M),
           0 if diff == 0 else 1, 0)
    # numeric, high degree
    rng = np.random.default_rng(7)
    Mn = 60
    Ls = letters_up_to(Mn, Mn)
    ex = np.array([[float(x) for x in kklp_exponents(L, charges2)[0]] for L in Ls])
    sg = np.array([-1.0 if is_ferm(L) else 1.0 for L in Ls])
    worst = 0.0
    for _ in range(3):
        pp = 0.4 * rng.random() * np.exp(2j * np.pi * rng.random())
        qq = 0.4 * rng.random() * np.exp(2j * np.pi * rng.random())
        uu = (0.5 + rng.random(3)) * np.exp(2j * np.pi * rng.random(3))
        val = np.sum(sg * pp ** ex[:, 0] * qq ** ex[:, 1] * uu[0] ** ex[:, 2] * uu[1] ** ex[:, 3] * uu[2] ** ex[:, 4])
        ref = 1 - np.prod(1 - uu) / ((1 - pp) * (1 - qq))
        worst = max(worst, abs(val - ref))
    report("%d letters (a,b<=%d) at 3 random complex points, |sum - closed form|" % (len(Ls), Mn), worst, 1e-13)


# ------------------------------------------------------------------ Part 2: small-omega expansion
def L_exact(w, mu):
    """sum_{n>=1} (1/n) i_s(p^n, q^n, u^n) for Re w > 0, |u_I| <= 1 (see docstring)."""
    w = np.asarray(w, complex)
    xA = [sum(mu[I] for I in A) for A in SUB]
    head = sum(SG[k] * np.log(1 - np.exp(-xA[k])) for k in range(1, 8))
    N = int(math.ceil(46 / min(w.real)))
    n = np.arange(1, N + 1)
    pn, qn = np.exp(-n * w[0]), np.exp(-n * w[1])
    An = sum(SG[k] * np.exp(-n * xA[k]) for k in range(8))
    terms = An * (-pn - qn + pn * qn) / ((1 - pn) * (1 - qn)) / n
    return head + complex(math.fsum(terms.real), math.fsum(terms.imag))


def L_plain(w, mu, N):
    """The plain plethystic sum, for |u_I| < 1 strictly (cross-check of L_exact)."""
    n = np.arange(1, N + 1)
    pn, qn = np.exp(-n * w[0]), np.exp(-n * w[1])
    un = [np.exp(-n * x) for x in mu]
    i_s = 1 - (1 - un[0]) * (1 - un[1]) * (1 - un[2]) / ((1 - pn) * (1 - qn))
    terms = i_s / n
    return complex(math.fsum(terms.real), math.fsum(terms.imag))


def analytic(m):
    """f, h, H and sum_I d_I f from the polylog formulas (mpmath, principal branches)."""
    mp.mp.dps = 25
    m = [mp.mpc(x) for x in m]
    f = h = ds = mp.mpc(0)
    for k, A in enumerate(SUB):
        z = mp.exp(-sum(m[I] for I in A)) if A else mp.mpf(1)
        f += -SG[k] * mp.polylog(3, z)
        h += -SG[k] * mp.polylog(2, z) / 2
        ds += SG[k] * len(A) * mp.polylog(2, z)
    return complex(f), complex(h), complex(h + ds / 3), complex(ds)


def fit(ts, Y):
    B = np.array([[1, t, t * t * math.log(t), t * t, t ** 3, t ** 4, t ** 5] for t in ts], complex)
    c, *_ = np.linalg.lstsq(B, Y, rcond=None)
    return c


def part2_3():
    print("PART 2: small-omega fits of L = sum_n (1/n) i_s at complex points")
    phi0 = 2.2247468715390893          # KKLP (4.15), recomputed in indep_delta_r1.py
    pts = [("KKLP classical saddle, m = -i phi0 (1,1,1)", [-1j * phi0] * 3, (1.0, 1.0)),
           ("same, complex w", [-1j * phi0] * 3, (0.7 + 0.4j, 1.3 - 0.2j)),
           ("protected, m = -2 pi i/3 (1,1,1)", [-2j * np.pi / 3] * 3, (1.0, 1.7)),
           ("protected unequal, m = -i (1.7, 2.2, 2pi-3.9)", [-1.7j, -2.2j, -(2 * np.pi - 3.9) * 1j], (0.8 + 0.35j, 1.1 - 0.25j)),
           ("classical-type unequal, m = -i (2.0, 2.3, 2.4)", [-2.0j, -2.3j, -2.4j], (1.0, 2.5)),
           ("generic, Re m > 0", [0.35 - 1.9j, 0.15 - 2.6j, 0.5 - 1.2j], (0.9 - 0.3j, 1.2 + 0.4j))]
    ts = np.geomspace(0.02, 0.0015, 14)
    # cross-check of the evaluation of L against the plain sum (needs |u| < 1 strictly)
    m, w = pts[-1][1], (0.05 * pts[-1][2][0], 0.05 * pts[-1][2][1])
    mu = [x + (w[0] + w[1]) / 3 for x in m]
    report("L_exact vs plain plethystic sum (generic point, t = 0.05)", abs(L_exact(w, mu) - L_plain(w, mu, 20000)), 1e-9)
    controls = []
    for name, m, (w1, w2) in pts:
        s = w1 + w2
        f, h, H, ds = analytic(m)
        prot = abs(sum(m) + 2j * np.pi) < 1e-12
        b0 = 0.25 + (w1 / w2 + w2 / w1) / 12
        dpred = -w1 * w2 if prot else -(1 - b0) * w1 * w2
        Ysh = np.array([L_exact((t * w1, t * w2), [x + t * s / 3 for x in m]) * t * t * w1 * w2 for t in ts])
        Yun = np.array([L_exact((t * w1, t * w2), m) * t * t * w1 * w2 for t in ts])
        cs, cu = fit(ts, Ysh), fit(ts, Yun)
        print(" point: %s, w = (%s, %s)" % (name, np.round(w1, 3), np.round(w2, 3)))
        report("f_fit - f(m)", abs(cs[0] - f), 1e-10)
        report("H_fit - [h + (1/3) sum_I d_I f](m)   (fixed m, shift (w1+w2)/3)", abs(cs[1] / s - H), 1e-7)
        report("h_fit - [-(1/2) Li2-combination](mu)   (fixed mu)", abs(cu[1] / s - h), 1e-7)
        report("log-coefficient d_fit - d_pred%s" % (" (protected: -w1 w2)" if prot else ""), abs(cs[2] - dpred), 1e-4)
        R = (Ysh - f - s * H * ts) / ts ** 2
        Bl = np.array([[1, math.log(t), t, t * t] for t in ts], complex)
        cl, *_ = np.linalg.lstsq(Bl, R, rcond=None)
        report("remainder (Y - f - sHt)/t^2 = a + b log t + O(t): max deviation", np.max(np.abs(Bl @ cl - R)), 1e-4)
        if prot:
            e2 = m[0] * m[1] + m[1] * m[2] + m[2] * m[0]
            report("protected surface: f(m) - m1 m2 m3/2", abs(f - m[0] * m[1] * m[2] / 2), 1e-12)
            report("protected surface: H_fit - (m1m2+m2m3+m3m1)/6", abs(cs[1] / s - e2 / 6), 1e-7)
        controls.append((name, m, (w1, w2), H, h, ds))
    print("PART 3: negative controls (the wrong forms must miss the fitted data by O(1))")
    global ALL_OK
    for name, m, (w1, w2), H, h, ds in controls[:3]:
        s = w1 + w2
        Ywr = np.array([L_exact((t * w1, t * w2), [x + t * s / 2 for x in m]) * t * t * w1 * w2 for t in ts])
        Yun = np.array([L_exact((t * w1, t * w2), m) * t * t * w1 * w2 for t in ts])
        Hwr, hfit = fit(ts, Ywr)[1] / s, fit(ts, Yun)[1] / s
        miss1, miss2 = abs(Hwr - H), abs(hfit - 2 * h)
        ok = miss1 > 1e-2 and miss2 > 1e-2
        ALL_OK = ALL_OK and ok
        print("  %s" % name)
        print("    shift (w1+w2)/2: |H_fit - H| = %.4f (must be O(1)); |H_fit - (h + sum_I d_I f/2)| = %.1e (shift mechanism)"
              % (miss1, abs(Hwr - (h + ds / 2))))
        print("    h normalised as -G2 instead of -G2/2: |h_fit - (-G2)| = %.4f (must be O(1))  -> %s"
              % (miss2, "controls fail as they should" if ok else "CONTROL DID NOT FAIL"))
        report("shift-mechanism check: H_fit(shift/2) - (h + (1/2) sum_I d_I f)", abs(Hwr - (h + ds / 2)), 1e-7)

if __name__ == "__main__":
    t0 = time.time()
    part1()
    part2_3()
    print("verify_freeenergy: %s  (%.1f s)" % ("ALL PASS" if ALL_OK else "SOME CHECKS FAILED", time.time() - t0))
