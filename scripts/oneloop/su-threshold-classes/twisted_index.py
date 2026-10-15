"""Charge-conjugation-twisted index of SU(3) N=4 SYM (full 1/16-BPS letters), per letter content and per sector.

C acts on every adjoint letter by sigma(X) = -X^T (an outer automorphism of su(3)).  On the gauge invariants of a
content (tensor product of Sym^n / Lambda^n of the adjoint over the letters), Tr_Inv(C) = int_G Tr(Ad_g o sigma) dg.
By the twisted Weyl integration formula this is a 1-dimensional integral over the torus fixed by the pinned
representative sigma_0 = Ad_J o sigma:  S = {diag(e^{i th}, 1, e^{-i th})},
    int_G F(Ad_g sigma) dg = int_S F(s sigma_0) jac(s) ds / int_S jac(s) ds,   jac = det(1 - Ad(s sigma_0))|_{g/s}.
(An inner factor does not change the integral over G.)  Characters: h_n / e_n of the eigenvalues of Ad(s sigma_0),
from its power sums by Newton's identities, as in fullsector/dims_full.content_dim for the untwisted dimension.

Index at a weight: sum over contents of (-1)^F Tr_Inv(C) (twisted) or (-1)^F dim Inv (ordinary); SU(3)_R hw
multiplicities by Weyl alternation.  Validation: brute-force C-parity counts from words (c_parity.C_multi)."""
import sys, math, random
from collections import Counter, defaultdict
import numpy as np
from fastcontents import contents_at
from fortuity.full.dims import content_dim, weyl_orbit_signed
from fortuity.full.letters import is_ferm

M = 512
TH = 2 * np.pi * np.arange(M) / M


def sl3_basis():
    B = []
    for i in range(3):
        for j in range(3):
            if i != j:
                E = np.zeros((3, 3), dtype=complex); E[i, j] = 1; B.append(E)
    B.append(np.diag([1, -1, 0]).astype(complex)); B.append(np.diag([0, 1, -1]).astype(complex))
    return B


BASIS = sl3_basis()
BM = np.array([b.reshape(-1) for b in BASIS]).T          # 9 x 8, columns = basis vectors
BPINV = np.linalg.pinv(BM)
J = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]], dtype=complex)   # antidiagonal: sigma_0 = Ad_J o sigma preserves S


def twisted_map(th):
    s = np.diag([np.exp(1j * th), 1, np.exp(-1j * th)])
    cols = []
    for X in BASIS:
        Y = s @ (J @ (-X.T) @ np.linalg.inv(J)) @ np.linalg.inv(s)
        cols.append(BPINV @ Y.reshape(-1))
    return np.array(cols).T                                  # 8 x 8 matrix of Ad(s) o sigma_0


MATS = [twisted_map(t) for t in TH]
EIG = np.array([np.linalg.eigvals(Mt) for Mt in MATS])      # M x 8


def jac(eigs):
    # drop the eigenvalue 1 belonging to Lie(S) (the one closest to 1), multiply (1 - lambda) over the rest
    k = np.argmin(np.abs(eigs - 1))
    return np.prod(1 - np.delete(eigs, k))


JAC = np.array([jac(e) for e in EIG])
assert np.max(np.abs(JAC.imag)) < 1e-8 * np.max(np.abs(JAC.real)), 'jacobian not real'
JAC = JAC.real
assert JAC.min() > -1e-9, JAC.min()
ZJ = JAC.sum()

_PS = {}


def power_sums(kmax):
    if kmax not in _PS:
        _PS[kmax] = [None] + [np.sum(EIG ** k, axis=1) for k in range(1, kmax + 1)]
    return _PS[kmax]


def twisted_char_factors(n, ferm):
    p = power_sums(n)
    h = [np.ones(M, dtype=complex)]
    for m in range(1, n + 1):
        if ferm:
            h.append(sum((-1) ** (k - 1) * p[k] * h[m - k] for k in range(1, m + 1)) / m)
        else:
            h.append(sum(p[k] * h[m - k] for k in range(1, m + 1)) / m)
    return h[n]


def trace_C(content):
    """Tr of C on the SU(3) invariants of a letter content (#C-even - #C-odd)."""
    f = JAC.astype(complex)
    for L, n in Counter(content).items():
        f = f * twisted_char_factors(n, is_ferm(L))
    val = f.sum() / ZJ
    r = int(round(val.real))
    assert abs(val - r) < 1e-6, (content, val)
    return r


def sign_F(content):
    return -1 if sum(1 for L in content if is_ferm(L)) % 2 else 1


def validate(nsample=12, seed=3):
    """Brute force: dims of C-even / C-odd invariants from words (maxlen 6) at random points."""
    from fortuity.full.sector import words_of
    from fastcontents import prime_cache
    from c_parity import C_multi
    from fortuity.full.evalg import Frame, nferm
    from fortuity.bmn.modp import P, RREFBasis
    random.seed(seed)
    out = []
    for sec, Y in [((2, 2, 4, 4, 4), 4), ((3, 3, 4, 4, 4), 3), ((1, 1, 4, 4, 4), 5), ((2, 2, 6, 4, 2), 4),
                   ((3, 3, 8, 8, 8), 7), ((4, 4, 6, 4, 4), 5), ((2, 2, 6, 6, 6), 6)]:
        prime_cache(sec, Y)
        for c in contents_at(sec, Y):
            if len(out) >= nsample:
                break
            from fortuity.full.wordenum import multitraces_with_content
            ws = multitraces_with_content(c, 6)
            if not ws:
                continue
            d = sum(1 for L in c if is_ferm(L))
            cw = [C_multi(w) for w in ws]
            letters = sorted({L for w in ws for n in w for L in n} | {L for s, w in cw if s for n in w for L in n})
            k = max(d + 4, 6) if d else 0
            fr = Frame(3, 40, k, [d], letters, seed=11 + d)
            allw = list({w for w in ws} | {w for s, w in cw if s})
            idx = {w: i for i, w in enumerate(allw)}
            X = fr.eval_words(allw, dtype=np.int64)
            plus, minus = [], []
            for w, (s, w2) in zip(ws, cw):
                a = X[idx[w]]
                b = (s * X[idx[w2]]) % P if s else np.zeros_like(a)
                plus.append((a + b) % P); minus.append((a - b) % P)
            rp = RREFBasis(X.shape[1]); rp.add_batched(np.array(plus)); rm = RREFBasis(X.shape[1]); rm.add_batched(np.array(minus))
            brute = rp.rank - rm.rank
            dim_exact = content_dim(3, c)
            out.append((c, dim_exact, rp.rank + rm.rank, brute, trace_C(c)))
    ok = all(b == t and dsum == de for _, de, dsum, b, t in out)
    for c, de, dsum, b, t in out:
        print(f"  content with {len(c)} letters: dim {de} (C-even+C-odd from words {dsum}), Tr C brute {b}, twisted integral {t}")
    print('validation', 'OK' if ok else 'MISMATCH', f'({len(out)} contents)')
    return ok


def index_at(lam2, only_bmn=False):
    """[lam] hw multiplicity of the ordinary and C-twisted index at SU(3)_R weight lam2 (doubled q), J1 = J2,
    summed over all sectors (2J1=2J2=n) and letter numbers Y (fixed index weight).  Also the per-(n, Y) breakdown."""
    j = sum(lam2)                       # level = 2 sum q
    assert j % 2 == 0
    sq = j // 2                          # sum q
    tot_I = tot_C = 0
    rows = []
    for A in range(0, sq // 3 + 1):     # A = total a = total b (J1 = J2); 3A + sum|S| = sum q
        if only_bmn and A > 0:
            continue
        for Y in range(max(1, math.ceil((sq - 3 * A) / 3)), sq - A + 1):
            n = sq - A - Y
            I = Ct = 0
            for sgn, w in weyl_orbit_signed(lam2):
                if min(w) < 0:
                    continue
                for c in contents_at((n, n) + tuple(w), Y):
                    if sum(L[0] for L in c) != A:
                        continue
                    sF = sign_F(c)
                    I += sgn * sF * content_dim(3, c)
                    Ct += sgn * sF * trace_C(c)
            if I or Ct:
                rows.append((n, Y, A, I, Ct))
            tot_I += I; tot_C += Ct
    return tot_I, tot_C, rows


if __name__ == '__main__':
    print(f"jacobian: min {JAC.min():.3g}, mean {JAC.mean():.4g}")
    validate()
    lam2 = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '14,8,8').split(','))
    for bmn in (True, False):
        I, Ct, rows = index_at(lam2, only_bmn=bmn)
        print(f"\n[{'BMN letters only' if bmn else 'all letters'}] weight 2q={lam2}, J1=J2, hw multiplicity:"
              f" index I = {I}, C-twisted I_C = {Ct}, C-even part (I+I_C)/2 = {(I + Ct) / 2}, C-odd part (I-I_C)/2 = {(I - Ct) / 2}")
        for n, Y, A, a, b in rows:
            print(f"   2J1=2J2={n:3d}  Y={Y:2d}  d_BMN={A}:  I {a:5d}   I_C {b:5d}")
