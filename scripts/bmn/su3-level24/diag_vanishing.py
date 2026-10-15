"""How R_24^{(0,3)}, (R_26^{(0,4)})^i, R_30^{(0,5)} and (R_30^{(0,5)})^{ijk} of arXiv:2312.16443 (A.4) vanish on diagonal
matrices at N = 3, and how much of that is forced by the quantum numbers alone.

Diagonal traceless letters: phi^i = diag(x^i_k), psi_j = diag(th_{j,k}) (Grassmann), f = diag(f_k), sum over k = 0.
Part A: the four polynomials and their factors, x numeric (random mod p), theta symbolic; N = 3 and the control N = 4.
Part B (N = 3): dimension of the space of ALL functions of the eigenvalues with a given letter content
        (n_phi, n_psi, n_f) and SU(3) highest weight, with and without invariance under the Weyl group S_3.
        Exact linear algebra mod p on polynomials in the independent eigenvalues (labels 1, 2; label 3 = -1 - 2):
        kernel of the two raising operators, and of s - 1 for the two generators s of S_3.
Part C: other gauge-invariant operators with the same quantum numbers, evaluated on the diagonal.
usage: python3 diag_vanishing.py"""
import random
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np

P = 2 ** 31 - 1
R3 = range(3)
inv = lambda a: pow(a % P, P - 2, P)


def eps(a, b, c):
    return {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.get((a, b, c), 0)


# ---------------------------------------------------------------- Grassmann algebra, coefficients mod p
def gmul(a, b):
    out = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            if m1 & m2:
                continue
            s, t = 0, m1 >> 1
            while t:
                s += bin(t & m2).count('1')
                t >>= 1
            out[m1 | m2] = (out.get(m1 | m2, 0) + (-c1 * c2 if s & 1 else c1 * c2)) % P
    return {m: c for m, c in out.items() if c}


def gadd(*terms):
    out = {}
    for c, a in terms:
        for m, v in a.items():
            out[m] = (out.get(m, 0) + c * v) % P
    return {m: v for m, v in out.items() if v}


def gprod(*fs):
    out = {0: 1}
    for f in fs:
        out = gmul(out, f)
    return out


def gsum(terms):
    return gadd(*[(1, t) for t in terms])


def verdict(g):
    if isinstance(g, bool):
        return 'nonzero' if g else '0'
    return '0' if not g else 'nonzero (%d monomials)' % len(g)


# ---------------------------------------------------------------- parts A and C: operators on the diagonal
def part_AC(N, seed=5):
    rng = random.Random(seed)
    x = [[rng.randrange(1, P) for _ in range(N - 1)] for _ in R3]
    f = [rng.randrange(1, P) for _ in range(N - 1)]
    f.append(-sum(f) % P)
    for i in R3:
        x[i].append(-sum(x[i]) % P)                                    # x[i][k]
    th = [[{1 << (j * (N - 1) + k): 1} for k in range(N - 1)] for j in R3]
    for j in R3:
        th[j].append(gadd(*[(-1, t) for t in th[j]]))                  # th[j][k]
    K = range(N)
    u2 = {(i, j): sum(x[i][k] * x[j][k] for k in K) % P for i in R3 for j in R3}
    u3 = {(i, j, l): sum(x[i][k] * x[j][k] * x[l][k] for k in K) % P for i in R3 for j in R3 for l in R3}
    M = {(i, j): gadd(*[(x[i][k], th[j][k]) for k in K]) for i in R3 for j in R3}                 # tr(phi^i psi_j)
    tau = gsum(M[(a, a)] for a in R3)
    v2 = {(i, j): gadd((1, M[(i, j)]), (-inv(3) if i == j else 0, tau)) for i in R3 for j in R3}
    M3 = {(i, j, l): gadd(*[(x[i][k] * x[j][k], th[l][k]) for k in K]) for i in R3 for j in R3 for l in R3}
    t1 = {j: gsum(M3[(j, a, a)] for a in R3) for j in R3}                                         # tr(phi^j phi^a psi_a)
    v3 = {(i, j, l): gadd((1, M3[(i, j, l)]), (-inv(4) if i == l else 0, t1[j]), (-inv(4) if j == l else 0, t1[i]))
          for i in R3 for j in R3 for l in R3}
    G = [[gadd(*[(x[d][l], th[d][k]) for d in R3]) for l in K] for k in K]                        # G_kl = th_k . x_l
    out = {}

    def mm(A, B, n):
        return [[gsum(gmul(A[i][k], B[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]

    def tr(A, n):
        return gsum(A[i][i] for i in range(n))

    # --- part A
    E = {c: gadd(*[(eps(a, b, c) * u2[(a, d)], v2[(b, d)]) for a in R3 for b in R3 for d in R3 if eps(a, b, c)]) for c in R3}
    W = {a: gsum(gmul(v3[(a, c, d)], v2[(d, c)]) for c in R3 for d in R3) for a in R3}             # v^{ac}_d v^d_c
    out['R_24 = E_a W^a,  E_a = eps_{bca} u^{bd} v^c_d,  W^a = v^{ac}_d v^d_c'] = gsum(gmul(E[c], W[c]) for c in R3)
    if N == 3:
        n = [(x[(c + 1) % 3][0] * x[(c + 2) % 3][1] - x[(c + 2) % 3][0] * x[(c + 1) % 3][1]) % P for c in R3]
        xi = gadd((3, G[1][0]), (-3, G[0][1]))
        out['   E_a - 3 (G_21 - G_12) n_a   (n = x_1 cross x_2: E is along the normal of the plane)'] = \
            any(bool(gadd((1, E[c]), (-n[c], xi))) for c in R3)
        out['   n_a W^a   (W lies in the plane)'] = gadd(*[(n[a], W[a]) for a in R3])
    V = [[v2[(i, j)] for j in R3] for i in R3]
    V2 = mm(V, V, 3)
    V3 = mm(V2, V, 3)
    V5 = mm(V3, V2, 3)
    out['tr v^3'] = tr(V3, 3)
    if N == 3:
        al = [[gadd((1, th[j][A]), (-1, th[j][2])) for j in R3] for A in range(2)]                 # alpha_A = th_A - th_3
        m = [[gadd(*[(x[j][B], al[A][j]) for j in R3]) for B in range(2)] for A in range(2)]       # m_AB = alpha_A . x_B
        out['   tr v^3 - 3 (m_11 - m_22) m_12 m_21'] = gadd((1, tr(V3, 3)), (-3, gprod(gadd((1, m[0][0]), (-1, m[1][1])),
                                                                                         m[0][1], m[1][0])))
    out['R_30 [0,0] = tr v^5'] = tr(V5, 3)
    if N == 3:
        # the [3,0] entry contracted with eta^3: prefactor = sum_AB eta_A eta_B [eta, alpha_A, alpha_B], expanded in the
        # basis dual to (x_1, x_2, e_3): m_AB in-plane components of the fermions, mu_A = alpha_A . e_3 the third one
        e3 = [rng.randrange(1, P) for _ in R3]
        eta = [rng.randrange(1, P) for _ in R3]
        col = lambda c: [x[i][c] for i in R3] if c < 2 else e3
        D = sum(eps(a, b, c) * col(0)[a] * col(1)[b] * col(2)[c] for a in R3 for b in R3 for c in R3) % P
        et = [sum(eta[i] * col(c)[i] for i in R3) % P for c in range(3)]            # eta_1, eta_2, eta_3
        mu = [gadd(*[(e3[j], al[A][j]) for j in R3]) for A in range(2)]
        etaV = [gadd(*[(eta[j], v2[(j, a)]) for j in R3]) for a in R3]
        pre_eta = gadd(*[(eps(i, a, b) * eta[i], gmul(etaV[a], etaV[b])) for i in R3 for a in R3 for b in R3 if eps(i, a, b)])
        det3 = lambda A, B: gadd((et[0], gadd((1, gmul(m[A][1], mu[B])), (-1, gmul(mu[A], m[B][1])))),
                                 (-et[1], gadd((1, gmul(m[A][0], mu[B])), (-1, gmul(mu[A], m[B][0])))),
                                 (et[2], gadd((1, gmul(m[A][0], m[B][1])), (-1, gmul(m[A][1], m[B][0])))))
        rhs = gadd(*[(et[A] * et[B] * inv(D), det3(A, B)) for A in range(2) for B in range(2)])
        out['   [3,0] prefactor (eta^3 component) - (1/D) sum_AB eta_A eta_B det(eta; alpha_A; alpha_B)'] = gadd((1, pre_eta), (-1, rhs))
        Om = gprod(m[0][0], m[1][1], m[0][1], m[1][0])
        out['   m_AB tr v^3 + 3 Omega delta_AB, Omega = m_11 m_22 m_12 m_21'] = any(
            bool(gadd((1, gmul(m[A][B], tr(V3, 3))), (3 if A == B else 0, Om))) for A in range(2) for B in range(2))
        out['   Omega itself'] = Om
    Wp = {b: gsum(gmul(v2[(d, c)], v3[(b, c, d)]) for c in R3 for d in R3) for b in R3}
    for i in R3:
        out['R_26^%d = v^i_a v^a_b v^d_c v^{bc}_d' % (i + 1)] = gsum(gmul(V2[i][b], Wp[b]) for b in R3)
    bad, pre0 = 0, None
    for i in R3:
        for j in range(i, 3):
            for k in range(j, 3):
                tot = {}
                for (p, q, r) in set(permutations((i, j, k))):
                    pre = gadd(*[(eps(a, b, p), gmul(v2[(q, a)], v2[(r, b)])) for a in R3 for b in R3 if eps(a, b, p)])
                    pre0 = pre if pre0 is None else pre0
                    tot = gadd((1, tot), (1, gmul(pre, tr(V3, 3))))
                bad += bool(tot)
    out['R_30 [3,0] = eps^{ab(i} v^j_a v^{k)}_b tr v^3, all 10 components'] = bool(bad)
    out['   its prefactor eps^{ab1} v^1_a v^1_b alone'] = pre0
    # --- part C: other operators with the same quantum numbers (untraced generators allowed)
    G2 = mm(G, G, N)
    G3 = mm(G2, G, N)
    G4 = mm(G2, G2, N)
    out['C [0,0] (6,3,0): eps_{abc} u^{ade} M^b_d M^c_f M^f_e'] = gadd(*[
        (eps(a, b, c) * u3[(a, d, e)], gprod(M[(b, d)], M[(c, g)], M[(g, e)]))
        for a in R3 for b in R3 for c in R3 if eps(a, b, c) for d in R3 for e in R3 for g in R3])
    out['C [0,0] (6,3,0): eps_{abc} u^{ad} M^b_d t^c tau'] = gadd(*[
        (eps(a, b, c) * u2[(a, d)], gprod(M[(b, d)], t1[c], tau)) for a in R3 for b in R3 for c in R3 if eps(a, b, c)
        for d in R3])
    q = {(c, d): gadd(*[(x[c][k] * x[d][k] * f[k], G[k][k]) for k in K]) for c in R3 for d in R3}
    out['C [0,0] (7,1,1): eps eps u u tr(phi phi phi^a psi_a f)'] = gadd(*[
        (eps(a, b, c) * eps(d, e, g) * u2[(a, d)] * u2[(b, e)], q[(c, g)])
        for a in R3 for b in R3 for c in R3 if eps(a, b, c) for d in R3 for e in R3 for g in R3 if eps(d, e, g)])
    out['C [0,0] (5,5,0): sum_k G_kk (G^4)_kk'] = gsum(gmul(G[k][k], G4[k][k]) for k in K)
    out['C [0,0] (5,5,0): tau sum_k (G^2)_kk (G^2)_kk'] = gmul(tau, gsum(gmul(G2[k][k], G2[k][k]) for k in K))
    for name, ck in (('sum_k x_k (G^4)_kk', [G4[k][k] for k in K]),
                     ('sum_k x_k G_kk (G^3)_kk', [gmul(G[k][k], G3[k][k]) for k in K]),
                     ('sum_k x_k (G^2)_kk (G^2)_kk', [gmul(G2[k][k], G2[k][k]) for k in K]),
                     ('sum_kl x_k G_kl G_ll (G^2)_ll', [gsum(gprod(G[k][l], G[l][l], G2[l][l]) for l in K) for k in K])):
        val = [gadd(*[(x[i][k], ck[k]) for k in K]) for i in R3]
        out['C [1,0] (5,4,0): ' + name] = any(bool(v) for v in val)
    cr = lambda k, c: gadd(*[(eps(a, b, c), gmul(th[a][k], th[b][k])) for a in R3 for b in R3 if eps(a, b, c)])
    L = {c: gsum(gmul(th[c][l], G2[l][l]) for l in K) for c in R3}
    out['C [3,0] (5,5,0), component 111: sum_k (x^1_k)^3 (th_k cross th_k)^c sum_l th_{c,l} (G^2)_ll'] = gadd(*[
        (pow(x[0][k], 3, P), gmul(cr(k, c), L[c])) for k in K for c in R3])
    out['C [3,0] (5,5,0), component 111: u^{11} sum_l (th_l cross th_l)^1 (G^3)_ll'] = gadd(*[
        (u2[(0, 0)], gmul(cr(l, 0), G3[l][l])) for l in K])
    return out


# ---------------------------------------------------------------- part B: all functions on the N = 3 diagonal
NA = 2                      # independent eigenvalues; variables x^i_A -> 3A + i, theta_{j,A} -> bit 3A + j, f_A -> A
NX = 3 * NA
ZE, ZF = (0,) * NX, (0,) * NA


def weight(key):
    ex, mask, ef = key
    return tuple(sum(ex[3 * A + i] for A in range(NA)) - sum((mask >> (3 * A + i)) & 1 for A in range(NA)) for i in R3)


def space(nphi, npsi, nf):
    out = {}
    fl = []
    for c in combinations_with_replacement(range(NA), nf):
        fl.append(tuple(c.count(A) for A in range(NA)))
    for cx in combinations_with_replacement(range(NX), nphi):
        ex = tuple(cx.count(v) for v in range(NX))
        for ct in combinations(range(NX), npsi):
            mask = sum(1 << b for b in ct)
            for ef in fl:
                key = (ex, mask, ef)
                out.setdefault(weight(key), []).append(key)
    return out


def raise_op(src, dst, key):
    """sl(3) generator x^src -> x^dst, theta_dst -> -theta_src (pairings theta.x invariant): weight + e_dst - e_src"""
    ex, mask, ef = key
    out = {}
    for A in range(NA):
        vs, vd = 3 * A + src, 3 * A + dst
        if ex[vs]:
            e = list(ex)
            e[vs] -= 1
            e[vd] += 1
            k = (tuple(e), mask, ef)
            out[k] = out.get(k, 0) + ex[vs]
        if (mask >> vd) & 1 and not (mask >> vs) & 1:
            s1 = bin(mask & ((1 << vd) - 1)).count('1')
            m2 = mask & ~(1 << vd)
            s2 = bin(m2 & ((1 << vs) - 1)).count('1')
            k = (ex, m2 | (1 << vs), ef)
            out[k] = out.get(k, 0) - (-1) ** (s1 + s2)
    return out


def pmul(p, q):
    out = {}
    for (e1, m1, f1), c1 in p.items():
        for (e2, m2, f2), c2 in q.items():
            if m1 & m2:
                continue
            s, t = 0, m1 >> 1
            while t:
                s += bin(t & m2).count('1')
                t >>= 1
            k = (tuple(a + b for a, b in zip(e1, e2)), m1 | m2, tuple(a + b for a, b in zip(f1, f2)))
            out[k] = out.get(k, 0) + (-c1 * c2 if s & 1 else c1 * c2)
    return {k: v for k, v in out.items() if v}


def subst(key, R):
    """the monomial after the linear change of eigenvalue labels  z_A -> sum_B R[A][B] z_B  (z = x^i, theta_j, f)"""
    ex, mask, ef = key
    unit = lambda v, n: tuple(1 if w == v else 0 for w in range(n))
    out = {(ZE, 0, ZF): 1}
    for A in range(NA):
        for i in R3:
            lin = {(unit(3 * B + i, NX), 0, ZF): R[A][B] for B in range(NA) if R[A][B]}
            for _ in range(ex[3 * A + i]):
                out = pmul(out, lin)
    for b in range(NX):
        if (mask >> b) & 1:
            out = pmul(out, {(ZE, 1 << (3 * B + b % 3), ZF): R[b // 3][B] for B in range(NA) if R[b // 3][B]})
    for A in range(NA):
        lin = {(ZE, 0, unit(B, NA)): R[A][B] for B in range(NA) if R[A][B]}
        for _ in range(ef[A]):
            out = pmul(out, lin)
    return out


def rank_modp(M):
    M = M % P
    r = 0
    for c in range(M.shape[1]):
        if r == M.shape[0]:
            break
        nz = np.flatnonzero(M[r:, c])
        if not nz.size:
            continue
        p = r + int(nz[0])
        if p != r:
            M[[r, p]] = M[[p, r]]
        M[r] = M[r] * inv(int(M[r, c])) % P
        rows = np.flatnonzero(M[:, c])
        rows = rows[rows != r]
        if rows.size:
            M[rows] = (M[rows] - M[rows, c:c + 1] * M[r:r + 1]) % P
        r += 1
    return r


S3_GENS = ([[0, 1], [1, 0]], [[1, 0], [-1, -1]])          # (12) and (23) on the labels, with label 3 = -1 - 2


def hw_dims(nphi, npsi, nf, w):
    """(dim of the weight space, dim of highest-weight vectors, dim of S_3-invariant highest-weight vectors)"""
    cols = space(nphi, npsi, nf).get(tuple(w), [])
    if not cols:
        return 0, 0, 0
    res = []
    for weyl in (False, True):
        rows, ent = {}, []
        for c, key in enumerate(cols):
            imgs = [('E1', raise_op(1, 0, key)), ('E2', raise_op(2, 1, key))]
            if weyl:
                for t, R in enumerate(S3_GENS):
                    img = subst(key, R)
                    img[key] = img.get(key, 0) - 1
                    imgs.append(('s%d' % t, img))
            for tag, img in imgs:
                for k, v in img.items():
                    if v % P:
                        ent.append((rows.setdefault((tag, k), len(rows)), c, v % P))
        Mx = np.zeros((max(1, len(rows)), len(cols)), dtype=np.int64)
        for r, c, v in ent:
            Mx[r, c] = v
        res.append(len(cols) - rank_modp(Mx))
    return len(cols), res[0], res[1]


if __name__ == '__main__':
    for N in (3, 4):
        print(f"parts A, C: N = {N}, diagonal traceless, {3 * (N - 1)} independent Grassmann eigenvalues")
        for name, g in part_AC(N).items():
            print(f"  {name}: {verdict(g)}")
    print("part B: N = 3, all functions of the eigenvalues in the charge sector of each entry of (A.4);")
    print("        per letter content (n_phi, n_psi, n_f): highest-weight vectors / S_3-invariant highest-weight vectors")
    A4 = [(10, (1, 2), (5, 0)), (12, (0, 0), (6, 0)), (12, (2, 2), (6, 0)), (12, (0, 3), (4, 1)), (12, (1, 1), (4, 1)),
          (12, (2, 2), (4, 1)), (14, (1, 0), (5, 1)), (14, (0, 2), (5, 1)), (14, (2, 1), (5, 1)), (14, (1, 3), (5, 1)),
          (14, (3, 2), (5, 1)), (14, (1, 3), (3, 2)), (16, (0, 1), (4, 2)), (16, (1, 2), (4, 2)), (16, (2, 3), (4, 2)),
          (18, (0, 0), (5, 2)), (20, (1, 0), (4, 3)), (22, (2, 0), (5, 3)), (24, (0, 0), (6, 3)), (26, (1, 0), (5, 4)),
          (30, (0, 0), (5, 5)), (30, (3, 0), (5, 5))]
    for j, (p, q), (nphi, npsi) in [(18, (0, 0), (3, 3))] + A4:
        assert 2 * nphi + 4 * npsi == j
        parts, tot = [], 0
        for r in range(npsi // 2 + 1):
            content = (nphi + r, npsi - 2 * r, r)
            c3 = content[0] - content[1] - p - 2 * q
            assert c3 % 3 == 0
            c = c3 // 3
            d = hw_dims(*content, (p + q + c, q + c, c))
            tot += d[2]
            parts.append(f"{content}: {d[1]}/{d[2]}")
        tag = 'control: sector of tr v^3' if (j, nphi, npsi) == (18, 3, 3) else 'R_%d^(0,%d)' % (j, npsi)
        print(f"  t^{j} [{p},{q}] {tag:26s} S_3-invariant functions {tot:3d}   " + '  '.join(parts), flush=True)
