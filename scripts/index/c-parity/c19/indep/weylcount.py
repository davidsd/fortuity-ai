"""Independent count of S_N invariants (N = 2, 3) of super-polynomials in Cartan-valued (traceless diagonal) BMN letters.

Written from scratch (does not import the other agent's molien.py / brute.py).

Letter charges derived here from the formula in code/fortuity/bmn/words.py:8,
    q_I = n_phi_I + (n_psi - n_psi_I) + n_f ,   2J = n_psi + 2 n_f ,
applied to unit count vectors.  Letter 0..2 = phi_m (boson), 3..5 = psi_m (fermion), 6 = f (boson).

Each letter type L contributes coordinates on V0 = {x in C^N : sum x_i = 0}.
N = 2: one coordinate y_L (x = (y, -y)); the transposition acts y_L -> -y_L.  A monomial picks up (-1)^deg, so the
       invariants at a count vector are: the unique monomial (fermion exponents <= 1) iff Y = total degree is even.
N = 3: eigen-coordinates z_L = x1 + w x2 + w^2 x3, zb_L = x1 + w^2 x2 + w x3 (w = e^{2 pi i/3}); they span V0* (x) C.
       3-cycle: z -> w^2 z, zb -> w zb (so monomials are eigenvectors, charge #z - #zb mod 3);
       transposition (23): z <-> zb (linear substitution letter by letter).
       Invariants = span of {m + s(m)} over Z3-invariant monomials m.  Dimension = #(2-element orbits) + #(fixed m with
       s(m) = +m).  For fermionic letter with both z and zb: s(z zb) = zb z = - z zb.
Two implementations for N=3: explicit monomial enumeration with generic Grassmann reordering signs (enumerate_N3) and a
per-count-vector closed form (formula_N3).  They are cross-checked against each other.
"""
from itertools import product
from collections import defaultdict


def letter_table():
    out = []
    for L in range(7):
        n = [0] * 7
        n[L] = 1
        nphi, npsi, nf = n[0:3], n[3:6], n[6]
        npsit = sum(npsi)
        q = tuple(nphi[I] + (npsit - npsi[I]) + nf for I in range(3))
        twoJ = npsit + 2 * nf
        out.append((L, 3 <= L <= 5, twoJ, q))
    return out


LET = letter_table()


def count_vectors(N, Q):
    """all count vectors (n_0..n_6) with |q| <= Q, fermion exponents <= N-1 (Lambda^{>N-1} V0 = 0).
    yields (n, (2J, q1, q2, q3))."""
    w = [sum(q) for (_, _, _, q) in LET]
    ranges = []

    def rec(L, n, used):
        if L == 7:
            twoJ = sum(n[l] * LET[l][2] for l in range(7))
            q = tuple(sum(n[l] * LET[l][3][I] for l in range(7)) for I in range(3))
            yield tuple(n), (twoJ,) + q
            return
        maxn = (Q - used) // w[L]
        if LET[L][1]:
            maxn = min(maxn, N - 1)
        for k in range(maxn + 1):
            n.append(k)
            yield from rec(L + 1, n, used + k * w[L])
            n.pop()
    yield from rec(0, [], 0)


def count_N2(Q):
    out = defaultdict(int)
    for n, key in count_vectors(2, Q):
        if sum(n) % 2 == 0:
            out[key] += 1
    return dict(out)


def formula_N3(Q):
    out = defaultdict(int)
    for n, key in count_vectors(3, Q):
        # charge distribution mod 3 of choices
        dist = [1, 0, 0]
        fixed_possible = True
        nferm2 = 0
        for L in range(7):
            k = n[L]
            loc = [0, 0, 0]
            if LET[L][1]:
                if k == 0:
                    loc[0] += 1
                elif k == 1:
                    loc[1] += 1; loc[2] += 1      # z (+1) or zb (-1 = 2 mod 3)
                    fixed_possible = False
                else:
                    loc[0] += 1                   # z zb
                    nferm2 += 1
            else:
                for a in range(k + 1):
                    loc[(2 * a - k) % 3] += 1
                if k % 2:
                    fixed_possible = False
            new = [0, 0, 0]
            for i in range(3):
                for j in range(3):
                    new[(i + j) % 3] += dist[i] * loc[j]
            dist = new
        M0 = dist[0]
        F = 1 if fixed_possible else 0
        Fplus = 1 if (fixed_possible and nferm2 % 2 == 0) else 0
        assert (M0 - F) % 2 == 0
        d = (M0 - F) // 2 + Fplus
        if d:
            out[key] += d
    return dict(out)


def perm_sign_sort(seq):
    """sign of the permutation sorting seq (distinct items)."""
    a = list(seq)
    sgn = 1
    for i in range(len(a)):
        for j in range(len(a) - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                sgn = -sgn
    return sgn


def enumerate_N3(Q):
    """explicit monomials: bosons (L, nz, nzb); fermions as an ordered product of distinct generators (L, c), c in {0:z,1:zb}.
    s acts generator-wise c -> 1-c in place; then reorder to canonical (sorted) order with Grassmann sign."""
    out = defaultdict(int)
    for n, key in count_vectors(3, Q):
        choices = []
        for L in range(7):
            k = n[L]
            if LET[L][1]:
                if k == 0:
                    choices.append([()])
                elif k == 1:
                    choices.append([((L, 0),), ((L, 1),)])
                else:
                    choices.append([((L, 0), (L, 1))])
            else:
                choices.append([(('b', L, a, k - a),) for a in range(k + 1)])
        seen = set()
        d = 0
        for ch in product(*choices):
            charge = 0
            ferm = []
            bos = []
            for L, c in zip(range(7), ch):
                if LET[L][1]:
                    for (LL, cc) in c:
                        ferm.append((LL, cc))
                        charge += 1 if cc == 0 else -1
                else:
                    (_, LL, a, b) = c[0]
                    bos.append((LL, a, b))
                    charge += a - b
            if charge % 3:
                continue
            ferm_c = tuple(sorted(ferm))
            m = (tuple(bos), ferm_c)
            if m in seen:
                continue
            # image under s: in-place substitution then reorder
            img_f = [(LL, 1 - cc) for (LL, cc) in ferm_c]
            sgn = perm_sign_sort(img_f)
            sm = (tuple((LL, b, a) for (LL, a, b) in bos), tuple(sorted(img_f)))
            seen.add(m)
            if sm == m:
                if sgn == 1:
                    d += 1
            else:
                seen.add(sm)
                d += 1
        if d:
            out[key] += d
    return dict(out)


if __name__ == '__main__':
    print('letter table (L, fermionic, 2J, q):', LET)
