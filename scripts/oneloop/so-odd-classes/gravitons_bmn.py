"""Graviton (multi-graviton) polynomials of a BMN sector as explicit expressions, converted to superfield letters.
Spanning set: all products of the single-trace generators u_n, v_n, w_n (fortuity.bmn.words.graviton_generators,
n <= nmax) whose charges add up to the sector; linear dependences are harmless."""
from collections import defaultdict
from fortuity.bmn.words import graviton_generators, mul_expr, PHI, PSI, F
from so7_lift import CONV
from fortuity.full.words import add_term

CH = {PHI[0]: (0, (1, 0, 0)), PHI[1]: (0, (0, 1, 0)), PHI[2]: (0, (0, 0, 1)),
      PSI[0]: (1, (0, 1, 1)), PSI[1]: (1, (1, 0, 1)), PSI[2]: (1, (1, 1, 0)), F: (2, (1, 1, 1))}


def sector(expr):
    m = next(iter(expr))
    tj, q = 0, [0, 0, 0]
    for w in m:
        for l in w:
            a, b = CH[l]
            tj += a
            q = [x + y for x, y in zip(q, b)]
    return (tj,) + tuple(q)


def monomials(sec, nmax=8):
    gens = [(name, e) for name, e in graviton_generators(nmax, 1, qmax=sec) if e]
    gsec = [sector(e) for _, e in gens]
    gpar = [s[0] % 2 for s in gsec]          # fermionic iff 2J odd
    out = []

    def rec(start, rem, chosen):
        if not any(rem):
            out.append(tuple(chosen)); return
        for i in range(start, len(gens)):
            if all(a <= b for a, b in zip(gsec[i], rem)):
                rec(i + 1 if gpar[i] else i, tuple(b - a for a, b in zip(gsec[i], rem)), chosen + [i])
    rec(0, tuple(sec), [])
    return gens, out


def to_full(expr):
    O = defaultdict(int)
    for m, c in expr.items():
        sign, multi = c, []
        for w in m:
            ww = []
            for l in w:
                s, L = CONV[l]
                sign *= s
                ww.append(L)
            multi.append(tuple(ww))
        add_term(O, tuple(multi), sign)
    return {m: c for m, c in O.items() if c}


def graviton_exprs(sec, nmax=8):
    gens, monos = monomials(sec, nmax)
    for mono in monos:
        e = gens[mono[0]][1]
        for gi in mono[1:]:
            e = mul_expr(e, gens[gi][1])
        e = {m: c for m, c in e.items() if c}
        if e:
            yield mono, to_full(e)


if __name__ == "__main__":
    gens, monos = monomials((1, 3, 3, 3))
    print(len(gens), "generators fit;", len(monos), "graviton monomials in (1;3,3,3)")
