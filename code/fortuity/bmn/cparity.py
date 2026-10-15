"""Charge conjugation C: X -> -X^T on multi-trace words, and the letter number Y (T-029, result R-017).

C(tr X_1...X_L) = (-1)^L (-1)^(k(k-1)/2) tr(X_L...X_1),   k = number of Grassmann-odd letters (psi) in the word,
the second sign from reversing the order of the odd matrix entries.  C is a Lie-algebra automorphism of u(N)/su(N)
(C[X,Y] = [CX,CY]), so it commutes with Q; for su(2), so(n), usp(2n), g2 it acts trivially on invariants (inner, or
identically on the algebra).  Graviton parity lemma: every single-trace graviton generator (component of tr Phi^n)
has C = (-1)^n and n letters, so P := C (-1)^Y = +1 on every multi-graviton.
"""
from fortuity.bmn.words import canon_word, canon_multi, PSI
from fortuity.bmn.modp import P as PRIME


def conj_word(w):
    """C on one trace: returns (sign, canonical word) or (0, None)."""
    k = sum(1 for l in w if l in PSI)
    s0 = (-1) ** len(w) * (-1) ** (k * (k - 1) // 2)
    s, cw = canon_word(tuple(reversed(w)))
    if s == 0:
        return 0, None
    return s0 * s, cw


def conj_expr(expr, p=PRIME):
    out = {}
    for multi, c in expr.items():
        sign = 1
        words = []
        for w in multi:
            s, cw = conj_word(w)
            if s == 0:
                sign = 0
                break
            sign *= s
            words.append(cw)
        if not sign:
            continue
        s, m = canon_multi(words)
        if s:
            out[m] = (out.get(m, 0) + sign * s * c) % p
    return {m: v for m, v in out.items() if v % p}


def canon_expr(expr, p=PRIME):
    """Canonical form of an expression whose trace words may not be canonical necklaces."""
    out = {}
    for multi, c in expr.items():
        sign, words = 1, []
        for w in multi:
            s, cw = canon_word(tuple(w))
            if s == 0:
                sign = 0
                break
            sign *= s
            words.append(cw)
        if not sign:
            continue
        s, m = canon_multi(words)
        if s:
            out[m] = (out.get(m, 0) + sign * s * c) % p
    return {m: v for m, v in out.items() if v % p}


def letters(expr):
    ys = {sum(len(w) for w in m) for m in expr}
    assert len(ys) <= 1, ys
    return ys.pop() if ys else None
