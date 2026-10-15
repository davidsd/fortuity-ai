"""Helpers: load O from JSON; build graviton polynomials from generator names in the code's normalisation."""
import json
from collections import defaultdict
from fortuity.bmn.words import graviton_generators, mul_expr, expr_sector
from fortuity.bmn.symdirect import weyl_multi, PERMS, perm_sign

L = {'X': 0, 'Y': 1, 'Z': 2}
GENS = dict(graviton_generators(3, 1))

def parse_name(s):
    """'u^{XY}' -> ('u',2,(0,1)); 'v^{X}_{Y}' -> ('v',2,(0,),1); 'v^{XY}_{Z}' -> ('v',3,(0,1),2); 'w^{ZZ}' -> ('w',3,(2,2))"""
    kind = s[0]
    up = s[s.index('{') + 1:s.index('}')]
    idx = tuple(sorted(L[c] for c in up))
    if kind == 'u':
        return ('u', len(idx), idx)
    if kind == 'w':
        return ('w', len(idx) + 1, idx)
    low = s[s.rindex('{') + 1:s.rindex('}')]
    return ('v', len(idx) + 1, idx, L[low])

def poly_from_terms(terms):
    """terms: list of (coeff, [names]) -> expression dict (integer coefficients), product in the given order."""
    out = defaultdict(int)
    for c, names in terms:
        e = {(): 1}
        for n in names:
            e = mul_expr(e, GENS[parse_name(n)])
        for m, v in e.items():
            out[m] += c * v
    return {m: v for m, v in out.items() if v}

def load_O(path):
    d = json.load(open(path))
    return {tuple(tuple(w) for w in m): c for c, m in d['terms']}, d

def weyl_image(expr, s, with_sign=True):
    """expr under the Weyl element s (pure relabelling if with_sign=False)."""
    out = defaultdict(int)
    for m, c in expr.items():
        sg, m2 = weyl_multi(m, s)
        if sg:
            if not with_sign:
                nlet = sum(1 for w in m for l in w if l != 6)
                if perm_sign(s) == -1 and nlet % 2 == 1:
                    sg = -sg
            out[m2] += sg * c
    return {m: v for m, v in out.items() if v}

def same(a, b, scale=1):
    if set(a) != set(b):
        return False
    return all(a[m] == scale * b[m] for m in a)


def canon_expr(expr):
    """Canonicalise every word (cyclic rotation with fermionic sign) and every multi-trace (sorted, sign);
    merges keys that the BMN code stores separately (it tolerates non-canonical rotations)."""
    from fortuity.bmn.words import canon_word, canon_multi
    out = defaultdict(int)
    for m, c in expr.items():
        sign = 1
        ws = []
        for w in m:
            s, cw = canon_word(w)
            if s == 0:
                sign = 0
                break
            sign *= s
            ws.append(cw)
        if sign == 0:
            continue
        s2, cm = canon_multi(ws)
        if s2 == 0:
            continue
        out[cm] += sign * s2 * c
    return {m: v for m, v in out.items() if v}


def ratrec(a, p):
    """Rational reconstruction of a mod p: r/s with |r|, |s| <= sqrt(p/2), or None."""
    from fractions import Fraction
    a %= p
    r0, r1, s0, s1 = p, a, 0, 1
    bound = int((p // 2) ** 0.5)
    while r1 > bound:
        q = r0 // r1
        r0, r1, s0, s1 = r1, r0 - q * r1, s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return Fraction(r1, s1) if s1 > 0 else Fraction(-r1, -s1)
