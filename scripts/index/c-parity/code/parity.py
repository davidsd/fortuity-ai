"""Moved to the shared library as fortuity.bmn.cparity (R-017, 2026-09-29); shim + the generator check script."""
from fortuity.bmn.cparity import conj_word, conj_expr, canon_expr, letters, PRIME   # noqa: F401

if __name__ == '__main__':
    # check: every graviton generator (n <= 7) is a C-eigenvector with eigenvalue (-1)^n
    from fortuity.bmn.words import graviton_generators
    bad = 0
    tot = 0
    for name, e in graviton_generators(7, 1):
        if not e:
            continue
        tot += 1
        e = canon_expr(e)
        Y = letters(e)
        ce = conj_expr(e)
        sgn = (-1) ** Y
        diff = {m: (ce.get(m, 0) - sgn * e.get(m, 0)) % PRIME for m in set(ce) | set(e)}
        if any(diff.values()):
            bad += 1
            print('NOT an eigenvector:', name, 'Y =', Y)
    print(f'{tot} graviton generators (n <= 7) checked: {tot - bad} satisfy C = (-1)^Y')
