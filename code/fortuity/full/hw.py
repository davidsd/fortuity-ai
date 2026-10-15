"""SU(3)_R highest-weight vectors for full-letter multitrace words.

Letters are theta-Taylor coefficients (a, b, S) of the Chang-Lin superfield.  The R-symmetry rotates the thetas;
the raising operators theta_1 d/dtheta_0 and theta_2 d/dtheta_1 induce even derivations on the field algebra:
    E1: (a, b, S) -> (a, b, S with 1 -> 0)   (1 in S, 0 not in S), coefficient +1,
    E2: (a, b, S) -> (a, b, S with 2 -> 1)   (2 in S, 1 not in S), coefficient +1
(the theta reordering signs cancel in this basis; checked against [E, Q] = 0 in test_hw.py).
E1 raises the weight (2q1, 2q2, 2q3) by (2, -2, 0), E2 by (0, 2, -2).  Both preserve the shape (a, b, |S|) of every
letter, so hw vectors are computed shape by shape as the common left null space of the E1, E2 image matrices.
"""
import numpy as np
from collections import defaultdict
from .words import canon_word, canon_multi
from .wordenum import shape
from fortuity.bmn.modp import P
from fortuity.hwcoh.ffla import nullspace_right


def E_letter(L, i):
    a, b, S = L
    src, dst = (1, 0) if i == 1 else (2, 1)
    if src in S and dst not in S:
        return (a, b, tuple(sorted((dst if x == src else x) for x in S)))
    return None


def E_multi(multi, i):
    out = defaultdict(int)
    for a, w in enumerate(multi):
        for pos, L in enumerate(w):
            nl = E_letter(L, i)
            if nl is None:
                continue
            s, cw = canon_word(w[:pos] + (nl,) + w[pos + 1:])
            if not s:
                continue
            sm, cm = canon_multi(multi[:a] + (cw,) + multi[a + 1:])
            if not sm:
                continue
            out[cm] += s * sm
    return {m: c for m, c in out.items() if c}


def nullspace_left(M):
    r, c = M.shape
    if r == 0:
        return np.zeros((0, 0), dtype=np.int64)
    if c == 0:
        return np.eye(r, dtype=np.int64)
    return nullspace_right(np.ascontiguousarray(M.T))


def hw_vectors(words, verbose=False):
    """Blocks (word_indices, H) with the rows of H (mod p) spanning the hw vectors of one shape."""
    by_shape = defaultdict(list)
    for idx, w in enumerate(words):
        by_shape[shape(w)].append(idx)
    blocks = []
    for sh, idxs in sorted(by_shape.items(), key=lambda kv: -len(kv[1])):
        cols = {}
        rows = []
        for j in idxs:
            d = {}
            for i in (1, 2):
                for m, c in E_multi(words[j], i).items():
                    d[(i, m)] = c
            for key in d:
                cols.setdefault(key, len(cols))
            rows.append(d)
        if cols:
            M = np.zeros((len(idxs), len(cols)), dtype=np.int64)
            for r, d in enumerate(rows):
                for key, c in d.items():
                    M[r, cols[key]] = c % P
            H = nullspace_left(M)
        else:
            H = np.eye(len(idxs), dtype=np.int64)
        if verbose:
            print(f"    shape {len(idxs)} words, {len(cols)} images -> {H.shape[0]} hw", flush=True)
        if H.shape[0]:
            blocks.append((np.array(idxs), H % P))
    return blocks
