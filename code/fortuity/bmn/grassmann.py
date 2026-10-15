"""Exterior algebra Lambda[theta_1..theta_k] over F_p, homogeneous components.

A Grassmann-valued object of degree d is stored as an array whose axis 1 has length C(k, d),
indexed by sorted d-subsets of {0..k-1}.  Structure constants S[(d1,d2)] have shape
(C(k,d1+d2), C(k,d1), C(k,d2)) with entries in {0, +1, -1}.
"""
from itertools import combinations
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=None)
def subsets(k, d):
    return [tuple(c) for c in combinations(range(k), d)]


@lru_cache(maxsize=None)
def subset_index(k, d):
    return {s: i for i, s in enumerate(subsets(k, d))}


@lru_cache(maxsize=None)
def structure(k, d1, d2):
    """Structure tensor for Lambda^d1 x Lambda^d2 -> Lambda^(d1+d2)."""
    d = d1 + d2
    S1, S2 = subsets(k, d1), subsets(k, d2)
    idx = subset_index(k, d)
    T = np.zeros((len(subsets(k, d)), len(S1), len(S2)), dtype=np.int64)
    for i, s in enumerate(S1):
        ss = set(s)
        for j, t in enumerate(S2):
            if ss & set(t):
                continue
            # sign of sorting concatenation s+t : number of pairs (a in s, b in t) with a > b
            inv = sum(1 for a in s for b in t if a > b)
            T[idx[tuple(sorted(s + t))], i, j] = -1 if inv % 2 else 1
    return T


def gmul_scalar(A, B, k, p):
    """Product of homogeneous Grassmann scalars: A: dict{deg: (P, C)}, B likewise -> dict."""
    out = {}
    for d1, a in A.items():
        for d2, b in B.items():
            d = d1 + d2
            if d > k:
                continue
            T = structure(k, d1, d2)
            r = np.einsum('oij,pi,pj->po', T, a, b) % p
            out[d] = (out[d] + r) % p if d in out else r
    return out


def gmul_matrix(A, B, k, p):
    """Product of homogeneous Grassmann matrices: A: dict{deg: (P, C, N, N)}."""
    out = {}
    for d1, a in A.items():
        for d2, b in B.items():
            d = d1 + d2
            if d > k:
                continue
            T = structure(k, d1, d2)
            if d1 == 0 and d2 == 0:
                r = np.matmul(a, b) % p
            else:
                ab = np.einsum('pinm,pjml->pijnl', a, b)  # (P, C1, C2, N, N), entries < 2^52*N
                ab %= p
                r = np.einsum('oij,pijnl->ponl', T, ab) % p
            out[d] = (out[d] + r) % p if d in out else r
    return out


def gtrace(A, p):
    return {d: np.trace(a, axis1=2, axis2=3) % p for d, a in A.items()}


def gadd(A, B, p, coeff=1):
    out = dict(A)
    for d, b in B.items():
        r = (coeff * b) % p
        out[d] = (out[d] + r) % p if d in out else r
    return out
