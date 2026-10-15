"""Q_1 (one-loop supercharge on letters with derivatives, q1d.py) of K word vectors, in parallel.

apply(words, C) -> (img, V): img = list of output multitraces, V = K x len(img) int64 mod p with
V[k] = Q_1(sum_j C[k, j] words[j]).
- Q_1's coefficients are fractions mod p (up to 2^26), so the sparse products are formed in two 13-bit halves of C:
  every term is < 2^39 and a sum of fewer than 2^24 of them fits in int64.
- Workers (fork) take contiguous word slices and return their image words packed into fixed-width uint64 keys
  (canonical multitraces have unique keys).  The parent merges by key with numpy (lexsort + reduceat), so a dense
  representative (millions of words, ~90 image words each at T_2) never needs a Python dict over all image words."""
import multiprocessing as mp
import numpy as np
import scipy.sparse as sp
import q1d
from fortuity.bmn.modp import P

W = 4              # uint64 per key: 5 symbols of 12 bits each -> 20 symbols (letters + necklace separators)
SEP = 4095


def _sym(L):
    a, b, S = L
    assert a < 16 and b < 16, L
    m = 0
    for s in S:
        m |= 1 << s
    return 1 + ((a << 7) | (b << 3) | m)


def _unsym(x):
    x -= 1
    return (x >> 7, (x >> 3) & 15, tuple(s for s in range(3) if (x >> s) & 1))


def encode(words):
    out = np.zeros((len(words), W), dtype=np.uint64)
    for i, w in enumerate(words):
        syms = []
        for n_, neck in enumerate(w):
            if n_:
                syms.append(SEP)
            syms.extend(_sym(L) for L in neck)
        assert len(syms) <= 5 * W, w
        for q in range(0, len(syms), 5):
            v = 0
            for s in syms[q:q + 5]:
                v = (v << 12) | s
            v <<= 12 * (5 - len(syms[q:q + 5]))
            out[i, q // 5] = v
    return out


def decode(keys):
    words = []
    for row in keys:
        syms = []
        for v in row:
            v = int(v)
            for q in range(5):
                s = (v >> (12 * (4 - q))) & 4095
                if s:
                    syms.append(s)
        necks, cur = [], []
        for s in syms:
            if s == SEP:
                necks.append(tuple(cur))
                cur = []
            else:
                cur.append(_unsym(s))
        necks.append(tuple(cur))
        words.append(tuple(necks))
    return words


_G = {}


def _task(args):
    i0, i1 = args
    words, C, N, op = _G['words'], _G['C'], _G['N'], _G['op']
    idx, img, rs, cs, vs = {}, [], [], [], []
    for j in range(i0, i1):
        for m, c in (op(words[j]) if op else q1d.Q1d_multi(words[j], N)).items():
            i = idx.get(m)
            if i is None:
                i = idx[m] = len(img)
                img.append(m)
            rs.append(i)
            cs.append(j - i0)
            vs.append(c)
    K = C.shape[0]
    if not img:
        return np.zeros((0, W), dtype=np.uint64), np.zeros((K, 0), dtype=np.int64)
    M = sp.csr_matrix((np.asarray(vs, dtype=np.int64) % P, (np.asarray(rs, dtype=np.int64), np.asarray(cs, dtype=np.int64))),
                      shape=(len(img), i1 - i0))
    Cc = np.ascontiguousarray(C[:, i0:i1].T) % P                   # n x K
    lo, hi = Cc & 0x1FFF, Cc >> 13
    V = ((M @ lo) % P + ((M @ hi) % P) * (1 << 13)) % P             # len(img) x K
    return encode(img), np.ascontiguousarray(V.T)


def _merge(parts):
    """Sum (keys, V) parts over equal keys: one (unique sorted keys, K x n) pair."""
    ks = np.concatenate([k for k, _ in parts])
    vs = np.concatenate([v for _, v in parts], axis=1)
    if not len(ks):
        return ks, vs
    order = np.lexsort(ks.T[::-1])
    ks, vs = ks[order], vs[:, order]
    new = np.ones(len(ks), dtype=bool)
    new[1:] = (ks[1:] != ks[:-1]).any(axis=1)
    st = np.flatnonzero(new)
    return ks[st], np.add.reduceat(vs % P, st, axis=1) % P


def apply(words, C, N=3, workers=8, chunk=4000, log=None, buffer=30_000_000, op=None):
    """Q_1 (or another linear operator op: word -> {word: coeff}) of the K combinations sum_j C[k, j] words[j]:
    (img, K x len(img) int64), all-zero columns dropped."""
    C = np.asarray(C, dtype=np.int64) % P
    K, n = C.shape
    _G.update(words=words, C=C, N=N, op=op)
    tasks = [(i, min(n, i + chunk)) for i in range(0, n, chunk)]
    glob = (np.zeros((0, W), dtype=np.uint64), np.zeros((K, 0), dtype=np.int64))
    buf, nbuf = [], 0

    def take(part):
        nonlocal glob, buf, nbuf
        buf.append(part)
        nbuf += len(part[0])
        if nbuf >= buffer:
            glob = _merge([glob] + buf)
            buf, nbuf = [], 0
    if workers > 1 and len(tasks) > 1:
        with mp.get_context('fork').Pool(workers) as pool:
            for t, part in enumerate(pool.imap_unordered(_task, tasks)):
                take(part)
                if log and (t + 1) % max(1, len(tasks) // 20) == 0:
                    log(f"  Q_1: {t + 1}/{len(tasks)} word slices; {len(glob[0])} merged image words")
    else:
        for tk in tasks:
            take(_task(tk))
    keys, V = _merge([glob] + buf)
    keep = np.flatnonzero(V.any(axis=0))
    return decode(keys[keep]), np.ascontiguousarray(V[:, keep])
