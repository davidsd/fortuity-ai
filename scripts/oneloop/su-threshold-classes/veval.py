"""Post-hoc V rows for an existing fullpar prep: evaluate new word vectors at the prep's functionals, so an exactness
test against a stored B needs no new prep.

fullpar's functionals are, for each column chunk ci of fermion degree d, the values of the words at
Frame(N, npts, k, [d], letters, seed + 7919 d + 1000003 ci) with the C-projection (eval_csec), where k = frame_k(N, d) and
letters = sorted letters of the prep's target words in play: the Q_0 images of the preimage words used by its hw rows,
plus the words of its own V rows (--orig-v, if it had any).  The letter list fixes which random matrix each letter gets,
so it is rebuilt exactly; the chunk widths come from the prep's run log ('chunk d=.. cols c0:c1 done'), or, without a
log, one chunk per degree with npts = ceil((m_d + MARGIN) / C(k, d)).  New words with a letter outside the list cannot be
evaluated: they are reported (such a word occurs in no Q_0 image of the hw rows).

usage: python3 veval.py PREP HWCACHE VECS OUTDIR [--log RUNLOG] [--orig-v ctrl.pkl,q1.pkl] [--workers W]
VECS: a pickle dict(words=[...], coef=K x n array, n_ctrl=c), rows 0..c-1 being controls, or 'ctrl.pkl,q1.pkl' (lists
of {word: coeff}).  OUTDIR gets V.i32 (K x ncols), meta.json (the prep's, with nV / n_ctrl) and a symlink to PREP/B.i32."""
import argparse, json, math, os, pickle, re, sys, time
import multiprocessing as mp
import numpy as np
import fullpar
from fullpar import frame_k, eval_csec, Frame, nferm
from fortuity.bmn.modp import P, matmul_mod
try:
    from fortuity.full.words import Q_multi
except ImportError:
    from fortuity.full.words import Q_multi

log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_vecs(spec):
    if spec.endswith('.pkl') and ',' not in spec:
        d = pickle.load(open(spec, 'rb'))
        if isinstance(d, dict):
            return d['words'], np.asarray(d['coef'], dtype=np.int64) % P, int(d.get('n_ctrl', 0))
        spec_l = [spec]
    else:
        spec_l = spec.split(',')
    lists = [pickle.load(open(f, 'rb')) for f in spec_l]
    rows = [r for L in lists for r in L]
    words = sorted({w for r in rows for w in r})
    idx = {w: i for i, w in enumerate(words)}
    C = np.zeros((len(rows), len(words)), dtype=np.int64)
    for k, r in enumerate(rows):
        for w, c in r.items():
            C[k, idx[w]] = c % P
    return words, C, (len(lists[0]) if len(lists) > 1 else 0)


def chunk_layout(meta, logfile, N):
    seed = meta['seed']
    if logfile:
        # one log, or the logs of all column slices (comma-separated): global chunk columns, sorted and deduplicated
        pat = re.compile(r'chunk d=(\d+) cols (\d+):(\d+) done')
        found = sorted({tuple(int(x) for x in m.groups()) for f in logfile.split(',') for m in map(pat.search, open(f)) if m},
                       key=lambda c: c[1])
    else:
        found, col = [], 0
        for d, md in sorted((int(d), m) for d, m in meta['target_m_d'].items()):
            k = frame_k(N, d)
            w = max(1, math.ceil((md + fullpar.MARGIN) / math.comb(k, d))) * math.comb(k, d)
            found.append((d, col, col + w))
            col += w
    chunks, ci = [], {}
    for d, c0, c1 in found:
        k = frame_k(N, d)
        cpp = math.comb(k, d)
        assert (c1 - c0) % cpp == 0, (d, c0, c1, cpp)
        i = ci.get(d, 0)
        ci[d] = i + 1
        chunks.append((d, (c1 - c0) // cpp, k, seed + 7919 * d + 1000003 * i, c0))
    assert sum(np_ * math.comb(k, d) for d, np_, k, _, _ in chunks) == meta['ncols']
    assert [c[4] for c in chunks] == sorted(c[4] for c in chunks)
    for d, md in meta['target_m_d'].items():
        d = int(d)
        k = frame_k(N, d)
        tot = sum(c[1] for c in chunks if c[0] == d)
        assert tot == max(1, math.ceil((md + fullpar.MARGIN) / math.comb(k, d))), (d, tot)
    return chunks


_G = {}


def _task(args):
    ci, i0, i1 = args
    d, npts, k, sd, c0 = _G['chunks'][ci]
    fr = Frame(_G['N'], npts, k, [d], _G['letters'], seed=sd)
    sel = _G['sel'][d][i0:i1]
    words = [_G['words'][j] for j in sel]
    C = _G['C'][:, sel]
    acc = None
    for j0 in range(0, len(words), 20000):
        X = eval_csec(fr, words[j0:j0 + 20000], dtype=np.int64)
        Z = matmul_mod(np.ascontiguousarray(C[:, j0:j0 + 20000]), X)
        acc = Z if acc is None else (acc + Z) % P
    return ci, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prep')
    ap.add_argument('hwcache')
    ap.add_argument('vecs')
    ap.add_argument('outdir')
    ap.add_argument('--log', default=None)
    ap.add_argument('--orig-v', default=None, help="the prep's own V rows (ctrl.pkl,q1.pkl), if it had any")
    ap.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '8')))
    ap.add_argument('--slice-words', type=int, default=200000)
    a = ap.parse_args()
    T0 = time.time()
    meta = json.load(open(os.path.join(a.prep, 'meta.json')))
    N = meta['N']
    fullpar.CSEC = meta.get('csec', 0)
    chunks = chunk_layout(meta, a.log, N)
    log(f"prep {a.prep}: {len(chunks)} chunks, {meta['ncols']} columns, csec {fullpar.CSEC}, seed {meta['seed']}")
    # the target letters: Q_0 images of the used preimage words (+ the prep's own V words)
    ws, groups, ms, stats = pickle.load(open(a.hwcache, 'rb'))
    used = np.unique(np.concatenate([p[0] for g in groups for p in g['parts']]))
    tw = set()
    for i in used:
        tw.update(Q_multi(ws[int(i)]))
    del ws, groups
    if a.orig_v:
        ow, _, _ = load_vecs(a.orig_v)
        tw.update(ow)
    letters = sorted({L for w in tw for n in w for L in n})
    log(f"target words in play {len(tw)} (prep: {meta['target_words']}), {len(letters)} letters ({time.time()-T0:.0f}s)")
    assert len(tw) == meta['target_words'], "target word list not reproduced"
    del tw
    words, C, n_ctrl = load_vecs(a.vecs)
    K = C.shape[0]
    lset = set(letters)
    foreign = np.array([any(L not in lset for n in w for L in n) for w in words], dtype=bool)
    degs = np.array([nferm(w) for w in words], dtype=np.int64)
    cdeg = {c[0] for c in chunks}
    off = ~np.isin(degs, list(cdeg))
    rep = dict(prep=a.prep, vecs=a.vecs, K=K, n_ctrl=n_ctrl, n_words=len(words),
               foreign_letter_words=int(foreign.sum()),
               rows_touching_foreign=[int(np.count_nonzero(C[k, foreign])) for k in range(K)],
               off_degree_words=int(off.sum()))
    log(f"{len(words)} words in {K} rows ({n_ctrl} controls); words with letters outside the prep's list: "
        f"{rep['foreign_letter_words']} (nonzero per row {rep['rows_touching_foreign']}); words of degrees without "
        f"functionals: {rep['off_degree_words']}")
    ok = ~foreign & ~off
    sel = {d: np.flatnonzero(ok & (degs == d)) for d in cdeg}
    _G.update(chunks=chunks, N=N, letters=letters, words=words, C=C, sel=sel)
    tasks = [(ci, i0, min(len(sel[c[0]]), i0 + a.slice_words)) for ci, c in enumerate(chunks)
             for i0 in range(0, len(sel[c[0]]), a.slice_words)]
    V = np.zeros((K, meta['ncols']), dtype=np.int64)
    width = lambda c: c[1] * math.comb(c[2], c[0])
    with mp.get_context('fork').Pool(a.workers) as pool:
        for t, (ci, acc) in enumerate(pool.imap_unordered(_task, tasks)):
            c0 = chunks[ci][4]
            V[:, c0:c0 + width(chunks[ci])] = (V[:, c0:c0 + width(chunks[ci])] + acc) % P
            if (t + 1) % max(1, len(tasks) // 20) == 0:
                log(f"  {t + 1}/{len(tasks)} evaluation tasks ({time.time()-T0:.0f}s)")
    os.makedirs(a.outdir, exist_ok=True)
    V.astype(np.int32).tofile(os.path.join(a.outdir, 'V.i32'))
    bl = os.path.join(a.outdir, 'B.i32')
    if not os.path.exists(bl):
        os.symlink(os.path.abspath(os.path.join(a.prep, 'B.i32')), bl)
    m2 = dict(meta, nV=K, n_ctrl=n_ctrl, n_vecs=K - n_ctrl, prep_dir=a.outdir, veval=rep, time_veval=round(time.time() - T0))
    json.dump(m2, open(os.path.join(a.outdir, 'meta.json'), 'w'), indent=1)
    log(f"wrote {a.outdir}/V.i32 ({K} x {meta['ncols']}) and meta.json ({time.time()-T0:.0f}s)")


if __name__ == '__main__':
    main()
