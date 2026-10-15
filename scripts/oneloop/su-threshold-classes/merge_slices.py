"""Reassemble a fullpar prep computed in column slices (fullpar.py --col-slice i/K --prep-dir PREP/slice_<i>of<K>) into
PREP/B.i32, PREP/V.i32 and PREP/meta.json, exactly as an unsliced prep would have written them (gpu_rank.py input).
Checks that the slices agree on every global quantity and cover each column once.
usage: python3 merge_slices.py PREP K [--blk 4096]"""
import argparse, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('prep')
ap.add_argument('K', type=int)
ap.add_argument('--blk', type=int, default=4096)
a = ap.parse_args()
t0 = time.time()
dirs = [os.path.join(a.prep, f'slice_{i}of{a.K}') for i in range(a.K)]
metas = [json.load(open(os.path.join(d, 'meta.json'))) for d in dirs]
m0 = metas[0]
for k in ('N', 'SEC', 'Y', 'maxlen', 'seed', 'p', 'case', 'csec', 'ncols', 'ncart', 'nB', 'nV', 'n_ctrl', 'n_vecs',
          'tp_m', 'tp_rows', 'target_words', 'q0_nnz'):
    vals = {json.dumps(m.get(k)) for m in metas}
    assert len(vals) == 1, (k, vals)
for i, m in enumerate(metas):
    assert m['col_slice'] == [i, a.K], (i, m['col_slice'])
nB, nV, ncols, ncart = m0['nB'], m0['nV'], m0['ncols'], m0['ncart']
cover = np.zeros(ncols + ncart, dtype=np.int8)
for m in metas:
    for _, c0, w, _ in m['slice_chunks'] + m['slice_cchunks']:
        cover[c0:c0 + w] += 1
assert (cover == 1).all(), f"columns covered {np.bincount(cover)} times"
ws = [m['ncols_local'] + m['ncart_local'] for m in metas]
Bs = [np.memmap(os.path.join(d, 'B.i32'), dtype=np.int32, mode='r', shape=(max(nB, 1), w)) for d, w in zip(dirs, ws)]
Vs = [np.memmap(os.path.join(d, 'V.i32'), dtype=np.int32, mode='r', shape=(max(nV, 1), max(m['ncols_local'], 1)))
      for d, m in zip(dirs, metas)]


def assemble(srcs, nrows, width, path, with_cart):
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        for r0 in range(0, max(nrows, 1), a.blk):
            r1 = min(max(nrows, 1), r0 + a.blk)
            out = np.zeros((r1 - r0, width), dtype=np.int32)
            for S, m in zip(srcs, metas):
                blk = np.asarray(S[r0:r1])
                for _, c0, w, l0 in m['slice_chunks'] + (m['slice_cchunks'] if with_cart else []):
                    out[:, c0:c0 + w] = blk[:, l0:l0 + w]
            out.tofile(fh)
    assert os.path.getsize(tmp) == 4 * max(nrows, 1) * width
    os.replace(tmp, path)


assemble(Bs, nB, ncols + ncart, os.path.join(a.prep, 'B.i32'), True)
print(f"B {nB} x {ncols + ncart} assembled ({time.time() - t0:.0f}s)", flush=True)
assemble(Vs, nV, max(ncols, 1), os.path.join(a.prep, 'V.i32'), False)
meta = {k: v for k, v in m0.items() if k not in ('col_slice', 'ncols_local', 'ncart_local', 'slice_chunks',
                                                    'slice_cchunks')}
meta.update(prep_dir=a.prep, merged_slices=a.K, timing_eval=sum(m['timing_eval'] for m in metas),
            timing_combine=sum(m['timing_combine'] for m in metas), time=max(m['time'] for m in metas),
            slice_times=[m['time'] for m in metas])
json.dump(meta, open(os.path.join(a.prep, 'meta.json'), 'w'), indent=1)
print(f"merged {a.K} slices into {a.prep} ({time.time() - t0:.0f}s)", flush=True)
