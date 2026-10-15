"""Sizes of the singlet (hw [0,0]) complexes needed for the one-loop test of the SU(3) j=24 and SU(2) j=36 classes."""
import sys, time
from fastcontents import prime_cache, contents_at
from fortuity.full.dims import content_dim
from fortuity.full.sector import hw_mult_by_degree, words_of

def report(N, sec, Y, ml, count_words=True):
    t = time.time()
    prime_cache(sec, Y)
    D = sum(content_dim(N, c) for c in contents_at(sec, Y))
    m = hw_mult_by_degree(N, sec, Y)
    nw = len(words_of(sec, Y, ml)) if count_words else -1
    print(f"N={N} sec={sec} Y={Y}: weight-space dim {D}, singlet hw mult {sum(m.values())} {m}, words(ml={ml}) {nw} ({time.time()-t:.0f}s)", flush=True)

which = sys.argv[1] if len(sys.argv) > 1 else 'both'
if which in ('su3', 'both'):
    for sec, Y in [((3,3,8,8,8), 9), ((4,4,8,8,8), 8), ((2,2,8,8,8), 10), ((2,2,8,8,8), 9), ((3,3,8,8,8), 8)]:
        report(3, sec, Y, 6)
if which in ('su2', 'both'):
    for sec, Y in [((9,9,12,12,12), 9), ((10,10,12,12,12), 8), ((8,8,12,12,12), 10), ((8,8,12,12,12), 9), ((9,9,12,12,12), 8)]:
        report(2, sec, Y, 3)
