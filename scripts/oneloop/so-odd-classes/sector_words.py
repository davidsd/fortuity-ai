"""All multitrace words of a full-letter sector at fixed letter number Y and total derivative count D (d_BMN = D/2)."""
from fortuity.full.words import contents_of_sector, multitraces_with_letters


def words(sec, Y, D, fast=True):
    """fast=True: FKM necklaces with fixed content (wordenum_full, same canonical forms); fast=False: the permutation
    enumeration of words_full (slow beyond ~8 letters)."""
    from fortuity.full.wordenum import multitraces_with_content
    out = []
    for c in contents_of_sector(sec, Y):
        if len(c) == Y and sum(L[0] + L[1] for L in c) == D:
            out += multitraces_with_content(c, Y) if fast else multitraces_with_letters(c, Y)
    return out


if __name__ == "__main__":
    import time
    from collections import Counter
    from fortuity.full.evalg import nferm
    for sec, Y in (((1, 1, 6, 6, 6), 7), ((0, 0, 6, 6, 6), 8)):
        t = time.time()
        W = words(sec, Y, 2)
        print(sec, "Y =", Y, ":", len(W), "words; by fermion degree", dict(Counter(nferm(m) for m in W)),
              f"({time.time() - t:.0f} s)", flush=True)
