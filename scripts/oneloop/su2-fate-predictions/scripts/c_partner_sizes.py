import os, sys
REPO = os.environ['REPO']
sys.path[:0] = [f'{REPO}/scripts/oneloop/su-threshold-classes', f'{REPO}/code']
from fastcontents import prime_cache   # writes ./cache/contents_*.pkl
from fortuity.full import sector as S
S.CACHE_DIR = 'cache'
def m_hw(sec, Y):
    prime_cache(sec, Y)
    return S.hw_mult_by_degree(2, sec, Y)    # {fermion degree: exact SU(3)_R hw multiplicity}
def partner_sizes(twoJ, q):
    Y = sum(q) - twoJ
    T = (twoJ - 1, twoJ - 1) + tuple(2 * x for x in q)
    return {n: sum(m_hw(T, Y + 1 - n).values()) for n in range(1, min(q) + 1)}
print(partner_sizes(5, (4, 4, 4)))           # {1: 231, 2: 884, 3: 609, 4: 51}
print(m_hw((8, 8, 16, 12, 12), 11))          # (8,6,6) one-loop target: {..., 8: 244, 10: 3}
print(m_hw((10, 10, 18, 14, 12), 11))    # (9,7,6) one-loop target: degree 10 -> 6
print(m_hw((10, 10, 16, 14, 12), 10))    # (8,7,6) preimage: degree 10 -> 2
