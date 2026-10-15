import os, sys
REPO = os.environ['REPO']
sys.path[:0] = [f'{REPO}/scripts/oneloop/su-threshold-classes', f'{REPO}/code']
from fastcontents import prime_cache   # writes ./cache/contents_*.pkl
from fortuity.full import sector as S
S.CACHE_DIR = 'cache'
# Fermion-degree split of exact SU(3)_R hw multiplicities in sectors of T-040's all-orders part, to see where
# frame_k's cap k <= 3(N^2-1) = 9 (SU(2)) leaves degrees >= 10 untested. Sector notation (2J1, 2J2, 2q1, 2q2, 2q3).
def m_hw(sec, Y):
    prime_cache(sec, Y)
    return dict(sorted(S.hw_mult_by_degree(2, sec, Y).items()))
for label, sec, Y in [
        ('su2:10:8,7,6 one-loop target', (9, 9, 16, 14, 12), 11),
        ('su2:9:8,6,6 order-2 target T_2', (8, 8, 16, 12, 12), 10),
        ('su2:10:8,7,6 order-2 outbound target', (8, 8, 16, 14, 12), 11)]:
    m = m_hw(sec, Y)
    print(f'{label} {sec} Y={Y}: {m}; degree >= 10: {sum(v for d, v in m.items() if d >= 10)}')
