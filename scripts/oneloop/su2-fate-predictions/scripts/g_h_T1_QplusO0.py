import os, sys
REPO = os.environ['REPO']
sys.path[:0] = [f'{REPO}/scripts/oneloop/su-threshold-classes', f'{REPO}/code']
from fastcontents import prime_cache
from fortuity.full import sector as S
S.CACHE_DIR = 'cache'
sec, Y = (5, 5, 10, 8, 8), 7
up, low = (sec[0] + 1, sec[1] + 1) + sec[2:], (sec[0] - 1, sec[1] - 1) + sec[2:]
for s, y in [(sec, Y), (up, Y - 1), (low, Y + 1)]:
    prime_cache(s, y)
quiet = lambda *a: None
rT = S.Sector(2, sec, Y, 3, log=quiet).run()      # m(T) and r_up = rank of Q_0 into T  (= r_in)
rL = S.Sector(2, low, Y + 1, 3, log=quiet).run()  # r_up of the image sector = rank of Q_0 out of T (= r_out)
print(f"m={rT['m']} r_in={rT['r_up']} r_out={rL['r_up']} h={rT['m'] - rT['r_up'] - rL['r_up']} checks {rT.get('check')} {rL.get('check')}")
