"""T-023: sector-level task list for a complete hw scan at one or more levels j, ranked by cost.
Each output line: 'heavy|light lam n cost', where cost = exact hw multiplicity m of the sector (from the SO(7) dims;
the formal words, hence the hw-vector cost, are group independent). The top `--heavy K` sectors go to the heavy queue.
usage (from code/): python ../scripts/bmn/so-usp-g2-scans/make_sector_tasks.py DIMS J [J ...] [--heavy K]"""
import sys, pickle
from fortuity.bmn.words import contents_of_sector
from fortuity.hwcoh.sector import hw_mult_nf

argv = sys.argv[1:]
K = 12
if '--heavy' in argv:                 # remove the option and its value before reading the levels (the value used to be
    i = argv.index('--heavy')         # read as a level: '--heavy 12' added the j=12 sectors to the j=28 task list)
    K = int(argv[i + 1])
    argv = argv[:i] + argv[i + 2:]
args = argv
dims = pickle.load(open(args[0], 'rb'))
js = [int(x) for x in args[1:] if x.isdigit()]
tasks = []
for j in js:
    s = j // 2
    for q1 in range(s, -1, -1):
        for q2 in range(min(q1, s - q1), -1, -1):
            q3 = s - q1 - q2
            if q3 > q2:
                continue
            lam = (q1, q2, q3)
            for n in range(0, 3 * s + 1):
                if not contents_of_sector((n,) + lam):
                    continue
                m = sum(hw_mult_nf(dims, n, lam, f) for f in range(0, n // 2 + 1))
                tasks.append((m, lam, n))
tasks.sort(key=lambda t: -t[0])
for i, (m, lam, n) in enumerate(tasks):
    print('heavy' if i < K else 'light', ','.join(map(str, lam)), n, m)
