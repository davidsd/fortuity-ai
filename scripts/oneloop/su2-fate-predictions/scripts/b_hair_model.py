import json, os, itertools
from collections import Counter
REPO, JMAX = os.environ['REPO'], 44
pred = Counter()
for nO, r, k in itertools.product(range(2), range(4), range(4)):
    W = Counter()                      # SU(3)_R weights q of O_nO * Q_+^r * w_2^k
    for Qs in itertools.combinations(range(3), r):
        for ws in itertools.combinations_with_replacement(range(3), k):
            W[tuple(4 + 2 * nO + (i in Qs) + k + ws.count(i) for i in range(3))] += 1
    twoJ, Y = 5 + 4 * nO + r + 2 * k, 7 + 2 * nO + 2 * k
    for lam in W:                      # Weyl alternation -> multiplicity of the irrep with highest weight lam
        if not lam[0] >= lam[1] >= lam[2] or 2 * sum(lam) > JMAX:
            continue
        lr = (lam[0] + 1, lam[1], lam[2] - 1)
        mult = sum((-1) ** sum(p[a] > p[b] for a in range(3) for b in range(a + 1, 3)) *
                   W.get(tuple(lr[p[i]] - (1, 0, -1)[i] for i in range(3)), 0) for p in itertools.permutations(range(3)))
        if mult:
            pred[(2 * sum(lam), lam, twoJ, Y)] += mult
            print(f'j={2*sum(lam)} lam={lam} 2J={twoJ} Y={Y} x{mult}  [O{nO} Q+^{r} w2^{k}]')
obs = Counter()
for c in json.load(open(f'{REPO}/data/su2/bmn_hw_cohomology_j44.json'))['columns']:
    for s in c['sectors']:
        if s['BH']:
            obs[(c['j'], tuple(c['lam']), s['twoJ'], sum(c['lam']) - s['twoJ'])] += s['BH']
print('mismatches (pred, obs):', {key: (pred[key], obs[key]) for key in set(pred) | set(obs) if pred[key] != obs[key]})
