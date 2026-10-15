import json, os
REPO = os.environ['REPO']
d = json.load(open(f'{REPO}/data/su2/bmn_hw_cohomology_j44.json'))
for c in d['columns']:
    for s in c['sectors']:
        if s['BH']:
            q, tJ = c['lam'], s['twoJ']
            print(c['j'], tuple(q), c['rep'], f"2J={tJ} Y={sum(q) - tJ} E={s['E']} m={s['m']} h={s['h']} grav={s['grav']} BH={s['BH']}")
    if c['euler'] != c['index']:
        print('incomplete:', c['j'], c['lam'], 'sectors 2J =', [s['twoJ'] for s in c['sectors']], 'euler', c['euler'], 'index', c['index'])
