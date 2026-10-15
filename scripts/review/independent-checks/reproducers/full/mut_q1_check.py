"""Independent check: do planted gauge-structure defects in fortuity.full.q1 survive the committed
test_q1_anticommutes_with_q0 loop? (copy of the module, textual mutation, same sampling as the test)"""
import importlib.util, random, sys, types
from collections import defaultdict
import fortuity.full.q1 as true_q1
from fortuity.full.words import Q_multi, Q_expr, sectors_at_level
from fortuity.full.wordenum import words_of_sector

src = open(true_q1.__file__).read()
MUTS = {
    'N_to_N+1': [('factor = N\n', 'factor = N + 1\n'), ('factor *= N\n', 'factor *= (N + 1)\n')],
    'rejoin_ADEB': [('new = A + E + D + B', 'new = A + D + E + B')],
    'drop_two_trace': [('        if p != q:\n', '        if p != q:\n            continue\n')],
}

def _bmn(w):
    return all(L[0] == 0 and L[1] == 0 for x in w for L in x)

def committed_test(Q1_multi, Q1_expr):
    random.seed(3)
    for N in (2, 3):
        for n in (8, 10):
            for s in sectors_at_level(n):
                ws = [w for w in words_of_sector(s, 4) if _bmn(w)]
                for w in random.sample(ws, min(2, len(ws))):
                    tot = defaultdict(int)
                    for m, c in Q_expr(Q1_multi(w, N)).items(): tot[m] += c
                    for m, c in Q1_expr(Q_multi(w), N).items(): tot[m] += c
                    if any(tot.values()):
                        return 'FAILS (defect caught)'
    return 'PASSES (defect invisible)'

def differs(Q1m):
    random.seed(7)
    nd = nt = 0
    for n in (8, 10):
        for s in sectors_at_level(n):
            ws = [w for w in words_of_sector(s, 4) if _bmn(w)]
            for w in random.sample(ws, min(3, len(ws))):
                nt += 1
                nd += Q1m(w, 3) != true_q1.Q1_multi(w, 3)
    return f'{nd}/{nt} sampled SU(3) words give a different Q_1'

for name, reps in MUTS.items():
    s = src
    for a, b in reps:
        assert a in s, (name, a)
        s = s.replace(a, b)
    mod = types.ModuleType('fortuity.full.q1_mut', )
    mod.__package__ = 'fortuity.full'
    exec(compile(s, f'q1_{name}.py', 'exec'), mod.__dict__)
    print(f'{name:16s} committed test: {committed_test(mod.Q1_multi, mod.Q1_expr)};  {differs(mod.Q1_multi)}', flush=True)
print('unmutated        committed test:', committed_test(true_q1.Q1_multi, true_q1.Q1_expr))
