"""Write the T-025 task files: every dominant column (q1 >= q2 >= q3 >= 0, j = 2(q1+q2+q3)) up to the stated j.
Heavy columns get one array task each (highest j and near-singlet first); light columns are bundled six per task."""


def cols(s):
    return [(q1, q2, s - q1 - q2) for q1 in range(s, -1, -1) for q2 in range(min(q1, s - q1), -1, -1)
            if 0 <= s - q1 - q2 <= q2]


def lines(N, dims, gen, smin, smax, bundle_upto):
    G = 2 * N + 1
    heavy = sorted([(s, c) for s in range(max(smin, bundle_upto + 1), smax + 1) for c in cols(s)],
                   key=lambda t: (-t[0], -t[1][2], -t[1][1]))
    out = [f"{N} {dims} {gen} so{G}_c_{'_'.join(map(str, c))}.jsonl {','.join(map(str, c))}" for _, c in heavy]
    light = [c for s in range(smin, min(smax, bundle_upto) + 1) for c in cols(s)]
    for i in range(0, len(light), 6):
        out.append(f"{N} {dims} {gen} so{G}_light_{i // 6}.jsonl " + ' '.join(','.join(map(str, c)) for c in light[i:i + 6]))
    return out


if __name__ == '__main__':
    so9 = ['4 dims_so9_j22.pkl 10 so9_ctrl_5_3_3_j22.jsonl 5,3,3'] + lines(4, 'dims_so9_j22.pkl', 10, 1, 10, 7)
    sets = {'tasks_so9.txt': so9,
            'tasks_so11_big.txt': lines(5, 'dims_so11_j24.pkl', 12, 12, 12, 0),
            'tasks_so11.txt': lines(5, 'dims_so11_j24.pkl', 12, 1, 11, 8),
            # added 2026-09-30: SO(7) below its first index difference (j = 18), plus the whole j = 18 level, whose
            # column (3,3,3) holds the R-005 class (positive control)
            'tasks_so7.txt': lines(3, 'dims_so7_j18.pkl', 10, 1, 9, 9)}
    for fn, L in sets.items():
        open(fn, 'w').write('\n'.join(L) + '\n')
        print(fn, len(L), 'tasks,', sum(len(x.split()) - 4 for x in L), 'columns')
