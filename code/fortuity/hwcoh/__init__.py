"""Q-cohomology of the BMN sector restricted to SU(3)_R highest-weight vectors (2026-09-26).

Per sector (2J, lam): hw vectors of the free multitrace space (per shape), random evaluation with one Grassmann
frame per fermion degree, exact hw multiplicities from an n_f-refined Haar integral (self-check rank = m), ranks mod p
(FLINT if built, else numpy), graviton-algebra hw vectors.  Output per SU(3)_R irrep: cohomology, graviton and
non-graviton multiplicities.  Validated against fortuity.bmn.symdirect (tests/test_hwcoh.py) and the exact index.
Drivers: run_sectors (restartable JSONL), assemble (+ --index Euler-characteristic check), make_tasks.
"""
