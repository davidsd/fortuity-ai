"""Full 1/16-BPS letter sector of N=4 SYM for SU(N): letters with gaugino and derivatives (Chang-Lin superfield),
tree-level Q_0, SU(3)_R highest-weight complex, exact dimensions, the one-loop supercharge Q_1 of Budzik-Kulp, and
block restriction.  Promoted 2026-10-01 from scripts/full-letter/su-checks (T-016); conventions and
validation in that README and in code/tests/test_full.py.

Letters are theta-Taylor coefficients (a, b, S) of the BPS superfield: a, b = numbers of D_1, D_2 derivatives,
S a subset of {0,1,2}; |S| = 0 gaugino (a+b >= 1), 1 scalar, 2 chiralino, 3 field strength; fermionic iff |S| even.
Sector labels are (2J1, 2J2, 2q1, 2q2, 2q3) at fixed letter number Y.

Modules: letters, words (canonical multitraces, Q_0, sector contents), wordenum (fast enumeration), evalpoints and
evalg (random evaluation; evalg.Frame has cartan= and u2block= options), dims (exact dimensions, hw multiplicities),
hw (SU(3)_R raising operators, hw vectors), sector (one hw-complex unit with the rank = m check), cohom (direct
cohomology per sector), q1 (Q_1 on words of derivative-free letters), restrict (SU(N) > SU(N-1) x U(1) monomials).
Disk caching is off unless FORTUITY_FULL_CACHE (or sector.CACHE_DIR) is set.
"""
