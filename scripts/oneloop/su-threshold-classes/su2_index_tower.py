"""SU(2) threshold class O_0 and the index (summary/fortuity-summary.pdf).

All sectors (5-k, 5-k; 8,8,8), k an integer ("level"), have the same charges under the Cartan of the commutant
psu(1,2|3) (J_R and q_j = R_j + J_L), so they contribute to the same coefficient of the index, with sign (-1)^k.
This script counts, for every level and every letter number Y, the lowest-weight vectors of O_0's weight (14
conditions: E_1, E_2, D, K_+-, S_a, T_{al,a}) in the one-species abelian module.  Coulomb-type classes inject into
this module, so a zero means: no Coulomb-type lowest-weight class of that weight in that sector.  (At odd Y the
Cartan image of every SU(2) operator vanishes, so odd-Y rows are not images of classes.)
usage: python3 su2_index_tower.py   -> runs/su2_O0_index_tower.txt"""
import sys, time
sys.argv = ['x']
import numpy as np
import abelian_lw as A
import su2_lw as S
from fortuity.hwcoh import ffla
t0 = time.time()
for k in (1, -1, 3, -3, 5, -5, 7, 0, 2, -2, 4, -4):
    sec = (5 - k, 5 - k, 8, 8, 8)
    for Y in range(1, 15):
        basis = S.monomials1(sec, Y)
        if not basis:
            continue
        M = S.stacked([{m: 1} for m in basis], A.OPS)
        lw = len(basis) - int(ffla.rank(M))
        print(f"level {k:+d} sector {sec} Y={Y:2d}: monomials {len(basis):6d}  LW vectors of O_0's weight {lw}   ({time.time()-t0:.0f}s)", flush=True)
