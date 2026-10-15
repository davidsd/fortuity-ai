import os, sys
sys.path.insert(0, os.environ['REPO'] + '/code')
from fortuity.full.dims import sector_dim
# Exact dimension (Weyl integration) of the SU(2) gauge-invariant space with Y = 4 in the sector (3,3;8,4,4), where
# {w_2,w_2} lives. e_L2_inputs.py finds that the evaluated target words have rank 95 at three seeds; if the exact
# dimension is also 95, the evaluation is injective on that space and the exactness verdict on {w_2,w_2} holds over F_p
# with certified functionals (not just observed rank stability).
print('exact dim V_T, SU(2), (3,3,8,4,4), Y=4:', sector_dim(2, (3, 3, 8, 4, 4), 4))
