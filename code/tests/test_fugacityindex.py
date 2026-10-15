"""fortuity.bmn.fugacityindex: fixed-fugacity evaluation reproduces the refined index (R-004 content at N = 3, 4)."""
import cmath
import numpy as np
from fortuity.bmn.fugacityindex import index_at, schur, irrep
from fortuity.bmn.groupindex import bmn_index


def test_matches_refined_and_r004():
    x, y = cmath.exp(0.7j), cmath.exp(2.1j)
    z = (x, 1 / y, y / x)
    # refined expansion of SO(7), summed at (x, y), equals the fixed-fugacity evaluation
    Z, A = bmn_index('SO_odd', 3, 12)
    ref = [sum(Z[j, a + A, b + A] * x ** a * y ** b for a in range(-A, A + 1) for b in range(-A, A + 1)) for j in range(13)]
    assert np.allclose(index_at('SO_odd', 3, 12, x, y, workers=1), ref, atol=1e-8)
    # R-004: SO(7) - Sp(3) = -t^18 (singlet); SO(9) - Sp(4) = -[2,0] t^22, and the conjugate [0,2] does not fit
    d3 = index_at('SO_odd', 3, 18, x, y, workers=1)[18] - index_at('Sp', 3, 18, x, y, workers=1)[18]
    assert abs(d3 + 1) < 1e-8
    d4 = index_at('SO_odd', 4, 22, x, y, workers=1)[22] - index_at('Sp', 4, 22, x, y, workers=1)[22]
    assert abs(d4 + schur(irrep(2, 0), z)) < 1e-8 and abs(d4 + schur(irrep(0, 2), z)) > 1e-3
