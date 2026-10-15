"""fortuity.full.lifting (T-019): the SO(7) class of R-005 is lifted at one loop (reproduces ChoiLee2025Konishi; the
exact space in (0,0;6,6,6), Y=8, d_BMN=1 has rank 741, their number). ~30 s."""
from fortuity.full.lifting import so7_main


def test_so7_class_lifted():
    res = so7_main()
    assert res['lifted'] and res['controls_exact'] and res['stable']
    assert res['ranks'][0] == (826, 741, 741)
