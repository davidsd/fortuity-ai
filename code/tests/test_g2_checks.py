"""R-010's checks (fortuity.bmn.g2_checks): the SO(7) class of eq. (4.14) of arXiv:2506.13887 is closed on so(7), not on
so(9), non-graviton at SO(7), and restricts to a nonzero non-exact graviton class of G2; trace words span the G2
invariants at low j."""
from fortuity.bmn.g2_checks import so7_class, vanishes, classify, trace_word_completeness
from fortuity.bmn.words import Q_expr


def test_so7_class_closed_on_so7_not_so9():
    QO = Q_expr(so7_class())
    assert vanishes(QO, 'SO_odd', 3, 0) and not vanishes(QO, 'SO_odd', 4, 0)


def test_so7_class_restricts_to_g2_graviton():
    O = so7_class()
    assert classify(O, 'SO_odd', 3) == (False, False, False)        # non-graviton at SO(7)
    assert classify(O, 'G2', 0) == (False, False, True)             # nonzero, non-exact, graviton in G2


def test_g2_trace_word_completeness_low_j():
    n, bad = trace_word_completeness('G2', 0, 10)
    assert n > 0 and not bad
