"""Moved to code/ (T-048, 2026-10-13): the group-aware sector is fortuity.full.sector.Sector(..., group=), the fast
content enumeration is fortuity.full.words.level_contents / contents_of_sector (same contents, sorted), the group
multiplicities are fortuity.full.sector.hw_mult_by_degree_group. Forwarding shim with the old names."""
from fortuity.full.sector import Sector, frame_k as _fk, hw_mult_by_degree_group as hw_mult_by_degree  # noqa: F401
from fortuity.full.words import level_contents, contents_of_sector as fast_contents_of_sector  # noqa: F401


def frame_k(group, N, d, **kw):
    return _fk(N, d, group=group, **kw)


class SectorG(Sector):
    def __init__(self, group, N, sec, Y, maxlen, **kw):
        super().__init__(N, sec, Y, maxlen, group=group, **kw)
