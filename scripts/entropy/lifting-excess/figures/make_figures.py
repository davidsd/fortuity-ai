#!/usr/bin/env python3
"""T-041 figures and derived numbers.  Reads saved outputs only: no saddle point, Legendre transform or holonomy
integrand is recomputed here.  The derived numbers quoted in the T-041 note, summarised in summary/fortuity-summary.pdf sec. 7.6 (extremes, monotonicity,
exponents, ratios, cross-checks) that are not printed verbatim in an output file are printed by this script into
numbers.txt, next to the file they were read from.

Usage:   python3 make_figures.py            (from any directory; about 1 s)
Needs:   Python 3 with numpy and matplotlib (run with Python 3.14.7, numpy 2.4.4, matplotlib 3.10.9).

Inputs (paths relative to scripts/entropy/lifting-excess/; all computed on the Perimeter cluster 'symmetry'):
  outputs/main/scan24_fine.tsv, scan12_patched.tsv        job 568138  code/rf/scan_kklp_charges.py (patched)
  outputs/main/torus_<group>_allext.out, conv_*.out        job 568138  code/cardy_torus_unequal.py (patched)
  outputs/extended/ray0.tsv ... ray12.tsv                  job 568139  physics/scan_extended.py
  outputs/indep/indep_grid12.tsv                           job 568141  indep/indep_delta_r1.py --grid 12 (chunks merged
                                                                       with the command in run_indep_grid.sbatch)
  outputs/indep/indep_wedge.tsv                            job 568140  indep/indep_delta_r1.py --file points_wedge.txt
  outputs/scan12_clean.tsv, outputs/torus_unequal_*.out   job 568105  the unpatched T-038 scripts (first run); optional:
                                                                       if absent, the parts of sections A, F and I that
                                                                       use them are skipped
  audit/t10_rim_along_rays.out                             local       audit test (rim height and theta* along rays)
  outputs/provenance.txt                                   symmetry    Slurm accounting record (optional)
Outputs (this folder):
  fig1_delta0_map.png        delta0 over the black-hole domain sum_I phi_I^2 < 1 in charge-fraction space
  fig2_r1_map.png            r1 (coefficient of eps) over the same domain
  fig3_delta0_rays_edge.png  delta0 along five rays, and along the edge Q3 = 0
  numbers.txt                derived numbers (extremes, monotonicity, edge and corner values, cross-checks,
                             holonomy-dominance summary), each with its source file
  compare_indep_vs_scan12_unpatched.out, compare_indep_vs_scan12_patched.out
                             output of indep/compare_scan.py (file comparison only)

Geometry.  Charge fractions phi_I = Q_I/sum Q lie in the plane sum phi = 1.  With c = (1/3,1/3,1/3),
e1 = (2,-1,-1)/sqrt6 and e2 = (0,1,-1)/sqrt2, the plane coordinates are X = (phi-c).e1, Y = (phi-c).e2.  Since
phi - c = xi/(3 cbar), the polar angle in this plane is the scan angle theta of xi = r (cos theta e1 + sin theta e2).
The black-hole domain sigma2(Q) > 0 is the disk |phi - c| < sqrt(2/3); rho = |phi - c|/sqrt(2/3) is 0 at equal
charges, 1/2 at the edge midpoints, 1 on the boundary circle (the corners lie on it).  The scans cover the ordered
wedge theta in [0, 60 deg] (xi1 >= xi2 >= xi3); S3 (permutations of the three charges) fills the rest.  The plots are
rotated by 90 degrees so that the corner phi = (1,0,0) is at the top.
"""
import contextlib
import importlib.util
import io
import itertools
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import matplotlib.tri as mtri                        # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Patch, Circle          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T041 = os.path.dirname(HERE)
OUT = os.path.join(T041, "outputs")
P = lambda *a: os.path.join(OUT, *a)                 # noqa: E731
REL = lambda p: os.path.relpath(p, T041)             # noqa: E731

# ---------------------------------------------------------------- style (light chart surface, recessive chrome)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NODATA = "#ecebe7"                                   # neutral gray: inside the domain, outside the computed hull
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]  # sequential blue, steps 100..700 (one hue)
ORANGE = ["#ffe9e2", "#fccbba", "#f3af97", "#e69476", "#d87957", "#c85f37", "#b64512", "#9a3400", "#7b2901"]  # one hue
# (OKLCH hue of the palette orange #eb6834, lightness 0.95 -> 0.40; monotone lightness and single hue validated)
RAYS = {0: "#86b6ef", 15: "#5598e7", 30: "#2a78d6", 45: "#1c5cab", 60: "#0d366b"}  # ordinal ramp (validated)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.titlesize": 10.5, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "mathtext.fontset": "dejavusans"})

R_DISK = math.sqrt(2 / 3)
GROUPS = ["su2", "su3", "su4", "g2", "so7", "usp6"]
# the 18 holonomy-dominance points, as charge fractions: the nine presets (audit/t13_presets_coverage.out), then the
# nine extra points (comments of EXTRA in code/cardy_torus_unequal.py), in the order of the torus outputs
HOLO = {"eq": (1 / 3, 1 / 3, 1 / 3), "r1a": (0.488, 0.256, 0.256), "r1b": (0.629, 0.185, 0.185),
        "r1c": (0.744, 0.128, 0.128), "r1d": (0.828, 0.086, 0.086), "r2a": (0.536, 0.232, 0.232),
        "r2b": (0.707, 0.147, 0.147), "r3a": (0.410, 0.410, 0.179), "r3b": (0.483, 0.483, 0.035),
        "r1e": (0.877, 0.062, 0.062), "r1f": (0.936, 0.032, 0.032), "r3c": (0.499, 0.499, 0.003),
        "r4a": (0.507, 0.333, 0.159), "r4b": (0.659, 0.333, 0.008), "r5a": (0.894, 0.095, 0.010),
        "r6a": (0.615, 0.258, 0.127), "r6b": (0.781, 0.213, 0.006), "r7a": (0.566, 0.419, 0.015)}


# ---------------------------------------------------------------- reading
def read_tsv(path):
    """Rows of a tab-separated output as dicts of floats; comment lines and non-numeric lines are skipped."""
    header, rows, comments = None, [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                comments.append(line)
                continue
            tok = line.split("\t")
            if header is None:
                header = tok
                continue
            if len(tok) != len(header):
                comments.append("SKIPPED: " + line)
                continue
            try:
                rows.append({k: float(v) for k, v in zip(header, tok)})
            except ValueError:
                comments.append("SKIPPED: " + line)
    return header, rows, comments


def fr(row):
    return row["Q1/sQ"], row["Q2/sQ"], row["Q3/sQ"]


def rho_of(f):
    s = f[0] ** 2 + f[1] ** 2 + f[2] ** 2
    return math.sqrt(max(0.0, (s - 1 / 3) / (2 / 3)))


def plane(f):
    """Display coordinates (u, v): corner (1,0,0) at the top, (0,1,0) lower left, (0,0,1) lower right."""
    X = (2 * f[0] - f[1] - f[2]) / math.sqrt(6)
    Y = (f[1] - f[2]) / math.sqrt(2)
    return -Y, X


def by_ray(rows):
    d = defaultdict(list)
    for r in rows:
        d[round(r["theta_deg"], 6)].append(r)
    for k in d:
        d[k].sort(key=lambda r: r["r"])
    return dict(sorted(d.items()))


def unique_points(rows):
    """Drop repeated r = 0 rows (the scans print the equal-charge point once per ray)."""
    seen, out = set(), []
    for r in rows:
        key = (0.0, 0.0) if r["r"] == 0 else (round(r["theta_deg"], 6), round(r["r"], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_extended():
    rows, ends = [], {}
    for k in range(13):
        path = P("extended", "ray%d.tsv" % k)
        _, rr, com = read_tsv(path)
        for r in rr:
            r["ray"] = k
        rows += rr
        ends[k] = [c for c in com if c.startswith("# ray")]
    return rows, ends


# ---------------------------------------------------------------- holonomy-dominance outputs
def parse_torus(path):
    """Patched cardy_torus_unequal.py output: per point the max competitor of each criterion, the verdict, the rim."""
    pts, cur = {}, None
    for line in open(path):
        m = re.match(r"== point (\S+)", line)
        if m:
            cur = {"name": m.group(1)}
            pts[cur["name"]] = cur
            continue
        if cur is None:
            continue
        m = re.match(r"(\S+) xi = \(([^)]*)\): Phi\*/pi = ([\d.]+), .*\(delta0 = ([\d.]+)\); grid (\d+) pts", line)
        if m:
            cur.update(group=m.group(1), xi=m.group(2), phi=float(m.group(3)), d0=float(m.group(4)), grid=int(m.group(5)))
            continue
        m = re.match(r"\s+(protected|classical) (R1|R2) .*max over other holonomies (\(none[^)]*\)|-?[\d.]+) -> (.*)$", line)
        if m:
            side, crit, val, verdict = m.groups()
            rim = re.search(r"rim max 1 \+ ([\d.e+-]+)", verdict)
            cur[(side, crit)] = (None if val.startswith("(") else float(val), verdict.split(";")[0].strip(),
                                 float(rim.group(1)) if rim else None)
            continue
        m = re.search(r"theta\* = exp\(-B-1/2\) = ([\d.e+-]+) \(B = ([\d.]+)\)", line)
        if m:
            cur["thetastar"] = float(m.group(1))
            continue
        m = re.match(r"\s+time ([\d.]+) s", line)
        if m:
            cur["time"] = float(m.group(1))
    return pts


def parse_torus_unpatched(path):
    """Unpatched output (job 568105): verdict lines and every listed value, per point and criterion."""
    pts, cur, crit = {}, None, None
    for line in open(path):
        m = re.match(r"== point (\S+)", line)
        if m:
            cur = {"name": m.group(1), "vals": defaultdict(list), "verdict": {}}
            pts[cur["name"]] = cur
            continue
        if cur is None:
            continue
        m = re.match(r"\s+(protected|classical) (R1|R2) .*: max ([\d.]+) -> (.*)$", line)
        if m:
            crit = (m.group(1), m.group(2))
            cur["verdict"][crit] = (float(m.group(3)), m.group(4).strip())
            continue
        m = re.match(r"\s+(-?[\d.]+) at theta/pi = \[([^\]]*)\]", line)
        if m and crit:
            cur["vals"][crit].append((float(m.group(1)), m.group(2).split()))
    return pts


# ---------------------------------------------------------------- figures
BOX = dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.88)


def domain_artists(ax):
    """Boundary circle, triangle Q_I = 0, corner and edge labels, the computed wedge."""
    t = np.linspace(0, 2 * np.pi, 721)
    ax.plot(R_DISK * np.cos(t), R_DISK * np.sin(t), color=INK2, lw=1.1, zorder=6)
    corners = [plane((1, 0, 0)), plane((0, 1, 0)), plane((0, 0, 1))]
    tri = np.array(corners + [corners[0]])
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=0.8, zorder=6)
    for (u, v), lab in zip(corners, [r"$\varphi=(1,0,0)$", r"$(0,1,0)$", r"$(0,0,1)$"]):
        ax.annotate(lab, (u, v), xytext=(u * 1.13, v * 1.10 + (0.012 if v > 0 else -0.012)), ha="center",
                    va="bottom" if v > 0 else "top", color=INK2, fontsize=9)
    # edge labels just inside each circular segment, segment labels further out
    for f, l1, l2 in [((0.5, 0.5, 0.0), r"$Q_3=0$", r"$Q_3<0$"), ((0.5, 0.0, 0.5), r"$Q_2=0$", r"$Q_2<0$"),
                      ((0.0, 0.5, 0.5), r"$Q_1=0$", r"$Q_1<0$")]:
        u, v = plane(f)
        ax.text(u * 1.22, v * 1.22, l1, ha="center", va="center", color=INK, fontsize=8, zorder=8, bbox=BOX)
        ax.text(u * 1.68, v * 1.68, l2, ha="center", va="center", color=INK, fontsize=8, zorder=8, bbox=BOX)
    # computed wedge theta in [0, 60 deg]: display angles 90..150 deg
    for a in (90, 150):
        ax.plot([0, R_DISK * math.cos(math.radians(a))], [0, R_DISK * math.sin(math.radians(a))], color=INK,
                lw=0.5, alpha=0.6, zorder=5, solid_capstyle="butt")
    a = math.radians(124)
    ax.annotate("computed wedge\n" + r"$\theta=0$" + "–" + r"$60^\circ$;" + "\nthe rest by\n" + r"$S_3$ symmetry",
                (R_DISK * math.cos(a), R_DISK * math.sin(a)), xytext=(-1.00, 0.74), ha="center", va="center",
                fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=1))


def label_positions(tri, Z, levels, angles):
    """Points where Z crosses each level along rays from the center at the given display angles (degrees)."""
    interp = mtri.LinearTriInterpolator(tri, Z)
    out = []
    for lev, ang in zip(levels, itertools.cycle(angles)):
        s = np.linspace(0.0, R_DISK * 0.995, 2000)
        x, y = s * math.cos(math.radians(ang)), s * math.sin(math.radians(ang))
        z = np.ma.filled(interp(x, y), np.nan)
        for i in range(len(s) - 1):
            if np.isfinite(z[i]) and np.isfinite(z[i + 1]) and (z[i] - lev) * (z[i + 1] - lev) <= 0:
                out.append((x[i], y[i]))
                break
    return out


def map_points(scan24, ext, key):
    """(u, v, value) over the whole disk: the wedge data and its S3 images, deduplicated."""
    base = [r for r in unique_points(scan24)]
    base += [r for r in ext if r["ray"] != 0 or r["r"] > 0.795]   # ray 0 beyond scan24's last theta = 0 point
    U, V, Z, seen = [], [], [], set()
    for r in base:
        f = fr(r)
        for perm in itertools.permutations(range(3)):
            g = (f[perm[0]], f[perm[1]], f[perm[2]])
            u, v = plane(g)
            k = (round(u, 9), round(v, 9))
            if k in seen:
                continue
            seen.add(k)
            U.append(u)
            V.append(v)
            Z.append(r[key])
    return np.array(U), np.array(V), np.array(Z), len(base)


def draw_map(fname, scan24, ext, key, band_colors, levels_lines, fmt, cbar_label, title, center_text,
             holonomy_pts=None, label_angles=(-7, 7)):
    """Filled bands between the contour levels (one-hue sequential ramp, light = small), surface-colored hairlines at
    the band edges, labels placed where the contours cross two rays near the theta = 30 deg direction."""
    U, V, Z, nbase = map_points(scan24, ext, key)
    tri = mtri.Triangulation(U, V)
    levels = [0.0] + list(levels_lines) + [float(Z.max()) * (1 + 1e-9)]
    assert len(band_colors) == len(levels) - 1
    fig, ax = plt.subplots(figsize=(7.6, 6.9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Circle((0, 0), R_DISK, facecolor=NODATA, edgecolor="none", zorder=0))
    cf = ax.tricontourf(tri, Z, levels=levels, colors=band_colors, zorder=1)
    cs = ax.tricontour(tri, Z, levels=levels_lines, colors=[SURFACE], linewidths=1.0, zorder=4)
    pos = label_positions(tri, Z, levels_lines, angles=label_angles)
    labels = ax.clabel(cs, levels_lines, fmt=fmt, fontsize=7.5, inline=False, manual=pos)
    for lab in labels:
        lab.set_color(INK)
        lab.set_rotation(0)
        lab.set_bbox(dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.9))
        lab.set_zorder(7)
    domain_artists(ax)
    ax.plot([0], [0], "o", ms=4.5, color=INK, mec=SURFACE, mew=1.0, zorder=9)
    ax.annotate(center_text, (0, 0), xytext=(-0.42, -0.10), fontsize=8.5, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=1, shrinkB=3), zorder=9, bbox=BOX)
    handles = [Patch(facecolor=NODATA, edgecolor="none",
                     label="outside the hull of the computed nodes (" + key_name(key) + r"$\to0$ there is extrapolated)")]
    if holonomy_pts:
        hu, hv = zip(*[plane(f) for f in holonomy_pts])
        ax.plot(hu, hv, "o", ms=4.4, mfc=SURFACE, mec=INK, mew=0.9, zorder=9, ls="none")
        handles.append(Line2D([], [], marker="o", ms=4.4, mfc=SURFACE, mec=INK, mew=0.9, ls="none",
                              label=r"holonomy-dominance test points (18, all in $Q_I\geq0$; 6 groups)"))
    handles.append(Line2D([], [], color=INK2, lw=1.1, label=r"boundary of the domain, $\sigma_2(Q)=0$"))
    handles.append(Line2D([], [], color=INK, lw=0.8, label=r"triangle $Q_I\geq0$ (the T-038 domain)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, fontsize=8,
              handlelength=1.6, columnspacing=1.6, labelcolor=INK2)
    cb = fig.colorbar(cf, ax=ax, shrink=0.60, pad=0.0, aspect=26, spacing="proportional")
    cb.set_label(cbar_label, color=INK2)
    cb.set_ticks([0.0] + list(levels_lines))
    cb.ax.set_yticklabels([fmt % v for v in [0.0] + list(levels_lines)])
    cb.outline.set_edgecolor(AXIS)
    cb.ax.tick_params(color=MUTED, labelcolor=INK2, labelsize=8)
    ax.set_xlim(-1.16, 1.0)
    ax.set_ylim(-0.84, 1.0)
    ax.set_title(title, loc="left", pad=4)
    fig.savefig(os.path.join(HERE, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return len(U), nbase, tri.triangles.shape[0]


def key_name(key):
    return r"$\delta_0$" if key == "delta0" else r"$r_1$"


def draw_rays_edge(fname, scan24, ext):
    s24 = by_ray(unique_points(scan24))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.22))
    for ax in (axl, axr):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(length=0)
    rays_used = {}
    center = s24[0.0][0]                                 # the equal-charge point (r = 0), shared by every ray
    for th, col in RAYS.items():
        rows = list(s24[float(th)])
        if rows[0]["r"] != 0:
            rows = [center] + rows
        k = th // 5
        rows += [r for r in ext if r["ray"] == k and (k != 0 or r["r"] > 0.795)]
        rows.sort(key=lambda r: r["r"])
        x = [rho_of(fr(r)) for r in rows]
        y = [r["delta0"] for r in rows]
        lab = {0: r"$\theta=0^\circ$ (toward a corner)", 60: r"$\theta=60^\circ$ (toward an edge midpoint)"}.get(
            th, r"$\theta=%d^\circ$" % th)
        axl.plot(x, y, color=col, lw=2.0, solid_joinstyle="round", solid_capstyle="round", label=lab, zorder=3)
        rays_used[th] = (len(rows), x[-1], y[-1])
        if th:
            e = s24[float(th)][-1]                       # scan24 point at r_edge (1 - 1e-3): the edge Q3 = 0
            axl.plot([rho_of(fr(e))], [e["delta0"]], "o", ms=6.5, mfc=SURFACE, mec=col, mew=1.6, zorder=5)
    axl.axvline(1.0, color=INK2, lw=0.9, zorder=2)
    axl.text(0.985, 0.0116, "domain boundary\n" + r"$\sigma_2(Q)=0$", ha="right", va="top", fontsize=8, color=INK2)
    axl.plot([], [], "o", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.6, ls="none",
             label=r"crosses the edge $Q_3=0$ (beyond: $Q_3<0$)")
    axl.set_xlim(0, 1.02)
    axl.set_ylim(0, 0.0128)
    axl.set_xlabel(r"$\rho=|\varphi-(\frac{1}{3},\frac{1}{3},\frac{1}{3})|\,/\sqrt{2/3}$   (0: equal charges, 1: boundary)")
    axl.set_ylabel(r"$\delta_0$")
    axl.legend(loc="lower left", fontsize=8, handlelength=1.8, labelcolor=INK2)
    axl.set_title(r"(a) $\delta_0$ along five rays from equal charges", loc="left")
    # right: the edge Q3 = 0, one point per ray theta = 2.5 .. 60 deg (scan24 at r_edge (1 - 1e-3))
    edge = [s24[th][-1] for th in s24 if th > 0]
    edge.sort(key=lambda r: r["Q1/sQ"])
    x = [r["Q1/sQ"] for r in edge]
    y = [r["delta0"] for r in edge]
    col = BLUE[8]
    axr.plot(x, y, color=col, lw=2.0, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    axr.plot(x, y, "o", ms=3.2, color=col, mec=col, zorder=4, ls="none")
    axr.annotate("%.4f at the midpoint\n(0.4999, 0.4999, 0.0003)" % y[0], (x[0], y[0]), xytext=(0.535, 0.0116),
                 fontsize=8, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=3))
    axr.annotate("%.4f at\n(%.3f, %.3f, %.4f)" % (y[-1], edge[-1]["Q1/sQ"], edge[-1]["Q2/sQ"], edge[-1]["Q3/sQ"]),
                 (x[-1], y[-1]), xytext=(0.70, 0.0022), fontsize=8, color=INK, va="center",
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkB=3))
    axr.axvline(1.0, color=INK2, lw=0.9, zorder=2)
    axr.text(0.993, 0.0118, "corner (1, 0, 0):\n" + r"$\delta_0\to0$ (extrapolated)", ha="right", va="top", fontsize=8,
             color=INK2)
    axr.set_xlim(0.48, 1.01)
    axr.set_ylim(0, 0.0128)
    axr.set_xlabel(r"$\varphi_1=Q_1/\sum Q$ on the edge ($\varphi_2=1-\varphi_1$, $\varphi_3=0$)")
    axr.set_ylabel(r"$\delta_0$")
    axr.set_title(r"(b) $\delta_0$ along the edge $Q_3=0$ (24 points)", loc="left")
    fig.savefig(os.path.join(HERE, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rays_used, edge


# ---------------------------------------------------------------- numbers
class Log:
    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        self.lines.append(s)

    def sec(self, title):
        self.lines += ["", "=" * 100, title, "=" * 100]


def monotone(rows, key, decreasing=True):
    v = [r[key] for r in rows]
    bad = [(rows[i]["r"], v[i], v[i + 1]) for i in range(len(v) - 1) if (v[i + 1] >= v[i] if decreasing else v[i + 1] <= v[i])]
    return bad


def run_compare(scan, indep, outname):
    sys.dont_write_bytecode = True                  # keep indep/ free of __pycache__
    spec = importlib.util.spec_from_file_location("compare_scan", os.path.join(T041, "indep", "compare_scan.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main(scan, indep)
    text = buf.getvalue()
    with open(os.path.join(HERE, outname), "w") as fh:
        fh.write("# python indep/compare_scan.py %s %s   (run by figures/make_figures.py)\n" % (REL(scan), REL(indep)))
        fh.write(text)
    rows = [l.split("\t") for l in text.splitlines()[1:] if l.endswith("\tok") or l.endswith("\tMISMATCH")]
    hdr = text.splitlines()[0].split("\t")
    return text.splitlines()[-1], rows, hdr


def main():
    L = Log()
    L("T-041 derived numbers, written by figures/make_figures.py (reads saved outputs only).")
    L("Paths are relative to scripts/entropy/lifting-excess/. Job IDs: 568105 (first run, unpatched T-038 scripts), 568138 (main,")
    L("patched code/), 568139 (extended domain), 568140 (independent points), 568141 (independent grid), 568296 (independent")
    L("check of the segments); all on 'symmetry'.")

    f24, f12, f12u = P("main", "scan24_fine.tsv"), P("main", "scan12_patched.tsv"), P("scan12_clean.tsv")
    _, s24, c24 = read_tsv(f24)
    _, s12, c12 = read_tsv(f12)
    # job 568105 (first, unpatched run): optional, since no saved job script regenerates it
    have_u = os.path.exists(f12u) and all(os.path.exists(P("torus_unequal_%s.out" % g)) for g in GROUPS)
    s12u, c12u = (read_tsv(f12u)[1:]) if have_u else ([], [])
    ext, ext_ends = load_extended()
    _, ig, _ = read_tsv(P("indep", "indep_grid12.tsv"))
    _, iw, _ = read_tsv(P("indep", "indep_wedge.tsv"))

    # ------------------------------------------------------------ A. inventory
    L.sec("A. Inventory")
    if not have_u:
        L("outputs/scan12_clean.tsv or outputs/torus_unequal_<group>.out (job 568105) not found: the parts of sections A, F"
          " and I that use them are skipped")
    for name, rows, com in ((REL(f24), s24, c24), (REL(f12), s12, c12)) + (((REL(f12u), s12u, c12u),) if have_u else ()):
        rays = by_ray(rows)
        L("%-32s %4d data rows, %4d unique points (r = 0 printed once per ray), %2d rays; per ray: %s"
          % (name, len(rows), len(unique_points(rows)), len(rays), " ".join("%g:%d" % (k, len(v)) for k, v in rays.items())))
        for c in com:
            L("    comment/skipped line: " + c[:150])
    L("outputs/extended/ray0..12.tsv: %d rows; per ray (theta = 5k deg): %s" % (
        len(ext), " ".join("%d:%d" % (k, sum(1 for r in ext if r["ray"] == k)) for k in range(13))))
    for k in range(13):
        L("    ray%-2d %s" % (k, " | ".join(ext_ends[k])))
    L("    segment points (one charge negative, min Q/sQ < 0): %d" % sum(1 for r in ext if min(fr(r)) < 0))
    L("outputs/indep/indep_grid12.tsv: %d points; outputs/indep/indep_wedge.tsv: %d points" % (len(ig), len(iw)))
    prov = P("provenance.txt")
    if os.path.exists(prov):
        jobs = [l.split() for l in open(prov) if re.match(r"568\d\d\d\s", l)]
        secs = [sum(int(x) * w for x, w in zip(j[7].split(":"), (3600, 60, 1))) for j in jobs]
        L("Slurm wall times (outputs/provenance.txt, sacct): %s; total %d min %d s"
          % (", ".join("%s %s %s" % (j[0], j[1], j[7]) for j in jobs), sum(secs) // 60, sum(secs) % 60))

    # ------------------------------------------------------------ B. delta0 and r1: extremes, monotonicity
    L.sec("B. delta0 and r1 on the scanned triangle (scan24_fine, scan12_patched) and in the segments (extended)")
    for name, rows in ((REL(f24), s24), (REL(f12), s12)):
        u = unique_points(rows)
        mx = max(u, key=lambda r: r["delta0"])
        mn = min(u, key=lambda r: r["delta0"])
        L("%s: delta0 max %.9f at theta=%g r=%g fractions (%.6f, %.6f, %.6f); min %.9f at theta=%g r=%g (%.6f, %.6f, %.6f)"
          % (name, mx["delta0"], mx["theta_deg"], mx["r"], *fr(mx), mn["delta0"], mn["theta_deg"], mn["r"], *fr(mn)))
        rmx = max(u, key=lambda r: r["r1"])
        rmn = min(u, key=lambda r: r["r1"])
        L("%s: r1 max %.7f at r=%g; min %.7f at theta=%g r=%g (%.6f, %.6f, %.6f); all delta0 > 0: %s; all r1 > 0: %s"
          % (name, rmx["r1"], rmx["r"], rmn["r1"], rmn["theta_deg"], rmn["r"], *fr(rmn),
             all(r["delta0"] > 0 for r in u), all(r["r1"] > 0 for r in u)))
        zero = {r["delta0"] for r in rows if r["r"] == 0}
        L("    delta0 at r = 0 on every ray: %s" % sorted(zero))
        rays = by_ray(rows)
        bad_d = {k: monotone(v, "delta0") for k, v in rays.items()}
        bad_r = {k: monotone(v, "r1") for k, v in rays.items()}
        L("    delta0 strictly decreasing in r on every ray: %s; r1 strictly decreasing on every ray: %s"
          % (all(not b for b in bad_d.values()), all(not b for b in bad_r.values())))
        for k in rays:
            if bad_d[k] or bad_r[k]:
                L("      ray %g: delta0 violations %s; r1 violations %s" % (k, bad_d[k][:3], bad_r[k][:3]))
        bad_rho = [k for k, v in rays.items() if any(rho_of(fr(v[i + 1])) <= rho_of(fr(v[i])) for i in range(len(v) - 1))]
        L("    rho increasing with r on every ray: %s" % (not bad_rho))
    # extended segments, including the junction with the last (edge) point of scan24
    s24r = by_ray(unique_points(s24))
    allok_d, allok_r = True, True
    for k in range(13):
        rows = [r for r in ext if r["ray"] == k]
        if k:
            rows = [s24r[float(5 * k)][-1]] + rows   # junction: scan24 edge point, then the segment
        bd, br = monotone(rows, "delta0"), monotone(rows, "r1")
        allok_d &= not bd
        allok_r &= not br
        if bd or br:
            L("    extended ray %d: delta0 violations %s; r1 violations %s" % (k, bd[:3], br[:3]))
    L("outputs/extended: delta0 strictly decreasing along every ray (rays 1-12 joined to the scan24 edge point): %s; r1: %s"
      % (allok_d, allok_r))
    seg = [r for r in ext if min(fr(r)) < 0]
    L("outputs/extended segment points: delta0 in [%.3g, %.6f], r1 in [%.4g, %.5f]; all > 0: %s, %s"
      % (min(r["delta0"] for r in seg), max(r["delta0"] for r in seg), min(r["r1"] for r in seg), max(r["r1"] for r in seg),
         all(r["delta0"] > 0 for r in seg), all(r["r1"] > 0 for r in seg)))
    allpts = unique_points(s24) + [r for r in ext if r["ray"] != 0 or r["r"] > 0.795]
    L("all map points (scan24 + extended): %d; max delta0 %.9f (at r = 0); max r1 %.7f; min delta0 %.3g; min r1 %.4g"
      % (len(allpts), max(r["delta0"] for r in allpts), max(r["r1"] for r in allpts), min(r["delta0"] for r in allpts),
         min(r["r1"] for r in allpts)))
    eps_x = s24[0]["delta0"] / s24[0]["r1"]
    L("first-order crossover eps_x = delta0/r1 (where r1*eps equals delta0): at r = 0: %.4f" % eps_x)
    L("    the same as j/dim G at j1 = j2 = j, with eps = (dim G/j)^(1/3): j = %.1f dim G" % (1 / eps_x ** 3))

    L("Selected rows of %s (table of note sec. 3.1):" % REL(f24))
    L("    theta  r      fractions                       rho     Sprot/K       Scl/K         Phi/pi    delta0       r1")
    for th, r0 in ((0, 0.0), (0, 0.1), (0, 0.2), (0, 0.3), (0, 0.4), (0, 0.5), (0, 0.6), (0, 0.79), (30, 0.1), (30, 0.2),
                   (60, 0.08), (60, 0.16)):
        x = [r for r in s24 if r["theta_deg"] == th and abs(r["r"] - r0) < 1e-9][0]
        L("    %4g  %.2f   (%.6f, %.6f, %.6f)  %.4f  %.10f  %.10f  %.6f  %.9f  %.7f"
          % (th, r0, *fr(x), rho_of(fr(x)), x["Sprot/K"], x["Scl/K"], x["Phi/pi"], x["delta0"], x["r1"]))

    # ------------------------------------------------------------ C. the edge Q3 = 0
    L.sec("C. The edge Q3 = 0: last point of each bounded ray (r_edge (1 - 1e-3), or a grid point that is closer)")
    for name, rows in ((REL(f24), s24), (REL(f12), s12)):
        rays = by_ray(unique_points(rows))
        L(name)
        L("    theta    r (last)         fractions                         delta0       r1")
        for k, v in rays.items():
            if k == 0:
                continue
            e = v[-1]
            L("    %5.1f   %.5f          (%.6f, %.6f, %.6f)    %.9f  %.7f" % (k, e["r"], *fr(e), e["delta0"], e["r1"]))
    edge24 = sorted([v[-1] for k, v in by_ray(unique_points(s24)).items() if k > 0], key=lambda r: r["Q1/sQ"])
    mid = edge24[0]["delta0"]
    near = [r for r in edge24 if r["delta0"] >= 0.95 * mid]
    nxt = edge24[len(near)]
    L("edge points of scan24 with delta0 within 5%% of the midpoint value (>= %.6f): phi1 <= %.4f (theta >= %g deg);"
      " the next one, phi1 = %.4f (theta = %g deg), has %.6f" % (0.95 * mid, near[-1]["Q1/sQ"], near[-1]["theta_deg"],
                                                              nxt["Q1/sQ"], nxt["theta_deg"], nxt["delta0"]))

    # ------------------------------------------------------------ D. toward the corner on theta = 0
    L.sec("D. theta = 0 ray toward the corner (1, 0, 0)")
    th0 = [r for r in unique_points(s24) if r["theta_deg"] == 0]
    for r in (th0[-1],):
        L("%s: last point r=%g fractions (%.6f, %.6f, %.6f) delta0 %.9f r1 %.7f Phi/pi %.6f"
          % (REL(f24), r["r"], *fr(r), r["delta0"], r["r1"], r["Phi/pi"]))
    e0 = [r for r in ext if r["ray"] == 0][-1]
    L("outputs/extended/ray0.tsv: last point r=%g fractions (%.7f, %.7f, %.7f) delta0 %.7g r1 %.7g Phi/pi %.6f"
      % (e0["r"], *fr(e0), e0["delta0"], e0["r1"], e0["Phi/pi"]))
    g0 = [r for r in ig if r["theta_deg"] == 0]
    L("outputs/indep/indep_grid12.tsv theta = 0, r = %g: fractions (%.8f, %.8f, %.8f) delta0 %.6g r1 %.6g"
      % (g0[-1]["r"], *fr(g0[-1]), g0[-1]["delta0"], g0[-1]["r1"]))
    L("outputs/indep/indep_wedge.tsv theta = 0 (independent code, job 568140):")
    w0 = sorted([r for r in iw if r["theta_deg"] == 0 and r["r"] > 0], key=lambda r: r["r"])
    for r in w0:
        L("    r=%-4g small fraction %.6f  delta0 %.6g  r1 %.6g  delta0/r1 %.4f  Phi/pi %.6f"
          % (r["r"], r["Q2/sQ"], r["delta0"], r["r1"], r["delta0"] / r["r1"], r["Phi/pi"]))
    for a, b in zip(w0, w0[1:]):
        L("    local exponent d log delta0 / d log(small fraction), r %g -> %g: %.3f"
          % (a["r"], b["r"], math.log(b["delta0"] / a["delta0"]) / math.log(b["Q2/sQ"] / a["Q2/sQ"])))
    # finer steps: indep_grid12 at r = 0.3, 0.4, ..., 1.0 on theta = 0, then indep_wedge at r = 1.5
    g0r = {round(r["r"], 6): r for r in g0}
    seq = [g0r[round(0.1 * k, 6)] for k in range(3, 11)] + [r for r in w0 if abs(r["r"] - 1.5) < 1e-9]
    L("theta = 0 in steps of 0.1 (indep_grid12, r = 0.3 ... 1.0) and r = 1.5 (indep_wedge): Phi* - 2 pi, the angular")
    L("distance cut_cl of the classical u1 = exp(-m1) from the Li3 cut (1, inf) (= |arg u1|), and 2 delta0 Sprot/K / (Phi*-2pi)^2:")
    for r in seq:
        dphi = (r["Phi/pi"] - 2) * math.pi
        L("    r=%-4g small fraction %.6f  delta0 %.5g  Phi*-2pi %.5f  Sprot/K %.5f  cut_cl %.5f  2 delta0 Sprot/(Phi*-2pi)^2 %.3f"
          % (r["r"], r["Q2/sQ"], r["delta0"], dphi, r["Sprot/K"], r["cut_cl"], 2 * r["delta0"] * r["Sprot/K"] / dphi ** 2))
    for a, b in zip(seq, seq[1:]):
        lf = math.log(b["Q2/sQ"] / a["Q2/sQ"])
        L("    r %.1f -> %.1f (small fraction %.4f -> %.4f): local exponent of delta0 %.3f, of Phi*-2pi %.3f"
          % (a["r"], b["r"], a["Q2/sQ"], b["Q2/sQ"], math.log(b["delta0"] / a["delta0"]) / lf,
             math.log((b["Phi/pi"] - 2) / (a["Phi/pi"] - 2)) / lf))
    # where cut_cl falls below the patched torus script's rim-exclusion radius 0.1 (linear interpolation in r)
    for a, b in zip(seq, seq[1:]):
        if a["cut_cl"] >= 0.1 > b["cut_cl"]:
            rc = a["r"] + (a["cut_cl"] - 0.1) / (a["cut_cl"] - b["cut_cl"]) * (b["r"] - a["r"])
            L("    cut_cl = 0.1 (the rim-exclusion radius of code/cardy_torus_unequal.py) at r = %.2f on theta = 0" % rc)
    t10 = [l.split() for l in open(os.path.join(T041, "audit", "t10_rim_along_rays.out")) if re.match(r"\s+0\s+0\.800\s", l)]
    if t10:
        ts, h10 = float(t10[0][8]), float(t10[0][9])
        c08 = g0r[0.8]["cut_cl"]
        L("audit/t10_rim_along_rays.out, theta = 0, r = 0.8 (0.982, 0.009, 0.009): theta* %.3g, rim (SU(2), R1) %.2g;"
          " cut_cl there %.4f = %.1f theta*" % (ts, h10, c08, c08 / ts))

    # ------------------------------------------------------------ E. toward the arc sigma2 = 0
    L.sec("E. The circular segments (one charge negative): last point of each extended ray (outputs/extended/rayK.tsv)")
    L("    ray theta  r     fractions                          sigma2/sQ^2  Sprot/K    delta0      r1        Phi/pi")
    for k in range(1, 13):
        r = [x for x in ext if x["ray"] == k][-1]
        L("    %2d  %4.0f  %.2f  (%.5f, %.5f, %.5f)   %.3e    %.5f   %.4e  %.4e  %.6f"
          % (k, r["theta_deg"], r["r"], *fr(r), r["sigma2/sQ^2"], r["Sprot/K"], r["delta0"], r["r1"], r["Phi/pi"]))
    L("    rho of these last points: %s" % ", ".join("%.4f" % rho_of(fr([x for x in ext if x["ray"] == k][-1])) for k in range(1, 13)))
    L("Selected segment points (for the note):")
    for k, rr in ((6, (0.3, 0.4, 0.5)), (12, (0.2, 0.3, 0.4)), (1, (0.5, 1.0, 1.58))):
        for x in ext:
            if x["ray"] == k and any(abs(x["r"] - q) < 1e-9 for q in rr):
                L("    theta=%g r=%g (%.4f, %.4f, %.4f) rho %.4f delta0 %.6g r1 %.5g Phi/pi %.6f Scl/(K S0) %.4f minRe_m_prot %.4f"
                  % (x["theta_deg"], x["r"], *fr(x), rho_of(fr(x)), x["delta0"], x["r1"], x["Phi/pi"], x["Scl/(K*S0)"],
                     x["minRe_m_prot"]))
    unst = [x for x in ext if x["r"] > 0 and not (x["minRe_m_prot"] < 0 and x["minRe_m_cl"] < 0)]
    L("extended rows with r > 0 where min Re m_I >= 0 at either saddle: %d of %d (convergence flag, the T-041 physics notes, summarised in summary/fortuity-summary.pdf sec. 7.6)"
      % (len(unst), sum(1 for x in ext if x["r"] > 0)))
    ee = [x["eps_stable_est"] for x in ext if x["r"] > 0]
    L("eps_stable_est (first-order threshold for the stable phase) over the extended rows: %.3f ... %.2f" % (min(ee), max(ee)))
    for x in [x for x in ext if x["ray"] == 0 and 0 < x["r"] < 0.05]:
        L("    theta=0 r=%g (%.3f, %.3f, %.3f): eps_stable_est %.4f = %.2f r" % (x["r"], *fr(x), x["eps_stable_est"],
                                                                           x["eps_stable_est"] / x["r"]))
    smin = min(seg, key=lambda x: x["eps_stable_est"])
    smax = max(seg, key=lambda x: x["eps_stable_est"])
    L("    over the %d segment points: min %.4f (theta=%g r=%g, (%.4f, %.4f, %.4f)), max %.2f (theta=%g r=%g, last point)"
      % (len(seg), smin["eps_stable_est"], smin["theta_deg"], smin["r"], *fr(smin), smax["eps_stable_est"],
         smax["theta_deg"], smax["r"]))
    L("delta0/r1 (the eps at which r1*eps equals delta0) at the last point of each segment ray: " + ", ".join(
        "ray %d %.2g" % (k, [x for x in ext if x["ray"] == k][-1]["delta0"] / [x for x in ext if x["ray"] == k][-1]["r1"])
        for k in range(1, 13)))

    # ------------------------------------------------------------ F. cross-checks
    L.sec("F. Cross-checks between independent codes")
    s24_0 = {round(r["r"], 6): r for r in th0}
    e_0 = [r for r in ext if r["ray"] == 0]
    common = [(s24_0[round(r["r"], 6)], r) for r in e_0 if round(r["r"], 6) in s24_0]
    L("extended ray0 (physics/classical_max.py) vs scan24_fine theta = 0 (patched t038 code): %d common r; max|d delta0| %.2e,"
      " max|d r1| %.2e, max|d Phi/pi| %.2e" % (len(common), max(abs(a["delta0"] - b["delta0"]) for a, b in common),
                                               max(abs(a["r1"] - b["r1"]) for a, b in common),
                                               max(abs(a["Phi/pi"] - b["Phi/pi"]) for a, b in common)))
    ig0 = {round(r["r"], 6): r for r in g0}
    common = [(ig0[round(r["r"], 6)], r) for r in e_0 if round(r["r"], 6) in ig0]
    L("extended ray0 vs indep_grid12 theta = 0 (indep/indep_delta_r1.py): %d common r (to r = 0.9); max|d delta0| %.2e,"
      " max|d r1| %.2e, max|d Phi/pi| %.2e" % (len(common), max(abs(a["delta0"] - b["delta0"]) for a, b in common),
                                               max(abs(a["r1"] - b["r1"]) for a, b in common),
                                               max(abs(a["Phi/pi"] - b["Phi/pi"]) for a, b in common)))
    sel = [l.strip() for l in open(P("extended", "selftest.out")) if l.strip()]
    L("outputs/extended/selftest.out: " + " | ".join(sel[1:3]))
    p30 = [r for r in s12 if r["theta_deg"] == 30 and abs(r["r"] - 0.2) < 1e-9][0]
    L("    scan12_patched theta=30 r=0.2: delta0 %.9f r1 %.7f Phi/pi %.6f" % (p30["delta0"], p30["r1"], p30["Phi/pi"]))
    pairs = (((f12u, "compare_indep_vs_scan12_unpatched.out"),) if have_u else ()) + \
        ((f12, "compare_indep_vs_scan12_patched.out"),)
    for scan, out in pairs:
        last, rows, hdr = run_compare(scan, P("indep", "indep_grid12.tsv"), out)
        L("indep/compare_scan.py %s outputs/indep/indep_grid12.tsv -> figures/%s: '%s'" % (REL(scan), out, last))
        ok = [r for r in rows if r[-1] == "ok"]
        L("    max |difference| over the 'ok' rows: " + ", ".join(
            "%s %.1e" % (hdr[j], max(abs(float(r[j])) for r in ok)) for j in range(2, 10)))
        bad = [r for r in rows if r[-1] == "MISMATCH"]
        if bad:
            cols = sorted({hdr[j] for r in bad for j in range(2, 10) if abs(float(r[j])) > 6e-6})
            L("    MISMATCH rows: %d, all at theta = %s, r = %s; columns above 6e-6: %s"
              % (len(bad), sorted({r[0] for r in bad}), ", ".join(r[1] for r in bad), cols))
    # internal checks printed by the independent code (columns of indep_grid12.tsv and indep_wedge.tsv)
    for name, rows in (("outputs/indep/indep_grid12.tsv", ig), ("outputs/indep/indep_wedge.tsv", iw)):
        def mx(col, rr=rows):
            return max(abs(r[col]) for r in rr)
        far = [r for r in rows if abs(r["r1-r1_env"]) > 1e-10]
        L("%s (%d points): max|r1 - r1_env| %.1e (finite-eps Legendre slope vs envelope formula); max|r1_os - r1_env| %.1e;"
          " max|r1cub - r1| %.1e; max|dB - dA| %.1e; max ncross %d; max BH0 %.1e, cubic %.1e, BHfin %.1e"
          % (name, len(rows), mx("r1-r1_env"), mx("r1_os-r1_env"), mx("r1cub-r1"), mx("dB-dA"), int(mx("ncross")),
             mx("BH0"), mx("cubic"), mx("BHfin")))
        L("    points with |r1 - r1_env| > 1e-10: %s" % ", ".join(
            "theta=%g r=%g (%.1e, r1 %.4g)" % (r["theta_deg"], r["r"], r["r1-r1_env"], r["r1"]) for r in far))
        L("    d2ReS/dbeta2 (maximum over the bonus charge iff < 0): max %.3f; Phi* - 2 pi k: min %.4f"
          % (max(r["d2ReS/dbeta2"] for r in rows), min(r["|Phi-2pi k|"] for r in rows)))
    # the conjugate twin: Phi_unpatched + Phi_continued = 4 pi on those rows
    if have_u:
        igm = {(r["theta_deg"], round(r["r"], 6)): r for r in ig}
        tw = []
        for r in s12u:
            q = igm.get((r["theta_deg"], round(r["r"], 6)))
            if q and abs(r["Phi/pi"] - q["Phi/pi"]) > 6e-6:
                tw.append((r["r"], r["Phi/pi"], q["Phi/pi"], r["Phi/pi"] + q["Phi/pi"]))
        L("unpatched scan12 (job 568105) rows whose Phi/pi differs from the independent grid: %d; (r, Phi_u/pi, Phi_indep/pi,"
          " sum):" % len(tw))
        for t in tw:
            L("    r=%.2f  %.5f  %.6f  sum %.5f" % t)
    # where the cross-checks lie: the grid comparisons are in the triangle; the segments are checked by job 568296
    L("smallest charge fraction among the points compared: indep_grid12 %.6f, indep_wedge %.6f, scan12_patched %.6f;"
      " extended segment points (physics/classical_max.py): %d" % (
          min(min(fr(r)) for r in ig), min(min(fr(r)) for r in iw), min(min(fr(r)) for r in s12), len(seg)))
    cseg = P("indep_seg", "compare_segments.out")
    if os.path.exists(cseg):
        last = [l.rstrip("\n") for l in open(cseg) if l.strip()][-1]
        L("independent check of the segment points (job 568296; indep/compare_segments.py -> outputs/indep_seg/"
          "compare_segments.out): '%s'" % last)

    # ------------------------------------------------------------ G. holonomy dominance
    L.sec("G. Holonomy dominance (patched code/cardy_torus_unequal.py, job 568138): outputs/main/torus_<group>_allext.out")
    groups = ["su2", "su3", "su4", "g2", "so7", "usp6"]
    T = {g: parse_torus(P("main", "torus_%s_allext.out" % g)) for g in groups}
    crits = [("protected", "R1"), ("protected", "R2"), ("classical", "R1"), ("classical", "R2")]
    names = list(T["su2"].keys())
    L("points (%d): %s" % (len(names), " ".join(names)))
    verdicts = defaultdict(int)
    for g in groups:
        for n, c in T[g].items():
            for cr in crits:
                verdicts[c[cr][1]] += 1
    L("verdict lines over 6 groups x %d points x 4 criteria: %s" % (len(names), dict(verdicts)))
    L("per group and criterion: [smallest, largest] over the points of the largest competing local maximum, and at how many")
    L("points one was found (none = the refinement found no other local maximum at any point)")
    for g in groups:
        parts = []
        for cr in crits:
            vals = [c[cr][0] for c in T[g].values() if c[cr][0] is not None]
            parts.append("%s %s: %s" % (cr[0][:4], cr[1], ("[%.4f, %.4f] (%d pts)" % (min(vals), max(vals), len(vals)))
                                         if vals else "none"))
        grids = sorted({c["grid"] for c in T[g].values()})
        tt = sum(c.get("time", 0) for c in T[g].values())
        L("    %-5s grid %s pts | %s | sum of per-point times %.0f s" % (g, grids, " | ".join(parts), tt))
    L("per point: largest competitor over the 6 groups and 4 criteria; classical rim (R1, R2) range over groups; theta*")
    per_point_max = {}
    for n in names:
        best = max(((c[cr][0], g, cr) for g in groups for cr in crits for c in [T[g][n]] if c[cr][0] is not None))
        per_point_max[n] = best[0]
        rims1 = [T[g][n][("classical", "R1")][2] for g in groups]
        rims2 = [T[g][n][("classical", "R2")][2] for g in groups]
        L("    %-4s xi=(%s) delta0=%.6f | max competitor %.4f (%s, %s %s) | rim R1 %.2e-%.2e, R2 %.2e-%.2e | theta* %.2e"
          % (n, T["su2"][n]["xi"], T["su2"][n]["d0"], best[0], best[1], best[2][0], best[2][1], min(rims1), max(rims1),
             min(rims2), max(rims2), T["su2"][n]["thetastar"]))
    L("range over points of the largest competitor: %.4f (%s) to %.4f (%s)" % (
        min(per_point_max.values()), min(per_point_max, key=per_point_max.get),
        max(per_point_max.values()), max(per_point_max, key=per_point_max.get)))
    allr = [T[g][n][("classical", cr)][2] for g in groups for n in names for cr in ("R1", "R2")]
    L("classical rim heights over all groups, points, R1 and R2: %.2e to %.2e" % (min(allr), max(allr)))
    L("rim relative to delta0 (SU(2), R1): " + ", ".join(
        "%s %.1e (%.2g%%)" % (n, T["su2"][n][("classical", "R1")][2], 100 * T["su2"][n][("classical", "R1")][2] / T["su2"][n]["d0"])
        for n in names))
    L("Phi*/pi and delta0 identical in all six groups at every point: %s" % all(
        T[g][n]["phi"] == T["su2"][n]["phi"] and T[g][n]["d0"] == T["su2"][n]["d0"] for g in groups for n in names))
    # torus points that lie on the scan24 grid: delta0 (printed to 6 decimals there) against scan24_fine
    e1 = np.array([2, -1, -1]) / math.sqrt(6)
    e2 = np.array([0, 1, -1]) / math.sqrt(2)
    # the xi actually used (PRESETS and EXTRA of code/cardy_torus_unequal.py; the output prints them to 5 digits)
    src = open(os.path.join(T041, "code", "cardy_torus_unequal.py")).read()
    blk = src[src.index("PRESETS = {"):src.index("def run(")]
    xi_used = {m.group(1): [float(v) for v in m.group(2).split(",")]
               for m in re.finditer(r"'(\w+)': \(([^)]*)\)", blk)}
    agree = []
    for n in names:
        xi = np.array(xi_used.get(n) or [float(v) for v in T["su2"][n]["xi"].split(",")])
        rr = float(np.linalg.norm(xi))
        th = math.degrees(math.atan2(xi @ e2, xi @ e1)) if rr > 0 else 0.0
        cand = [r for r in unique_points(s24) if (rr == 0 and r["r"] == 0) or
                (abs(r["theta_deg"] - th) < 0.01 and abs(r["r"] - rr) < 2e-4)]
        hit = cand[0] if cand else None
        if hit is not None:
            t = math.radians(hit["theta_deg"])
            xg = hit["r"] * (math.cos(t) * e1 + math.sin(t) * e2)
            agree.append((n, th, rr, T["su2"][n]["d0"], hit["delta0"], float(np.max(np.abs(xi - xg))),
                          T["su2"][n]["phi"], hit["Phi/pi"]))
    L("torus points on the scan24 grid (theta within 0.01 deg, r within 2e-4): delta0 in the torus output (6 decimals) vs scan24_fine:")
    for n, th, rr, a, b, dx, pa, pb in agree:
        L("    %-4s theta=%5.2f r=%.4f: torus %.6f, scan24 %.9f, |diff| %.1e; max|xi_used - xi_grid| %.1e;"
          " Phi*/pi torus %.4f, scan24 %.6f" % (n, th, rr, a, b, abs(a - b), dx, pa, pb))
    L("    over these %d points: max|d delta0| %.1e, max|xi_used - xi_grid| %.1e (xi_used: PRESETS/EXTRA of"
      " code/cardy_torus_unequal.py), Phi*/pi equal at the 4 printed decimals: %s"
      % (len(agree), max(abs(a[3] - a[4]) for a in agree), max(a[5] for a in agree),
         all(abs(round(a[7], 4) - a[6]) < 1e-9 for a in agree)))
    # where the 18 points lie, and how close the nearest competitor comes
    L("smallest charge fraction among the 18 points: %.3f (%s); all in the triangle Q_I >= 0: %s" % (
        min(min(f) for f in HOLO.values()), min(HOLO, key=lambda k: min(HOLO[k])), all(min(f) >= 0 for f in HOLO.values())))
    L("SU(2) classical R2, largest competitor toward the edge midpoint (theta = 60): " + ", ".join(
        "%s %.4f" % (n, T["su2"][n][("classical", "R2")][0]) for n in ("eq", "r3a", "r3b", "r3c")))
    big = max(per_point_max.values())
    L("smallest margin over all points, groups and criteria: 1 - %.4f = %.3f" % (big, 1 - big))
    # secondary bonus maxima at the identity (or a central element) in the classical R2 scans: delta/(3 pi) away from 0
    sec2 = []
    for g in GROUPS:
        txt = open(P("main", "torus_%s_allext.out" % g)).read()
        for m in re.finditer(r"== point (\S+)\n(.*?)(?=== point|\Z)", txt, re.S):
            block = m.group(2).split("classical R2", 1)[1]
            for v, d in re.findall(r"\n\s+([\d.]+) at theta/pi = \[[^\]]*\], delta/\(3 pi\) = (-?[\d.]+)\s+\[identity/central",
                                   block):
                if abs(float(d)) > 0.01 and float(v) < 1:
                    sec2.append((float(v), m.group(1), g, float(d)))
    top = sorted({(v, n, d) for v, n, g, d in sec2}, reverse=True)       # the same in every group
    L("classical R2: secondary bonus maxima at the identity (|delta/(3 pi)| > 0.01), in all six groups; largest: %s"
      % ", ".join("%.4f (%s, delta/(3 pi) = %.4f)" % t for t in top[:4]))
    # the rim against delta0 further toward the corner (audit/t10 at theta = 0, r = 0.8; delta0 from indep_grid12)
    if t10:
        d08 = g0r[0.8]["delta0"]
        L("rim/delta0 at theta = 0, r = 0.8 (0.982, 0.009, 0.009): %.2g / %.4g = %.3f (audit/t10, indep_grid12)"
          % (h10, d08, h10 / d08))
    # envelope factor: a relative rise h of Re log Z at the identity's potentials shifts S_cl by h Re log Z_id, and at the
    # Cardy saddle Re log Z_id = K Re g_cl / 3 with Re g_cl = Scl/K - Re sum m xi; on theta = 0, sum Re m = 0 gives
    # Re sum m xi = (3 r / sqrt 6) Re m1, and Re m1 = minRe_m_cl (outputs/extended/ray0.tsv)
    L("envelope factor Re log Z_id / S_prot (shift of delta0 = rim height x factor) on theta = 0:")
    fac = {}
    for x in [r for r in ext if r["ray"] == 0 and any(abs(r["r"] - q) < 1e-9 for q in (0.0, 0.5, 0.8))]:
        reg = x["Scl/K"] - 3 * x["r"] / math.sqrt(6) * x["minRe_m_cl"]
        fac[round(x["r"], 6)] = reg / 3 / x["Sprot/K"]
        L("    r=%-3g (%.3f, %.3f, %.3f): Re g_cl %.4f, Sprot/K %.4f, factor %.3f" % (x["r"], *fr(x), reg, x["Sprot/K"],
                                                                            fac[round(x["r"], 6)]))
    h1f = T["su2"]["r1f"][("classical", "R1")][2]
    r1f = [r for r in s24 if r["theta_deg"] == 0 and abs(r["r"] - 0.5) < 1e-9][0]
    L("    r1f (r = 0.5): rim %.2g, rim/delta0 %.4f, x factor %.4f; rim/r1 = %.2g (the eps above which r1*eps exceeds"
      " the rim)" % (h1f, h1f / r1f["delta0"], h1f / r1f["delta0"] * fac[0.5], h1f / r1f["r1"]))
    if t10:
        L("    r = 0.8: rim/delta0 %.3f, x factor %.3f" % (h10 / d08, h10 / d08 * fac[0.8]))
    L("    cross-check: the R2 rim (Legendre transform redone, normalized to the identity's S) over the R1 rim, SU(2): " +
      ", ".join("%s %.2e/%.2e = %.3f" % (n, T["su2"][n][("classical", "R2")][2], T["su2"][n][("classical", "R1")][2],
                                         T["su2"][n][("classical", "R2")][2] / T["su2"][n][("classical", "R1")][2])
                for n in ("eq", "r1d", "r1e", "r1f")))
    # where the largest SU(2) competitor sits
    txt = open(P("main", "torus_su2_allext.out")).read()
    locs = []
    for m in re.finditer(r"== point (\S+)\n(.*?)(?=== point|\Z)", txt, re.S):
        n, block = m.group(1), m.group(2)
        for crit in ("protected R2", "classical R2"):
            sec = block.split(crit, 1)[1]
            mx = re.search(r"max over other holonomies ([\d.]+)", sec).group(1)
            loc = re.search(r"\n\s+" + re.escape(mx) + r" at theta/pi = \[([^\]]*)\]", sec).group(1)
            locs.append((n, crit, float(mx), float(loc)))
    L("SU(2): theta/pi of the largest competitor (protected R2, classical R2) at the 18 points: %s"
      % sorted({round(l[3], 4) for l in locs}))
    L("table of the note (largest competitor per group and criterion, 3 decimals, over the 18 points):")
    for g in groups:
        cells = []
        for cr in crits:
            vals = [c[cr][0] for c in T[g].values() if c[cr][0] is not None]
            cells.append(("%.3f to %.3f (%d points)" % (min(vals), max(vals), len(vals))) if vals else "none")
        L("    %-5s | %s" % (g, " | ".join(cells)))
    L.sec("H. Grid convergence (job 568138): outputs/main/conv_<group>_<point>.out at steps (1/32, 1/120) vs defaults (1/16, 1/60)")
    for g, n in (("su3", "r1d"), ("g2", "r5a"), ("so7", "r4b"), ("usp6", "r1e")):
        cv = parse_torus(P("main", "conv_%s_%s.out" % (g, n)))[n]
        df = T[g][n]
        L("    %-4s %-3s grid %d -> %d points" % (g, n, df["grid"], cv["grid"]))
        for cr in crits:
            a, b = df[cr], cv[cr]
            L("        %-9s %s: default %s / fine %s ; verdicts '%s' / '%s' ; rim %s / %s" % (
                cr[0], cr[1], "none" if a[0] is None else "%.7f" % a[0], "none" if b[0] is None else "%.7f" % b[0],
                a[1], b[1], "-" if a[2] is None else "%.2e" % a[2], "-" if b[2] is None else "%.2e" % b[2]))
    L.sec("I. First run with the unpatched T-038 scripts (job 568105): outputs/torus_unequal_<group>.out")
    if not have_u:
        L("    skipped: the job 568105 outputs are not present")
    for g in (groups if have_u else []):
        U = parse_torus_unpatched(P("torus_unequal_%s.out" % g))
        nd = [(n, cr, v) for n, c in U.items() for cr, v in c["verdict"].items() if v[1] != "identity dominates"]
        L("    %-5s %d points; verdicts other than 'identity dominates': %s" % (
            g, len(U), "; ".join("%s %s %s: max %.7f -> %s at theta/pi = %s" % (n, cr[0], cr[1], v[0], v[1],
                                 U[n]["vals"][cr][0][1]) for n, cr, v in nd)))
        # every value the unpatched run lists also appears in the patched run at the same point (same numbers)
        txt = open(P("main", "torus_%s_allext.out" % g)).read()
        blocks = {m.group(1): m.group(2) for m in re.finditer(r"== point (\S+)\n(.*?)(?=== point|\Z)", txt, re.S)}
        miss = []
        for n, c in U.items():
            for cr, lst in c["vals"].items():
                for v, _ in lst:
                    if v != 1.0 and v < 1.0 and ("%.7f" % v) not in blocks[n]:
                        miss.append((n, cr, v))
        L("          values below 1 listed by the unpatched run that do not appear in the patched output: %s"
          % (miss if miss else "none"))
    # ------------------------------------------------------------ figures
    L.sec("J. Figures")
    holo = list(HOLO.values())
    n1 = draw_map("fig1_delta0_map.png", s24, ext, "delta0",
                  [BLUE[0], BLUE[2], BLUE[4], BLUE[6], BLUE[8], BLUE[10], BLUE[12]],      # steps 100, 200, ..., 700
                  [0.002, 0.004, 0.006, 0.008, 0.010, 0.012], "%.3f",
                  r"$\delta_0=S_{\rm cl,max}/S_{\rm prot}-1$  (leading Cardy order)",
                  r"KKLP excess $\delta_0$ over the black-hole domain, in charge fractions $\varphi_I=Q_I/\sum Q$",
                  "equal charges:\n" + r"$\delta_0=%.5f$ (maximum)" % s24[0]["delta0"], holonomy_pts=holo)
    n2 = draw_map("fig2_r1_map.png", s24, ext, "r1", [ORANGE[0], ORANGE[1], ORANGE[2], ORANGE[3], ORANGE[4], ORANGE[6]],
                  [0.01, 0.02, 0.03, 0.04, 0.05], "%.2f",
                  r"$r_1=\partial\delta/\partial\epsilon$ at $\epsilon=0$ (fixed $\xi$ and $j_1/j_2$)",
                  r"First-order coefficient $r_1$ in $\delta=\delta_0+r_1\,\epsilon+o(\epsilon)$",
                  "equal charges:\n" + r"$r_1=%.4f$ (maximum)" % s24[0]["r1"], label_angles=(-27, 21, -9, 5, -3))
    L("fig1_delta0_map.png: %d wedge points -> %d plotted points after S3 images and de-duplication, %d triangles" % (n1[1], n1[0], n1[2]))
    L("fig2_r1_map.png:     %d wedge points -> %d plotted points, %d triangles" % (n2[1], n2[0], n2[2]))
    rays_used, edge = draw_rays_edge("fig3_delta0_rays_edge.png", s24, ext)
    for th, (n, x, y) in rays_used.items():
        L("fig3 (a) ray theta=%2d: %3d points (scan24_fine + extended), last rho %.4f, last delta0 %.3g" % (th, n, x, y))
    L("fig3 (b) edge: %d points, phi1 from %.6f (delta0 %.9f) to %.6f (delta0 %.9f)" % (
        len(edge), edge[0]["Q1/sQ"], edge[0]["delta0"], edge[-1]["Q1/sQ"], edge[-1]["delta0"]))
    with open(os.path.join(HERE, "numbers.txt"), "w") as fh:
        fh.write("\n".join(L.lines) + "\n")
    print("\n".join(L.lines))


if __name__ == "__main__":
    sys.exit(main())
