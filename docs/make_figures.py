# -*- coding: utf-8 -*-
"""
Generate all data-driven figures for docs/DDA-manual.md.

Run from the repo root inside the DDA conda env:
    mamba activate DDA
    python docs/make_figures.py

Every figure is computed from real code / real measurement files in this repo.
NUMBERING follows document order of the manual (renumbered 2026-11-11, report
§15.5 — the old figNN names live on in earlier report sections as history):
  fig01_adda_cost         analysis/phase4b_bench_results.json
  fig02_resolution_scheme fmin classes + N_dipoles from adda_shape_files/HongPlate_Id9/*.meta.mat
  fig03_size_parameter    sizes from the same meta files
  fig04_dielectric        acf.eps_ice_matzler06  (Mätzler 2006, implemented in acf)
  fig05_orientation_geometry  schematic (3D): Euler/ARO geometry + k-attached scattering grid
  fig06_sym_closure       measured voxel-mirror closure, report Phase 4 §4.6
  fig07_solve_reduction   measured solve counts, report Phase 4 §4.8
  fig08_forward_peak      exact-Mie peak evolution + measured SH-ntrunc demand vs x
                          (data: analysis/phase9_results.json, mie via phase9 scan port)
  fig09_eikonal_model     CONCEPTUAL model figure: (a) optics recipe on the real
                          x=5 HongPlate slab, (b) face-on vs grazing validity icons
                          (psi vs psi=pi, TIR critical angle), (c) measured psi(beta)
                          marched through the real voxels
  fig10_split_concept     CONCEPTUAL split figure: (a) store-the-residual data-flow,
                          (b) SH coefficient triangles 25->6 at x=5 face-on (12.5x),
                          (c) the per-cell decision gate
  fig11_delta_scaling     full/model/residual angular grids + ntrunc bars for three
                          real scan cells (data: analysis/phase9_grids_*.npz, produced
                          by analysis/phase9_dump_grids.py — ADDA re-solve, cached)
  fig12_validity_map      per-cell model-validity map from the Phase-9A scan
                          (data: analysis/phase9_results.json)
  fig13_backlog           analysis/phase5_backlog_inventory.txt
  fig14..fig17            baseline results (delegated to baselines/plot_baseline_figs.py,
                          data: baselines/adda_1.2/, mirrored into docs/figures for Appendix E)

Outputs: docs/figures/figNN_*.png (300 dpi) — referenced by the manual.
"""

import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, 'docs', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, 'adda_python_scripts'))

import adda_control_functions as acf  # noqa: E402

C0 = 299792458.0

# ---------------------------------------------------------------- style ---
# Classical textbook look (matches the LaTeX PDF built by docs/build_latex.py):
# serif (Times) + STIX math, closed figure frame with inward ticks, no grid
# clutter, and a MUTED textbook colour palette (deep desaturated primaries).
# Where an encoding carries the message (safe/unsafe, stored/what-if,
# harmful/beneficial, X/Y polarization) it is ADDITIONALLY carried by linestyle,
# marker, fill texture or open-vs-filled, so every figure still reads in a
# greyscale printout; "bad/hypothetical" content is hatched or open,
# "stored/actual" content is solid.
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.titlesize': 9.5,
    'axes.titleweight': 'normal',
    'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': True, 'axes.spines.right': True,
    'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'legend.fontsize': 8, 'legend.frameon': False,
    'legend.handlelength': 2.2,
    'savefig.facecolor': 'white',
})

# muted textbook palette (deep desaturated primaries, print-safe)
BLUE   = '#0b5394'
RED    = '#b02418'
GREEN  = '#1c7c3c'
OCHRE  = '#8a6d1d'
PURPLE = '#6a3d9a'
TEAL   = '#2e7d82'
LIGHT  = '#b8c8d4'      # pale steel blue — neutral/background elements
# solid/dashed/dotted/dash-dot for up to four series of one family
DASHES = ['-', '--', ':', '-.']


def savefig(fig, name, tight=True):
    path = os.path.join(FIGDIR, name)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('wrote', os.path.relpath(path, REPO))


# ------------------------------------------------------- Fig 04 (dielectric) ---
def fig_dielectric():
    f = np.logspace(7, 12, 1500)                      # 0.01 GHz .. 1 THz (acf valid range)
    Ts = [190.0, 230.0, 270.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for T, ls, tc in zip(Ts, DASHES, (BLUE, OCHRE, RED)):   # cool->warm = cold->warm ice
        eps = np.array([acf.eps_ice_matzler06(ff, T) for ff in f])
        ax1.plot(f / 1e9, eps.real, color=tc, ls=ls, label=f'{T:.0f} K')
        ax2.plot(f / 1e9, eps.imag, color=tc, ls=ls, label=f'{T:.0f} K')
    ax1.set_yscale('linear')
    ax1.set_ylabel(r"$\varepsilon'$ (real part)")
    ax2.set_yscale('log')
    ax2.set_ylabel(r"$\varepsilon''$ (imag. part)")
    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlabel('frequency [GHz]')
        ax.axvspan(1, 887, color=BLUE, alpha=0.08, zorder=0)   # production channel range
        ax.legend()
    ax1.text(1.15, 3.113, 'production channel set\n(35 channels, 1–886 GHz)',
             fontsize=7.5, ha='left')
    ax1.set_title("Mätzler (2006) ice permittivity — acf.eps_ice_matzler06")
    ax2.set_title(r"loss: $\varepsilon''=\alpha(T)/f + \beta(T)\,f$")
    savefig(fig, 'fig04_dielectric.png')


# -------------------------------------------------- Fig 03 (size parameter) ---
def _meta_table(habit='HongPlate_Id9'):
    files = sorted(__import__('glob').glob(
        os.path.join(REPO, 'adda_shape_files', habit, '*.meta.mat')))
    rows = []
    for fn in files:
        m = acf.loadmat(fn)['meta']
        rows.append((m['diameter_vol_eq'] * 1e6, m['frequency_min'],
                     int(m['N_dipoles'])))
    return np.array(rows)                     # Dveq[µm], fmin[Hz], N


def fig_size_parameter():
    rows = _meta_table()
    sizes = np.unique(rows[:, 0])
    f = np.logspace(np.log10(1e9), np.log10(1000e9), 400)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for D in sizes:
        ax.plot(f / 1e9, np.pi * D * 1e-6 * f / C0, color=LIGHT, lw=0.8,
                zorder=1)
    for D, ls, tc in zip((10., 100., 1000., 2596.), DASHES,
                         (BLUE, GREEN, OCHRE, PURPLE)):
        ax.plot(f / 1e9, np.pi * D * 1e-6 * f / C0, color=tc, ls=ls, lw=1.6,
                label=f'$D_{{veq}}$ = {D:g} µm', zorder=3)
    ax.axhline(10, color=RED, ls='--', zorder=4)
    ax.text(1.05, 12.2, 'size_parameter_limit = 10 — jobs above this line are skipped',
            fontsize=7.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [GHz]')
    ax.set_ylabel(r'size parameter $x=\pi D_{veq}/\lambda$')
    ax.set_title(r'HongPlate_Id9 size grid (51 sizes, 10–2596 µm): $x=\pi D_{veq} f/c_0$')
    ax.legend()
    savefig(fig, 'fig03_size_parameter.png')


# ---------------------------------------------- Fig 02 (resolution scheme) ---
def fig_resolution_scheme():
    rows = _meta_table()
    classes = np.unique(rows[:, 1])
    # three real sizes from the HongPlate grid showing the three regimes
    D0 = 10.0
    def _near(target):
        return rows[np.argmin(np.abs(rows[:, 0] - target)), 0]
    picks = [(D0, 'degenerate: same N in all classes'),
             (_near(341), 'partial: coarse classes share N'),
             (_near(2596), 'full: N grows ~ fmin^3')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8),
                                   gridspec_kw={'width_ratios': [1.5, 1]})
    cols = [BLUE, TEAL, GREEN, OCHRE, PURPLE]              # one per fmin class
    edges = np.concatenate([[0.0], classes / 1e9, [1200.]])
    for k, fm in enumerate(classes):
        ax1.barh([0], edges[k + 1] - edges[k], left=edges[k], height=0.5,
                 color=cols[k], edgecolor='w')
        # label ABOVE the block — narrow blocks (120–214 GHz) cannot hold text
        ax1.text((edges[k] + edges[k + 1]) / 2, 0.28, f'fmin=\n{fm/1e9:g}',
                 ha='center', va='bottom', fontsize=7, color='k')
    ax1.barh([1], 1200 - 887, left=887, height=0.5, color='0.75', hatch='//')
    ax1.text((887 + 1200) / 2, 1, 'no class has fmin ≥ f\n(phase-5 replica skips;\nx-cap usually cuts)',
             ha='center', va='center', fontsize=7.5)
    ax1.text(95, 1, 'f < 192 GHz: smallest available class is\npicked -> runs, but grossly over-resolved',
             ha='left', va='center', fontsize=7.5)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['class used for f\nin this interval', 'edge cases'], fontsize=8)
    ax1.set_xticks(classes / 1e9)
    ax1.set_xticklabels([f'{fm/1e9:g}' for fm in classes], fontsize=8)
    ax1.set_ylim(-0.5, 1.7); ax1.set_xlim(0, 1200)
    ax1.set_xlabel('frequency of the *requested* calculation [GHz]'
                   '  (block ends at its fmin)')
    ax1.set_title('Resolution rule: coarsest class with fmin ≥ f\n(then ADDA rescales geometry via -eq_rad)')
    ax1.grid(axis='y', alpha=0)

    for (Dk, _lab), ls, mk, tc in zip(picks, ('-', '--', ':'),
                                       ('o', 's', '^'), (BLUE, OCHRE, RED)):
        sub = rows[rows[:, 0] == Dk]
        N_of_class = {fm: int(n) for fm, n in zip(sub[:, 1], sub[:, 2])}
        ax2.plot([f'{fm/1e9:g}' for fm in classes],
                 [N_of_class.get(fm, np.nan) for fm in classes], mk + ls,
                 color=tc, lw=1.4, label=f'$D_{{veq}}$={Dk:g} µm')
        if Dk == 10.0:
            for fm in classes:
                ax2.annotate(f'{N_of_class[fm]}', (fm/1e9, N_of_class[fm]),
                             textcoords='offset points', xytext=(0, 6),
                             fontsize=7, ha='center', color='k')
    ax2.set_yscale('log')
    ax2.set_xlabel('resolution class (frequency_min) [GHz]')
    ax2.set_ylabel('N_dipoles (fixed per shape file)')
    ax2.set_title('HongPlate_Id9 file sets: N vs class\n(small sizes degenerate, large ~fmin³)')
    ax2.legend(fontsize=8)
    savefig(fig, 'fig02_resolution_scheme.png')


# ------------------------------------------------------- Fig 01 (adda cost) ---
def fig_adda_cost():
    with open(os.path.join(REPO, 'analysis', 'phase4b_bench_results.json')) as fh:
        bench = json.load(fh)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    pts = {}
    for r in bench:
        pts.setdefault(r['x'], []).append(r)
    marks = {1: 'o', 3: 's', 5: 'D', 8: '^'}
    xcols = {1: BLUE, 3: GREEN, 5: OCHRE, 8: RED}
    for x, rs in sorted(pts.items()):
        ax1.scatter([r['N_dipoles'] for r in rs], [r['wall_s'] for r in rs],
                    marker=marks[x], color=xcols[x], s=34, label=f'$x$ = {x}',
                    zorder=3)
    N = np.array([100, 1e5])
    ax1.plot(N, 0.42 / 12312**2.3 * N**2.3, color='0.4', lw=1,
             label=r'guide $\propto N^{2.3}$ (fit)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('$N$ dipoles')
    ax1.set_ylabel('wall time per solve [s]')
    ax1.set_title('ADDA 1.4.0, single core (ice spheroid, 230 K, 183 GHz)')
    ax1.legend()

    x5 = [r for r in bench if r['x'] == 5 and r['dpl'] is not None]
    ref = [r for r in x5 if r['dpl'] == 7][0]['Cext']
    ax2.semilogy([r['dpl_eff'] for r in x5],
                 [abs(r['Cext'] - ref) / ref for r in x5], 'o-', color=BLUE,
                 lw=1.4, label=r'$C_\mathrm{ext}$ rel. err vs dpl = 7')
    auto = [r for r in bench if r['x'] == 5 and r['dpl'] is None]
    ax2.axvline(auto[0]['dpl_eff'], color='k', ls='--', lw=1)
    ax2.text(auto[0]['dpl_eff'] - 0.6, 2e-1, 'ADDA auto-dpl ($\\approx$18)',
             fontsize=8, ha='right')
    ax2.axhline(1e-3, color='0.4', ls=':', lw=1)
    ax2.text(4.2, 1.3e-3, 'solver-noise band (eps 3: $10^{-3}..10^{-11}$)',
             fontsize=8)
    ax2.set_xlabel('effective dipoles per wavelength ($dpl_\\mathrm{eff}$)')
    ax2.set_ylabel(r'$|C_\mathrm{ext}-C_\mathrm{ext}(\mathrm{dpl}=7)|'
                   r'\;/C_\mathrm{ext}(\mathrm{dpl}=7)$')
    ax2.set_title('discretization error dominates solver error ($x$=5)')
    ax2.legend(loc='lower right')
    savefig(fig, 'fig01_adda_cost.png')


# ------------------------------------------------ Fig 07 (solve reduction) ---
def fig_solve_reduction():
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    levels = ['L0: 12', 'L1: 42', 'L2: 162', 'L3: 642', 'L4: 2562']
    allv = [12, 42, 162, 642, 2562]
    xs = np.arange(len(levels))
    # grey background bars = every icosphere vertex would be solved (no mirrors)
    ax.bar(xs, allv, color='0.85', width=0.88, zorder=1,
           label='ALL icosphere vertices solved (no mirror flags)')
    for x, v in zip(xs, allv):
        ax.text(x, v * 1.08, str(v), ha='center', fontsize=8, color='0.35')
    # measured production runs (report §4.8): hatched bars = actual ADDA solves,
    # drawn ON the grey bar of the same refinement level, each directly labeled
    measured = [
        (2, 0.00, 27, '///',
         'ellipsoid gate (XY+XZ+YZ): solves 162→27, wall 3.0×'),
        (3, -0.22, 177, '\\\\\\',
         'HongPlate_Id9 (XY+XZ): solves 642→177, wall 1.73×'),
        (3, 0.22, 337, '...',
         'HongColumn_Id7 (YZ): solves 642→337, wall 1.39×'),
    ]
    for lvl, off, v, htch, name in measured:
        ax.bar(xs[lvl] + off, v, width=0.4, facecolor='0.97', edgecolor='k',
               hatch=htch, lw=1.0, zorder=3, label=name)
        ax.text(xs[lvl] + off, v * 1.15, str(v), ha='center', fontsize=8,
                color='k', fontweight='bold', zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(levels)
    ax.set_yscale('log'); ax.set_ylim(1, 8000)
    ax.set_xlabel('icosphere refinement level $L$ (vertices $V=10\\cdot4^{L}+2$)')
    ax.set_ylabel('ADDA solves per job')
    ax.set_title('Mirror fill-in reduces solves: grey = all vertices, hatched = '
                 'actually solved\n(rest are exact mirror fill-ins; '
                 'measured at the production refinement level, report §4.8)')
    ax.legend(fontsize=7.5, loc='upper left')
    savefig(fig, 'fig07_solve_reduction.png')


# -------------------------------------------------- Fig 06 (sym closure) ---
def fig_sym_closure():
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    habits = ['HongPlate_Id9\n(fine)', 'HongPlate_Id9\n(coarse)',
              'HongColumn_Id7\n(fine)', 'HongColumn_Id7\n(coarse)',
              'LiuSector_Id3\n(fine)', 'LiuSector_Id3\n(coarse)',
              'IconSnow_Id28\n(fine)', 'HexPlaAgg_Id20\n(fine)']
    mirrors = ['XY', 'XZ', 'YZ', 'C2z']
    # report §4.6 measurement table (1.000 = voxel set exactly closed)
    M = np.array([
        [1.000, 1.000, 1.000, 1.000],
        [1.000, 1.000, 0.986, 0.986],
        [1.000, 0.996, 1.000, 0.996],
        [0.986, 0.957, 1.000, 0.957],
        [0.999, 0.872, 0.838, 0.784],
        [0.990, 0.993, 0.997, 0.990],
        [0.069, 0.051, 0.045, np.nan],
        [0.068, 0.049, 0.027, np.nan]])
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    fig.subplots_adjust(left=0.15, right=0.52, top=0.82, bottom=0.09)
    # THE DECISION RULE, encoded redundantly (colour + texture) so it survives
    # greyscale printing too (a continuous colormap made 1.000 vs 0.986 visually
    # indistinguishable although one is safe and the other corrupting):
    #   solid green     closure == 1.000    → mirror flag may be enabled
    #   white + red hatch 1.000 > closure   → fill-in injects a real SHAPE error — forbidden
    #   grey cross-hatch habit intrinsically asymmetric → flag never applicable
    def cell(v):
        if not np.isfinite(v):
            return dict(fc='#ececec', hatch='xxxx', ec='0.55')
        if v >= 0.9995:
            return dict(fc='#c9dcc9', hatch=None, ec=GREEN)
        return dict(fc='1.0', hatch='///', ec=RED)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            st = cell(v)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, lw=1.5, **st))
            txt = 'n/a' if not np.isfinite(v) else f'{v:.3f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='k')
    ax.set_xlim(-0.5, len(mirrors) - 0.5)
    ax.set_ylim(len(habits) - 0.5, -0.5)
    ax.set_xticks(range(len(mirrors))); ax.set_xticklabels(mirrors)
    ax.set_xticks([], minor=True)
    ax.set_yticks(range(len(habits))); ax.set_yticklabels(habits, fontsize=8)
    ax.tick_params(length=0)
    ax.set_title('Voxel-set mirror closure per habit (analysis/phase4_symmetry.py):\n'
                 'flags MUST be set from the COARSEST resolution class (row pairs)')
    ax.legend(handles=[
        Patch(fc='#c9dcc9', ec=GREEN,
              label='1.000 — voxel set exactly closed: flag SAFE'),
        Patch(fc='1.0', hatch='///', ec=RED,
              label='<1.000 — NOT closed: fill-in = real shape error, never enable'),
        Patch(fc='#ececec', hatch='xxxx', ec='0.55',
              label='asymmetric habit — no mirror flag ever applies')],
        loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
    savefig(fig, 'fig06_sym_closure.png', tight=False)


# -------------------------------------------------------- Fig 13 (backlog) ---
def fig_backlog():
    txt = open(os.path.join(REPO, 'analysis', 'phase5_backlog_inventory.txt')).read()
    habits, jobs, cpu, cpu_asym = [], [], [], []
    for ln in txt.splitlines():
        m = re.match(r'\s*(\S+)\s+jobs=\s*(\d+).*?est.*?:\s*([\d.]+) CPU-h.*?asym would be ([\d.]+)', ln)
        if m:
            habits.append(m.group(1).replace('adda_config_', ''))
            jobs.append(int(m.group(2))); cpu.append(float(m.group(3)))
            cpu_asym.append(float(m.group(4)))
        elif 'TRO' in ln:
            m2 = re.search(r'jobs=(\d+)', ln)
            habits.append('HongPlate_Id9_TRO'); jobs.append(int(m2.group(1)))
            cpu.append(0.0); cpu_asym.append(0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    ys = np.arange(len(habits))
    ax1.barh(ys, jobs, color='#4a7ba6')
    for y, j in zip(ys, jobs):
        ax1.text(j + 40, y, f'{j}', va='center', fontsize=8)
    ax1.set_yticks(ys); ax1.set_yticklabels(habits, fontsize=8)
    ax1.set_xlabel('production jobs (T × f × size, after x-cap + resolution rule)')
    ax1.set_title('Phase-5 backlog:\nnumber of production jobs')
    ax1.invert_yaxis()

    h = 0.36
    ax2.barh(ys - h / 2, cpu_asym, height=h, facecolor='1.0', edgecolor='k',
             hatch='///', lw=0.8, zorder=2, label='without mirror flags')
    ax2.barh(ys + h / 2, cpu, height=h, color=BLUE, zorder=2,
             label='with mirror flags (production)')
    for y, (c, ca) in enumerate(zip(cpu, cpu_asym)):
        if ca > 0:
            ax2.text(ca * 1.05, y - h / 2, f'{ca:.0f}', va='center',
                     fontsize=7.5, color='k')
            ax2.text(c * 1.05, y + h / 2, f'{c:.0f}', va='center',
                     fontsize=7.5, color='k')
        else:
            ax2.text(1.5, y, 'minutes/job (orient-avg, no mirrors)',
                     va='center', fontsize=7.5)
    ax2.set_yticks(ys); ax2.set_yticklabels([])
    ax2.set_xscale('log'); ax2.set_xlim(1, 4000)
    ax2.set_xlabel('estimated single-core ARO solve time [CPU-h]')
    ax2.set_title('solve cost per habit:\ntop bar = no mirrors, bottom bar = with flags')
    ax2.invert_yaxis()
    ax2.legend(fontsize=7.5, loc='upper center',
               bbox_to_anchor=(0.5, -0.18), ncol=2)
    savefig(fig, 'fig13_backlog.png')


# ------------------------------------------- Fig 05 (orientation geometry) ---
def _wireframe_sphere(ax, radius=1.0, nlat=3, nlon=8, color='0.7', lw=0.55,
                      alpha=0.6, exclude_lat=None):
    """Light lat/lon wireframe of a unit sphere (matplotlib 3D)."""
    u = np.linspace(0, 2 * np.pi, 120)
    for lat in np.linspace(-90, 90, nlat + 2)[1:-1]:
        if exclude_lat is not None and abs(lat - exclude_lat) < 1e-6:
            continue
        la = np.deg2rad(lat)
        r = radius * np.cos(la)
        ax.plot(r * np.cos(u), r * np.sin(u),
                radius * np.sin(la) * np.ones_like(u),
                color=color, lw=lw, alpha=alpha)
    ax.plot(radius * np.cos(u), radius * np.sin(u), np.zeros_like(u),
            color=color, lw=lw, alpha=0.9)          # equator slightly stronger
    for lon in np.linspace(0, 180, nlon + 1)[:-1]:
        lo = np.deg2rad(lon)
        v = np.linspace(-np.pi / 2, np.pi / 2, 90)
        ax.plot(radius * np.cos(v) * np.cos(lo), radius * np.cos(v) * np.sin(lo),
                radius * np.sin(v), color=color, lw=lw, alpha=alpha)


def fig_orientation_geometry():
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11.0, 5.0))
    u = np.linspace(0, 2 * np.pi, 200)

    # ---- panel (a): lab frame — incidence k-hat and particle axis (zyz Euler)
    ax = fig.add_subplot(121, projection='3d')
    ax.set_axis_off()
    _wireframe_sphere(ax, nlat=2, nlon=6)
    R = 1.0

    # lab z axis (gravity)
    ax.quiver(0, 0, -0.1, 0, 0, R + 0.45, color='k', lw=1.7,
              arrow_length_ratio=0.06)
    ax.text(0, 0, 1.60, 'lab $z$ (gravity)', fontsize=9, ha='center')

    # ARO horizontal-plane ring (beta = 90 deg family)
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u),
            color='0.4', lw=1.2, ls='--', alpha=0.9)

    # incidence direction k-hat: theta_inc from z, azimuth fixed to 0 (x-z plane)
    t_inc = np.deg2rad(115)
    kx, kz = np.sin(t_inc), np.cos(t_inc)
    ax.quiver(0, 0, 0, kx, 0, kz, color=BLUE, lw=2.0,
              arrow_length_ratio=0.09)
    ax.text(kx * 1.08 - 0.05, -0.28, kz * 1.08 - 0.10, 'incidence $\\hat{k}$',
            fontsize=9, color=BLUE, ha='center')

    # theta_inc arc in the x-z plane (from +z down to k-hat)
    tt = np.linspace(0, t_inc, 60)
    ax.plot(np.sin(tt), np.zeros_like(tt), np.cos(tt), color=BLUE, lw=1.3)
    ta = np.deg2rad(78)
    ax.text(0.60 * np.sin(ta), -0.16, 0.60 * np.cos(ta), '$\\theta_{inc}$',
            color=BLUE, fontsize=12)

    # particle symmetry axis: tilt beta off z, own azimuth (the averaged Euler angles)
    b = np.deg2rad(55)
    az = np.deg2rad(-35)
    px, py, pz = np.sin(b) * np.cos(az), np.sin(b) * np.sin(az), np.cos(b)
    ax.quiver(0, 0, 0, px, py, pz, color=RED, lw=2.0, linestyle='--',
              arrow_length_ratio=0.09)
    ax.text(px * 1.20 + 0.15, py * 1.20, pz * 1.20 + 0.12, 'particle axis',
            fontsize=9, color=RED, ha='center')

    # beta arc: great-circle from +z to the particle axis, in its own vertical plane
    tt = np.linspace(0, b, 60)
    ax.plot(np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt),
            color=RED, lw=1.3)
    tb = np.deg2rad(44)
    ax.text(0.68 * np.sin(tb) * np.cos(az) + 0.22,
            0.68 * np.sin(tb) * np.sin(az) - 0.04,
            0.68 * np.cos(tb) - 0.02, '$\\beta$', color=RED, fontsize=12)

    ax.legend(handles=[
        Line2D([], [], color=BLUE, lw=2,
               label='incidence $\\hat{k}$: $\\theta_{inc}$ from lab-$z$,\n'
                     'incidence azimuth fixed (0°)'),
        Line2D([], [], color=RED, lw=2, ls='--',
               label='particle symmetry axis: tilt $\\beta$ from $z$'),
        Line2D([], [], color='0.4', lw=1.4, ls='--',
               label='ARO azimuth ring: $\\alpha,\\gamma$ averaged at fixed $\\beta$\n'
                     '(dashed ring: gravity-horizontal state $\\beta=90°$)')],
        loc='upper left', fontsize=7.5,
        handlelength=1.6, borderpad=0.7)

    ax.set_title('(a) lab frame: one particle orientation = zyz Euler angles\n'
                 r'$R(\alpha,\beta,\gamma)$ — ARO keeps $\beta$, averages $\alpha,\gamma$',
                 fontsize=9.5)
    ax.view_init(elev=16, azim=-62)
    ax.set_box_aspect([1, 1, 1])
    lim = 1.45
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    # ---- panel (b): the mueller/amplitude scattering grid attached to k-hat
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_axis_off()
    _wireframe_sphere(ax2, nlat=4, nlon=8)

    # propagation axis: theta = 0 at the north pole (straight ahead)
    ax2.quiver(0, 0, -0.15, 0, 0, 1.55, color=BLUE, lw=1.8,
               arrow_length_ratio=0.05)
    ax2.text(0.62, -0.38, 1.02, 'propagation $\\hat{k}$  →  $\\theta=0°$\n'
             'is straight ahead at the pole', fontsize=8.5, ha='left', va='center')
    ax2.scatter(0, 0, 1.0, color=BLUE, s=22, zorder=5)
    ax2.scatter(0, 0, -1.0, color=RED, s=30, zorder=5)
    ax2.text(0.12, 0, -1.04, '$\\theta=180°$\nexact backscatter', fontsize=8,
             color=RED, va='top')

    # theta = cone circle (latitude) around k
    th_c = np.deg2rad(60)
    ax2.plot(np.sin(th_c) * np.cos(u), np.sin(th_c) * np.sin(u),
             np.cos(th_c) * np.ones_like(u), color='k', lw=1.8)

    # phi = sweep circle at one theta
    ph_c = np.deg2rad(115)
    r_p = np.sin(ph_c)
    ax2.plot(r_p * np.cos(u), r_p * np.sin(u),
             np.cos(ph_c) * np.ones_like(u), color=OCHRE, lw=1.8)

    # a sample scattering direction n-hat(theta, phi)
    n_th, n_ph = np.deg2rad(115), np.deg2rad(140)
    nx, ny, nz = (np.sin(n_th) * np.cos(n_ph), np.sin(n_th) * np.sin(n_ph),
                  np.cos(n_th))
    ax2.quiver(0, 0, 0, nx, ny, nz, color=PURPLE, lw=1.7, linestyle='-.',
               arrow_length_ratio=0.12)

    ax2.legend(handles=[
        Line2D([], [], color='k', lw=2,
               label='$\\theta$-rings ($\\theta$=const): 181 pts, 0…180°'),
        Line2D([], [], color=OCHRE, lw=2,
               label='$\\varphi$-sweep at one $\\theta$: 361 pts, 0…360°'),
        Line2D([], [], color=PURPLE, lw=2, ls='-.',
               label='one grid point $\\hat{n}(\\theta,\\varphi)$ of\n'
                     'ampl/mueller_scatgrid'),
        Line2D([], [], color=RED, marker='o', ls='', ms=6,
               label='Phase-6 backscatter slice:\n$\\Theta$=180°, $\\Phi$=0° row/column')],
        loc='upper left', fontsize=7.5,
        handlelength=1.6, borderpad=0.7)

    ax2.set_title("(b) every solve's $181\\times361$ scattering grid lives on a sphere\n"
                  r"attached to $\hat{k}$ — mirror transforms remap only $\varphi$, never $\theta$",
                  fontsize=9.5)
    ax2.view_init(elev=18, azim=32)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)

    fig.subplots_adjust(wspace=-0.25)
    savefig(fig, 'fig05_orientation_geometry.png')


# ------------------------------------------- Figs 08–12 (Phase 9) ---
def _p9():
    """import the Phase-9A scan module (validated Mie port + shape parser)."""
    sys.path.insert(0, REPO)
    import analysis.phase9_fwdmodel_scan as p9
    return p9


def _p9_results():
    return json.load(open(os.path.join(REPO, 'analysis', 'phase9_results.json')))


def fig_forward_peak():
    """Why the SH truncant grows with x: the diffraction peak (exact Mie) and
    the measured per-vertex SH ntrunc demand of the production pipeline."""
    p9 = _p9()
    m = np.sqrt(acf.eps_ice_matzler06(887e9, 230.0))
    xs = (1.0, 2.0, 5.0, 10.0)
    # sequential blue -> red ramp: the red large-x curves are the subject
    cols = dict(zip(xs, ('#a8c4d8', '#5b8fb9', BLUE, RED)))
    th = np.linspace(0.0, 180.0, 3601)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for xv in xs:
        an, bn = p9.mie_coeffs(m, xv)
        S1, _ = p9.mie_S(an, bn, th)
        I = np.abs(S1) ** 2
        ax1.plot(th, I, lw=1.1, color=cols[xv], label='x = %g' % xv)
        ax2.plot(th, I, lw=1.5, color=cols[xv], label='x = %g' % xv)
    ax1.set_xlim(0, 180)
    ax1.set_xlabel(r'$\theta$ [deg]')
    ax1.set_ylabel(r'$|S_1(\theta)|^2$   (Mie, ice at 887 GHz/230 K)')
    ax1.set_yscale('log')
    ax1.legend(ncol=2, fontsize=8)
    ax1.set_title('(a) whole sphere: the forward peak is\na large-x phenomenon')
    ax2.set_xlim(0, 50)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\theta$ [deg]')
    ax2.legend(fontsize=8)
    for xv, ls in ((5.0, '--'), (10.0, ':')):
        w = 219.9 / xv                       # first sphere minimum, a = Dveq/2
        ax2.axvline(w, color=cols[xv], ls=ls, lw=1.0)
        ax2.text(w + 1, 1.5, r'$\approx 220°/x$', color=cols[xv], fontsize=8,
                 rotation=90, va='bottom')
    ax2.set_title('(b) zoom: peak width $\\approx 220°/x$ — resolving\nit costs '
                  r'$\propto x$ spectral order')
    # measured SH demand (production expansion rule, HongPlate scan cells)
    xx = np.linspace(0.3, 11, 200)
    ax3.plot(xx, xx + 4 * xx ** (1 / 3) + 2, color='0.4', lw=1.4, ls='-',
             label='Mie partial-wave demand\n'
                   r'$n_{\max}=x+4x^{1/3}+2$ (Wiscombe)')
    ax3.axhline(180, color='0.3', lw=1.0, ls='-.')
    ax3.text(0.42, 205, 'Nyquist cap of the static 361-$\\varphi$ ring',
             color='0.3', fontsize=7.5)
    cells = _p9_results()['cells']
    def pick(suf):
        cc = [c for c in cells if c['tag'].startswith('HongPlate')
              and c['tag'].endswith(suf)]
        return sorted(cc, key=lambda c: c['x'])
    for suf, marker, name in (('_face', 'o', 'face-on (eikonal valid)'),
                              ('_t60', 's', '60° tilt (mixed validity)')):
        cc = pick(suf)
        ax3.scatter([c['x'] for c in cc], [c['ntrunc_full'] for c in cc],
                    marker=marker, s=46, color='k',
                    label='full Mueller ntrunc, ' + name)
        ax3.scatter([c['x'] for c in cc],
                    [c['ntrunc_res_fulleps'] for c in cc], marker=marker, s=46,
                    facecolors='none', edgecolors='k', lw=1.2,
                    label='residual ntrunc (full-eps), ' + name)
    ax3.set_xlabel('size parameter x = $\\pi D_{veq}/\\lambda$')
    ax3.set_ylabel('achieved SH ntrunc')
    ax3.set_yscale('log')
    ax3.set_ylim(2, 400)
    ax3.set_title('(c) measured SH cost vs $x$:\n'
                  'filled = full field, open = model residual')
    ax3.legend(fontsize=6.6, loc='lower right')
    savefig(fig, 'fig08_forward_peak.png')


def _chord_max(grid, d, shat, ds_frac=0.25):
    """max interior chord length (µm) over all parallel rays along shat,
    marched on the voxel grid (same recipe as the Phase-9A eikonal model)."""
    box = np.array(grid.shape)
    cen = (box - 1) / 2.0
    ru = (np.indices(box).reshape(3, -1).T - cen) * d      # physical corner lattice
    e1 = np.cross(shat, [0.0, 0.0, 1.0])
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = e1 / n1
    e2 = np.cross(shat, e1)
    xi = ru @ e1
    eta = ru @ e2
    dp = d
    xig = np.arange(xi.min(), xi.max() + dp, dp)
    etag = np.arange(eta.min(), eta.max() + dp, dp)
    GG = np.meshgrid(xig, etag, indexing='ij')
    rho = GG[0].ravel()[:, None] * e1 + GG[1].ravel()[:, None] * e2
    R = np.abs(ru).max() + 2 * d
    ns = int(np.ceil(2 * R / (d * ds_frac))) + 1
    sp = -R + np.arange(ns) * d * ds_frac
    best = 0.0
    for i0 in range(0, rho.shape[0], 400):
        pts = rho[i0:i0 + 400, None, :] + sp[None, :, None] * shat[None, None, :]
        idx = np.rint(pts / d + cen).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < box), axis=-1)
        inside = np.zeros(idx.shape[:2], dtype=bool)
        ii = idx[ok]
        inside[ok] = grid[ii[:, 0], ii[:, 1], ii[:, 2]]
        best = max(best, inside.sum(axis=1).max())
    return best * d * ds_frac


def fig_eikonal_model():
    """CONCEPTUAL model figure (redesigned 2026-11-11): (a) the optics recipe
    drawn on the real x=5 HongPlate slab, (b) face-on vs grazing validity icons
    (single-pass psi vs TIR multi-pass), (c) the measured psi(beta) knob
    marched through the real voxels."""
    import glob as _glob
    from matplotlib.patches import Rectangle
    p9 = _p9()
    path = _glob.glob(os.path.join(
        REPO, 'adda_shape_files', 'HongPlate_Id9',
        '*Dveq00541um_Mass*Freq0887.000GHz.adda'))[0]
    vox = p9.parse_shape(path)
    u = vox - vox.min(axis=0)
    N = u.shape[0]
    D = 541.0                                     # um (meta diameter_vol_eq)
    d = (D / 2) * (4 * np.pi / (3 * N)) ** (1 / 3)
    box = u.max(axis=0) + 1
    grid = np.zeros(tuple(box), dtype=bool)
    grid[u[:, 0], u[:, 1], u[:, 2]] = True
    Dmax = box[0] * d
    t = box[2] * d
    m_ice = np.sqrt(acf.eps_ice_matzler06(887e9, 230.0))
    k = 2 * np.pi / (p9.C0 / 887e9 * 1e6)
    thc = np.degrees(np.arcsin(1.0 / m_ice.real))     # TIR critical angle
    psi_slab = abs(m_ice - 1) * k * t                # face-on slab-chord knob

    fig = plt.figure(figsize=(11.8, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 0.82],
                          hspace=0.32, wspace=0.20)

    # ---------- (a) the optics recipe on the real slab ----------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.add_patch(Rectangle((-Dmax / 2, -t / 2), Dmax, t, fc='#dfe7ee',
                           ec='0.3', alpha=1.0, hatch='++', lw=1.2))
    for xc in np.linspace(-0.42, 0.42, 7) * Dmax:
        ax.annotate('', xy=(xc, -t / 2 - 4), xytext=(xc, -t / 2 - 130),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))
    ax.text(0, -t / 2 - 158, r'incident plane wave $e^{iks}$',
            color='k', fontsize=8.5, ha='center')
    # two highlighted marched rays with different chords -> phase modulation
    for xr, lwch in ((0.16 * Dmax, 4.5), (-0.34 * Dmax, 1.6)):
        ax.plot([xr, xr], [-t / 2 - 130, t / 2 + 140], color=BLUE, lw=1.0, ls='--')
        ax.annotate('', xy=(xr, t / 2 + 140), xytext=(xr, t / 2 + 20),
                    arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.0))
        ax.plot([xr, xr], [-t / 2, t / 2], color=BLUE, lw=lwch,
                solid_capstyle='round')
    ax.annotate('chords $\\ell(\\rho)$: ray-marched through\nthe exact voxels '
                '(the only numerical step)',
                xy=(0.16 * Dmax, 0), xytext=(0.205 * Dmax, 0.115 * Dmax),
                fontsize=8, color='k',
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
    # exit phase screen: modulated phase drawn as a wavy line
    ys = t / 2 + 100
    xx = np.linspace(-Dmax / 2, Dmax / 2, 400)
    ax.plot(xx, ys + 16 * np.sin(2 * np.pi * xx / (0.16 * Dmax) + np.pi / 2),
            color=TEAL, lw=1.4)
    ax.text(-Dmax / 2, ys + 34,
            r'exit screen $e^{i\Delta(\rho)}$,'
            r' $\Delta=k(m-1)\,\ell(\rho)$',
            fontsize=8.5, color=TEAL, ha='left')
    for ang in (-25, 0, 25):
        aa = np.deg2rad(90 - ang)
        L = 0.15 * Dmax
        ax.annotate('', xy=(L * np.cos(aa), ys + L * np.sin(aa)),
                    xytext=(0, ys),
                    arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.1))
    ax.text(0.06 * Dmax, ys + 0.19 * Dmax,
            r'far field $=\frac{k^{3}}{4\pi}\iint G\,e^{-ik\hat{n}\cdot\rho}\,d^{2}\rho$'
            ' — analytic', color='0.3', fontsize=8.5)
    ax.annotate('', xy=(-Dmax / 2, t / 2 + 30), xytext=(Dmax / 2, t / 2 + 30),
                arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.9))
    ax.text(0, t / 2 + 42, '$D_{max}$ = %.0f µm' % Dmax, fontsize=8,
            ha='center', color='0.3')
    ax.text(-Dmax / 2 - 12, 0, 't = %.0f µm' % t, fontsize=8, color='0.3',
            ha='right', va='center', rotation=90)
    ax.text(0, -t / 2 - 265,
            'the ONLY numerical step is the ray-march — geometry enters '
            'voxel-exactly from the shipped .adda file,\n'
            'so the model exists for ANY habit at ANY orientation; '
            'one global complex constant c is fitted per (T, f, size)',
            fontsize=8.5, ha='center',
            bbox=dict(fc='w', ec='0.6', alpha=0.92))
    ax.set_xlim(-0.62 * Dmax, 0.62 * Dmax)
    ax.set_ylim(-t / 2 - 305, t / 2 + 0.315 * Dmax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title('(a) the recipe: march every ray through the exact voxels, '
                 'then radiate analytically from the exit phase screen',
                 fontsize=9.5)

    # ---------- (b) validity icons: face-on OK, grazing broken --------------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_xlim(-1.48, 1.48); axb.set_ylim(-2.30, 1.32)
    axb.set_aspect('equal'); axb.set_xticks([]); axb.set_yticks([])
    for sp_ in axb.spines.values():
        sp_.set_visible(False)
    hh = 0.16
    for cen, cen_txt, tilt, tc, txt in (
            (-0.75, -0.90, np.deg2rad(8), GREEN,
             '$\\checkmark$' + '\n'
             + r'$\psi\approx 0.7\ll\pi$' + '\n'
             + 'face-on' + '\n'
             + 'model WINS'),
            (0.75, 0.90, np.deg2rad(78), RED,
             '$\\times$' + '\n'
             + r'$\psi\approx 21\gg\pi$' + '\n'
             + 'edge-on' + '\n'
             + 'model FAILS')):
        # slab: thin rectangle rotated by the incidence tilt about its centre
        axb.add_patch(Rectangle((-0.52, -hh), 1.04, 2 * hh, angle=np.degrees(tilt),
                                rotation_point='xy', fc='#dfe7ee', ec='0.3',
                                hatch='++', lw=1.0))
        shat = np.array([np.sin(tilt), np.cos(tilt)])       # incidence dir
        sn = np.array([np.cos(tilt), -np.sin(tilt)])        # in-plane normal
        c0 = np.array([cen, 0.0])
        for off in (-0.30, 0.0, 0.30):                      # incident rays
            foot = c0 + off * sn - 0.62 * shat
            tip = c0 + off * sn - 0.02 * shat
            axb.annotate('', xy=tip, xytext=foot,
                         arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9))
        if tilt < np.deg2rad(30):
            # face-on: short straight chord, clean exit
            w = hh / np.cos(tilt)
            p_in = c0 - w * shat
            p_out = c0 + w * shat
            axb.plot([p_in[0], p_out[0]], [p_in[1], p_out[1]], color=BLUE,
                     lw=3.0, solid_capstyle='round')
            axb.annotate('', xy=c0 + 0.58 * shat, xytext=p_out,
                         arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.0,
                                         linestyle='--'))
        else:
            # grazing: long chord + one internal TIR bounce + shallow escape
            p_in = c0 + np.array([-0.50, -0.16])
            pb = c0 + np.array([0.42, 0.10])
            p_out = c0 + np.array([0.52, -0.13])
            axb.plot([p_in[0], pb[0]], [p_in[1], pb[1]], color=BLUE, lw=3.0,
                     solid_capstyle='round')
            axb.plot([pb[0], p_out[0]], [pb[1], p_out[1]], color=BLUE, lw=3.0,
                     solid_capstyle='round')
            axb.annotate('', xy=p_out + 0.14 * np.array([1, -0.4]), xytext=p_out,
                         arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.0,
                                         linestyle='--'))
            axb.annotate(r'TIR at exit face:'
                         + '\n' + r'multi-pass, $\varphi$ broken',
                         xy=(pb[0], pb[1] + 0.02), xytext=(0.66, 0.26),
                         fontsize=7.5, color=RED,
                         arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
        axb.text(cen_txt, -1.72, txt, fontsize=8.0, color=tc,
                 ha='center', va='center')
    axb.text(0.0, 0.92, 'same plate, same c — only the incidence tilt changes;\n'
             'NO calibration constant can repair the right-hand case',
             fontsize=8.5, ha='center', va='center',
             bbox=dict(fc='w', ec='0.6', alpha=0.92))
    axb.set_title('(b) the activation domain: phase thickness '
                  r'$\psi=|(m-1)k|\,\ell$ vs the' + '\n'
                  r'single-pass limit $\psi=\pi$ '
                  '(ice $n$ = %.2f at 887 GHz, TIR at $\\theta>%d°$)'
                  % (m_ice.real, int(round(thc))), fontsize=9.5)

    # ---------- (c) measured psi(beta) through the real voxels --------------
    ax2 = fig.add_subplot(gs[1, :])
    betas = np.arange(0.0, 91.0, 15.0)
    psi = np.array([abs(m_ice - 1) * k
                    * _chord_max(grid, d, np.array(
                        [np.sin(np.deg2rad(b)), 0, np.cos(np.deg2rad(b))]))
                    for b in betas])
    ax2.axhspan(0, np.pi, color=GREEN, alpha=0.07)
    ax2.axhline(np.pi, color='0.3', lw=1.1, ls='--')
    ax2.text(2, np.pi + 0.5, r'$\psi=\pi$: single-pass eikonal limit',
             color='0.3', fontsize=8)
    ax2.plot(betas, psi, lw=1.6, color=BLUE,
             label=r'$\psi_{max}(\beta)$ marched through the real x=5 HongPlate voxels')
    for b, lab, mk, off, ls, tc in ((0, 'face-on: scan model WINS\n'
                                     '(ntrunc 25→6, 55→41)', 'o', (7, 5.5), '-',
                                     GREEN),
                                    (60, 'tilted: already failing\n'
                                     '(Cext_model/Cext = −0.29)', 'x', (5, 2.2),
                                     '--', RED),
                                    (90, 'edge-on: worst failure\n'
                                     '(TIR at exit faces, $\\psi\\approx$ %.0f rad)'
                                     % psi[-1], 'x', (-52, -1.5), '--', RED)):
        i = int(np.argmin(np.abs(betas - b)))
        ax2.scatter([betas[i]], [psi[i]], s=55, color=tc, marker=mk, zorder=4)
        ax2.annotate(lab, xy=(betas[i], psi[i]),
                     xytext=(betas[i] + off[0], max(psi[i] + off[1], 1.0)),
                     fontsize=7.5, color=tc,
                     arrowprops=dict(arrowstyle='->', color=tc, lw=0.8,
                                     linestyle=ls))
    ax2.set_xlabel('incidence tilt $\\beta$ vs plate axis [deg]')
    ax2.set_ylabel(r'$\psi_{max}=|(m-1)k|\;\ell_{max}$  [rad]')
    ax2.set_xlim(-3, 95)
    ax2.set_ylim(0, max(psi[-1] * 1.12, np.pi * 2.2))
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.set_title('(c) the measured knob: $\\psi_{max}(\\beta)$ stays below $\\pi$ '
                  'face-on (green band, where the scan model wins) and crosses '
                  'it near $40°$–$45°$ — beyond, every scan cell fails', fontsize=9.5)
    savefig(fig, 'fig09_eikonal_model.png')


def fig_split_concept():
    """CONCEPTUAL Phase-9 split figure (added 2026-11-11): (a) the data-flow —
    ADDA still computes the full field, only the residual goes through the
    expensive SH route, model re-added exactly at read time; (b) the SH
    coefficient triangles bought at x=5 face-on (ntrunc 25 -> 6 = 12.5x);
    (c) the per-cell decision gate that makes the split safe."""
    from matplotlib.patches import FancyBboxPatch, Polygon

    fig = plt.figure(figsize=(11.8, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.90, 1.12], wspace=0.22)

    def box(ax, x, y, w, h, txt, fc='#eef2f6', ec='0.3', fs=8.0, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.012', fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs)

    def arrow(ax, p0, p1, color='0.25', lw=1.1, ls='-'):
        ax.annotate('', xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                    linestyle=ls))

    # ---------- (a) data flow: what changes is the EXPANSION, not the solve --
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-1.6, 11.6); ax.set_ylim(0, 6.4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    box(ax, 0.15, 4.6, 2.6, 1.1, 'ADDA solve\n(complete, unchanged)\nfull grid $f$',
        fc='#e6ecf5', ec=BLUE, lw=1.3)
    box(ax, 0.15, 1.5, 2.6, 1.1, 'model descriptor\nchord map $\\ell(\\rho)$\nfrom the same voxels',
        fc='#e8f2ea', ec=GREEN, lw=1.3)
    box(ax, 4.1, 3.0, 2.2, 1.1,
        r'$c\,F_{model}$' + '\n(calibrated once\nper $T,f,$ size)',
        fc='#f7f0e2', ec=OCHRE, lw=1.2)
    box(ax, 4.1, 5.7, 2.2, 0.65,
        r'subtract: $g=f-cF_{model}$', fc='#eef2f6')
    box(ax, 7.65, 3.95, 2.4, 1.2,
        'SH expansion\nof $g$ only\n' + r'$\propto n_{trunc}^{2}$' + ' — CHEAP',
        fc='#e8f2ea', ec=GREEN, lw=1.5)
    box(ax, 7.35, 1.35, 2.4, 1.35,
        'consumer (ARTS/RTTOV)\nrebuilds\n' + r'$f=cF_{model}+g$' + ' exactly',
        fc='#eef2f6')
    arrow(ax, (2.75, 5.15), (4.1, 5.95))
    arrow(ax, (2.75, 2.05), (4.1, 3.35))
    arrow(ax, (5.2, 4.1), (5.2, 5.7))
    arrow(ax, (6.3, 5.95), (7.65, 4.85))
    arrow(ax, (6.3, 3.35), (7.65, 2.45))
    arrow(ax, (8.85, 3.95), (8.85, 2.7), ls='--')
    ax.text(5.0, 0.5, 'the delta is scaled honestly: the model is present on BOTH '
            'sides, so the consumer\nreceives the identical field '
            '(residual truncation error rides on top for free)',
            fontsize=7.6, ha='center', bbox=dict(fc='w', ec='0.6', alpha=0.92))
    ax.set_title('(a) store the residual; keep the model as a descriptor', fontsize=9.5)

    # ---------- (b) coefficient triangles: 25 -> 6 at x=5 face-on -----------
    axb = fig.add_subplot(gs[1])
    axb.set_xlim(-2.6, 8.6); axb.set_ylim(-3.0, 7.2)
    axb.set_aspect('equal'); axb.set_xticks([]); axb.set_yticks([])
    for sp_ in axb.spines.values():
        sp_.set_visible(False)

    def tri(ox, oy, n, color, lw):
        pts = [(ox, oy), (ox + n + 1, oy), (ox, oy + n + 1)]
        axb.add_patch(Polygon(pts, closed=True, fill=False, ec=color, lw=lw))
    # full field (ntrunc 25) and residual (ntrunc 6) share the origin -> nesting
    tri(0.2, 0.0, 6, GREEN, 1.8)
    tri(0.2, 0.0, 25, BLUE, 1.5)
    axb.annotate('', xy=(0.35, 6.7), xytext=(0.35, 0.85),
                 arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.9))
    axb.text(0.10, 3.7, r'$n_{trunc}=25$', rotation=90, fontsize=8, color=BLUE,
             ha='right', va='center')
    axb.annotate('', xy=(1.55, 0.15), xytext=(0.60, 0.15),
                 arrowprops=dict(arrowstyle='<->', color=GREEN, lw=0.9))
    axb.text(1.1, 0.80, r'$6$', fontsize=8, color=GREEN, ha='center')
    axb.text(2.35, 4.4, 'today:\nfull field,\n351 coeffs\nper element',
             fontsize=8, color=BLUE, ha='center')
    axb.text(4.7, 1.30, 'split:\nresidual only,\n28 coeffs\nper element',
             fontsize=8, color=GREEN, ha='center')
    axb.text(3.0, -2.15, 'x = 5, HongPlate face-on: SH count '
             r'$=(n+1)(n+2)/2$' + '\n351 → 28 coefficients per element = 12.5× smaller',
             fontsize=7.6, ha='center', bbox=dict(fc='w', ec=GREEN, alpha=0.95))
    axb.set_title('(b) what it buys: the SH triangle shrinks', fontsize=9.5)

    # ---------- (c) the decision gate ---------------------------------------
    axc = fig.add_subplot(gs[2])
    axc.set_xlim(-2.0, 12.0); axc.set_ylim(0, 6.4)
    axc.set_xticks([]); axc.set_yticks([])
    for sp_ in axc.spines.values():
        sp_.set_visible(False)
    box(axc, 0.1, 5.3, 4.4, 0.7, 'per cell, BEFORE deciding:', fc='#eef2f6')
    box(axc, 0.1, 3.85, 4.4, 1.05,
        r'$C_{ext,model}$' + ' in the sane band\n(0.6–1.5×, sign > 0)\n'
        + r'AND $\|g\|<\|f\|$',
        fc='#f7f0e2', ec=OCHRE, lw=1.2)
    box(axc, 0.1, 2.30, 4.4, 1.05,
        'AND the physical bound\n' + r'$\psi_{max}\lesssim\pi$' + ' (1.9.15),\n'
        'plus $c$-consistency (dev)', fc='#f7f0e2', ec=OCHRE, lw=1.2)
    box(axc, 7.1, 3.85, 4.4, 1.05, 'ALL pass →\nsplit: store SH($g$) + descriptor',
        fc='#e8f2ea', ec=GREEN, lw=1.5)
    box(axc, 7.1, 1.35, 4.4, 1.05, 'ANY fails →\nstore plain full field (today)',
        fc='#f7e4e1', ec=RED, lw=1.5)
    arrow(axc, (2.3, 5.3), (2.3, 4.92))
    arrow(axc, (2.3, 3.85), (2.3, 3.37))
    arrow(axc, (4.5, 4.38), (7.1, 4.38))
    arrow(axc, (4.5, 2.82), (7.1, 1.95))
    axc.text(2.55, 5.05, 'pass', fontsize=7.5, color=GREEN)
    axc.text(2.55, 3.55, 'pass', fontsize=7.5, color=GREEN)
    axc.text(4.7, 4.72, 'all conditions met →', fontsize=7.2, color=GREEN)
    axc.text(4.85, 2.50, 'any violated →', fontsize=7.2, color=RED)
    axc.text(5.0, 0.42, 'HC x5 along-axis slips the $c$-gate (dev 0.14) but is caught '
             'by\n$C_{ext,model}=0.16×$ — several observables, never one alone',
             fontsize=7.6, ha='center', bbox=dict(fc='w', ec='0.6', alpha=0.92))
    axc.set_title('(c) the per-cell decision gate', fontsize=9.5)

    savefig(fig, 'fig10_split_concept.png')


def fig_delta_scaling():
    """Real angular grids: full field vs model vs residual, and the per-cell SH
    storage budget (win / poster child / harmful cells) — data from
    phase9_dump_grids.py.  Right column redesign (2026-11-10): exactly TWO bars —
    what is stored today vs what the model+residual split WOULD store (hatched =
    what-if) — plus an explicit verdict with the coefficient-count ratio. The
    old third 'own-eps' bar was dropped: that rule is not decision-relevant and
    its bar was the source of confusion (worked example stays in manual §1.9.6)."""
    import subprocess
    tags = [('HP_x2_face', 'x=2 face-on — model WIN'),
            ('HP_x5_face', 'x=5 face-on — poster child'),
            ('HP_x5_edge', 'x=5 edge-on — model HARMFUL')]
    npzs = [os.path.join(REPO, 'analysis', 'phase9_grids_%s.npz' % t)
            for t, _ in tags]
    if not all(os.path.exists(p) for p in npzs):
        subprocess.run([sys.executable,
                        os.path.join(REPO, 'analysis', 'phase9_dump_grids.py')],
                       check=True)
    fig, axes = plt.subplots(3, 3, figsize=(12.4, 7.6))
    fig.suptitle('Phase-9A delta-scaling probe on three REAL production cells — '
                 'rows = cell, columns: (left) forward peak θ≤40°, '
                 '(middle) whole sphere, (right) SH storage budget\n'
                 'READ: blue = ADDA full field, red = eikonal model / residual; '
                 'solid bar = stored today, hatched bar = what the model+residual '
                 'split would store; a split is admissible only if the residual '
                 'stays smaller than the field',
                 fontsize=10, y=0.995)
    for r, ((tag, rtitle), npz) in enumerate(zip(tags, npzs)):
        dd = np.load(npz)
        th = dd['thetas']
        sl = slice(None), 0                      # phi = 0 scattering-plane slice
        f0 = np.abs(dd['s_add'][:, :, 0][sl])
        m0 = np.abs(dd['s_mod'][:, :, 0][sl])
        g0 = np.abs(dd['g'][:, :, 0][sl])
        axf, axg, axb = axes[r]
        harmful = float(dd['ratio_total']) > 1.02
        c = th <= 40
        axf.semilogy(th[c], f0[c], lw=1.4, color=BLUE, label='ADDA full field')
        axf.semilogy(th[c], m0[c], lw=1.4, ls='--', color=RED,
                     label='eikonal model (calibrated)')
        axf.set_xlim(0, 40)
        axf.set_xlabel(r'$\theta$ [deg]')
        if r == 0:
            axf.set_ylabel(r'$|s_{11}(\theta)|$  (fwd, log)')
            axf.set_title('(a) model vs ADDA at the peak', fontsize=9.5)
        else:
            axf.set_title(rtitle, fontsize=9)
        axf.legend(fontsize=6.5)
        axg.semilogy(th, f0, lw=1.3, color=BLUE, label='full $|f|$')
        axg.semilogy(th, g0, lw=1.3, color=RED, ls='--',
                     label='residual $|g|=|f-F_{model}|$')
        axg.set_xlim(0, 180)
        axg.set_xlabel(r'$\theta$ [deg]')
        if r == 0:
            axg.set_ylabel(r'$|s_{11}(\theta)|$  (sphere, log)')
            axg.set_title('(b) whole sphere: residual vs full field',
                          fontsize=9.5)
        axg.legend(fontsize=6.5, loc='lower center', ncol=2, framealpha=0.9)
        axg.text(0.42, 0.97, '$|g|/|f|$ = %.2f' % float(dd['ratio_total']),
                 transform=axg.transAxes, fontsize=8, va='top',
                 ha='center')
        # ---- right column: storage budget, two bars only ----
        n_full = int(dd['ntrunc_full'])
        n_res = int(dd['ntrunc_res_fulleps'])
        # stored SH count per Mueller element is triangular (m>=0): (n+1)(n+2)/2
        cnt_full = (n_full + 1) * (n_full + 2) // 2
        cnt_res = (n_res + 1) * (n_res + 2) // 2
        axb.barh([1], [n_full], height=0.58, color=BLUE, edgecolor='#073763',
                 lw=0.8, zorder=2)
        if harmful:
            # gate FAILED: the "smaller" full-eps bar is an artefact of the
            # absolute tolerance — mark the what-if bar as rejected by colour
            # + texture (light red fill + cross-hatch)
            axb.barh([0], [n_res], height=0.58, facecolor='#f3d9d3',
                     edgecolor=RED, hatch='xxxx', lw=1.0, zorder=2)
            reslab = 'residual —\nREJECTED'
            verdict = ('REJECTED: $|g|/|f|$ = %.2f > 1\n'
                       '(broken model, see text)' % float(dd['ratio_total']))
        else:
            axb.barh([0], [n_res], height=0.58, facecolor='1.0',
                     edgecolor=RED, hatch='///', lw=1.0, zorder=2)
            reslab = 'residual\n(would store)'
            verdict = 'ADMISSIBLE: %.1f$\\times$ fewer coefficients\n' \
                      '(%d $\\to$ %d SH terms/element)' % (
                          cnt_full / cnt_res, cnt_full, cnt_res)
        axb.set_yticks([1, 0], ['full field\n(stored today)', reslab], fontsize=8)
        axb.set_xlim(0, max(n_full, n_res) * 1.42)
        axb.set_xlabel('SH ntrunc (coefficient truncation order)')
        axb.text(n_full * 1.02, 1, str(n_full), va='center', fontsize=9)
        axb.text(n_res * 1.02, 0, str(n_res), va='center', fontsize=9)
        axb.set_title('(c) storage budget — %s' % verdict, fontsize=8.5)
    savefig(fig, 'fig11_delta_scaling.png')


def fig_validity_map():
    """When is the model safe? Cext sanity vs residual fraction, per scan cell.
    Redesign (2026-11-10): no colormap/colorbar — the *decision* is binary
    (fewer vs more SH coefficients), so it is encoded as open vs filled
    markers (filled = harmful RED); marker shape still gives the incidence
    orientation."""
    from matplotlib.lines import Line2D
    d = _p9_results()
    cells = d['cells']
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    mk = {'face': 'o', 't60': 's', 'edge': '^', 'along': 'D'}
    for c in cells:
        xr = c['Cext_model'] / c['Cext_adda']
        yv = c['ratio_total']
        gain = c['ntrunc_res_fulleps'] / max(c['ntrunc_full'], 1)
        mkr = 'o'
        for k_, v in mk.items():
            if c['tag'].endswith(k_):
                mkr = v
        ax.scatter([xr], [yv], s=110, marker=mkr, zorder=3,
                   facecolors='none' if gain < 1 else RED,
                   edgecolors='k', linewidths=1.2)
    ax.axhline(1.0, color='k', lw=1.0, ls='--')
    ax.text(1.97, 1.02, r'above: $|g|>|f|$ — model actively HARMFUL',
            fontsize=8, ha='right', va='bottom')
    ax.axvspan(0.6, 1.5, color=BLUE, alpha=0.07, zorder=0)
    ax.text(1.05, 1.78, 'model Cext plausible\n(0.6–1.5 ×)', ha='center',
            fontsize=8, color=BLUE)
    for tag, dx, dy in (('HongPlate_Id9_x5_edge', -0.13, 0.16),
                        ('HongPlate_Id9_x5_t60', -0.36, -0.34),
                        ('HongColumn_Id7_x5_along', 0.10, -0.32),
                        ('HongPlate_Id9_x10_face', 0.10, 0.10)):
        c = next(cc for cc in cells if cc['tag'] == tag)
        ax.annotate(tag.replace('HongPlate_Id9_', 'HP ').replace(
            'HongColumn_Id7_', 'HC ').replace('_x', ' x'),
                    (c['Cext_model'] / c['Cext_adda'], c['ratio_total']),
                    xytext=(c['Cext_model'] / c['Cext_adda'] + dx,
                            c['ratio_total'] + dy), fontsize=7.5,
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='0.35'))
    ax.set_xlabel(r'$C_{ext,\,model}/C_{ext,\,ADDA}$   (model sanity check)')
    ax.set_ylabel(r'residual fraction $|g|/|f|$   (lower = model wins)')
    ax.set_ylim(-0.65, 2.05)
    ax.set_xlim(-0.7, 2.0)
    ax.legend(handles=[
        Line2D([], [], marker='o', ls='', ms=9, mfc='none', mec='k',
               label='face-on incidence'),
        Line2D([], [], marker='s', ls='', ms=9, mfc='none', mec='k',
               label='60° tilt'),
        Line2D([], [], marker='^', ls='', ms=9, mfc='none', mec='k',
               label='edge-on'),
        Line2D([], [], marker='D', ls='', ms=9, mfc='none', mec='k',
               label='along column axis'),
        Line2D([], [], marker='o', ls='', ms=9, mfc='none', mec='k', mew=1.2,
               label='open: fewer SH coefficients\nthan today (net gain)'),
        Line2D([], [], marker='o', ls='', ms=9, mfc=RED, mec='k',
               label='filled: model needs MORE\n(full-eps) — harmful')],
        loc='lower left', fontsize=7.5)
    ax.set_title('Phase-9A validity map — all 15 scan cells\n'
                 'Cext band + below dashed line = the gated activation domain',
                 fontsize=10)
    savefig(fig, 'fig12_validity_map.png')


# --------------------------------------------------------- Fig 14–17 ---
def fig_baseline_set():
    """Appendix E figures: regenerate the baseline report figures (data:
    frozen golden runs under baselines/adda_1.2/) and mirror them into
    docs/figures as fig14..fig17. Rendering is delegated to the baseline
    tooling itself (baselines/plot_baseline_figs.py) so the appendix can
    never drift from the reference report."""
    import shutil
    import subprocess
    script = os.path.join(REPO, 'baselines', 'plot_baseline_figs.py')
    subprocess.run([sys.executable, script], check=True, cwd=REPO)
    src = os.path.join(REPO, 'baselines', 'report')
    for srcname, dstname in [
            ('fig1_crosssections.png', 'fig14_baseline_crosssections.png'),
            ('fig2_sphere_mie.png', 'fig15_baseline_sphere_mie.png'),
            ('fig3_scattering_maps.png', 'fig16_baseline_scattering_3d.png'),
            ('fig4_ellipsoid_phase.png', 'fig17_baseline_ellipsoid_phase.png')]:
        shutil.copyfile(os.path.join(src, srcname), os.path.join(FIGDIR, dstname))
        print('wrote docs/figures/' + dstname)


if __name__ == '__main__':
    fig_dielectric()
    fig_size_parameter()
    fig_resolution_scheme()
    fig_adda_cost()
    fig_solve_reduction()
    fig_sym_closure()
    fig_backlog()
    fig_orientation_geometry()
    fig_forward_peak()
    fig_eikonal_model()
    fig_split_concept()
    fig_delta_scaling()
    fig_validity_map()
    fig_baseline_set()
    print('done.')
