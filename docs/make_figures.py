# -*- coding: utf-8 -*-
"""
Generate all data-driven figures for docs/DDA-manual.md.

Run from the repo root inside the DDA conda env:
    mamba activate DDA
    python docs/make_figures.py

Every figure is computed from real code / real measurement files in this repo:
  fig01_dielectric        acf.eps_ice_matzler06  (Mätzler 2006, implemented in acf)
  fig02_size_parameter    sizes from adda_shape_files/HongPlate_Id9/*.meta.mat
  fig03_resolution_scheme fmin classes + N_dipoles from the same meta files
  fig04_adda_cost         analysis/phase4b_bench_results.json
  fig05_solve_reduction   measured solve counts, report Phase 4 §4.8
  fig06_sym_closure       measured voxel-mirror closure, report Phase 4 §4.6
  fig07_backlog           analysis/phase5_backlog_inventory.txt
  fig08_orientation_geometry  schematic (3D): Euler/ARO geometry + k-attached scattering grid
  fig09..fig12            baseline results (delegated to baselines/plot_baseline_figs.py,
                          data: baselines/adda_1.2/, mirrored into docs/figures for Appendix E)
  fig13_forward_peak      exact-Mie peak evolution + measured SH-ntrunc demand vs x
                          (data: analysis/phase9_results.json, mie via phase9 scan port)
  fig14_eikonal_model     eikonal screen-model schematic + chord map of a real
                          HongPlate voxel shape
  fig15_delta_scaling     full/model/residual angular grids + ntrunc bars for three
                          real scan cells (data: analysis/phase9_grids_*.npz, produced
                          by analysis/phase9_dump_grids.py — ADDA re-solve, cached)
  fig16_validity_map      per-cell model-validity map from the Phase-9A scan
                          (data: analysis/phase9_results.json)

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
plt.rcParams.update({'figure.dpi': 150, 'font.size': 9,
                     'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def savefig(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print('wrote', os.path.relpath(path, REPO))


# ---------------------------------------------------------------- Fig 01 ---
def fig_dielectric():
    f = np.logspace(7, 12, 1500)                      # 0.01 GHz .. 1 THz (acf valid range)
    Ts = [190.0, 230.0, 270.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for T in Ts:
        eps = np.array([acf.eps_ice_matzler06(ff, T) for ff in f])
        ax1.plot(f / 1e9, eps.real, label=f'{T:.0f} K')
        ax2.plot(f / 1e9, eps.imag, label=f'{T:.0f} K')
    ax1.set_yscale('linear')
    ax1.set_ylabel(r"$\varepsilon'$ (real part)")
    ax2.set_yscale('log')
    ax2.set_ylabel(r"$\varepsilon''$ (imag. part)")
    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xlabel('frequency [GHz]')
        ax.axvspan(0.1, 1.5, color='0.85', zorder=0)    # window region
        ax.legend(framealpha=0.6)
    ax1.set_title("Mätzler (2006) ice permittivity — acf.eps_ice_matzler06")
    ax2.set_title(r"loss: $\varepsilon''=\alpha(T)/f + \beta(T)\,f$")
    savefig(fig, 'fig01_dielectric.png')


# ---------------------------------------------------------------- Fig 02 ---
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
        ax.plot(f / 1e9, np.pi * D * 1e-6 * f / C0, color='0.8', lw=0.8,
                zorder=1)
    for D, col in [(10., 'tab:blue'), (100., 'tab:orange'),
                   (1000., 'tab:red'), (2596., 'tab:purple')]:
        ax.plot(f / 1e9, np.pi * D * 1e-6 * f / C0, color=col, lw=1.8,
                label=f'$D_{{veq}}$ = {D:g} µm', zorder=3)
    ax.axhline(10, color='k', ls='--', zorder=4)
    ax.text(1.05, 12.2, 'size_parameter_limit = 10 — jobs above this line are skipped',
            fontsize=7.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('frequency [GHz]')
    ax.set_ylabel(r'size parameter $x=\pi D_{veq}/\lambda$')
    ax.set_title(r'HongPlate_Id9 size grid (51 sizes, 10–2596 µm): $x=\pi D_{veq} f/c_0$')
    ax.legend(framealpha=0.7)
    savefig(fig, 'fig02_size_parameter.png')


# ---------------------------------------------------------------- Fig 03 ---
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
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(classes)))
    edges = np.concatenate([[0.0], classes / 1e9, [1200.]])
    for k, fm in enumerate(classes):
        ax1.barh([0], edges[k + 1] - edges[k], left=edges[k], height=0.5,
                 color=cols[k], edgecolor='w')
        ax1.text((edges[k] + edges[k + 1]) / 2, 0, f'class\nfmin={fm/1e9:g}',
                 ha='center', va='center', fontsize=8,
                 color='w' if 0.3 < k/len(classes) < 0.95 else 'k')
    ax1.barh([1], 1200 - 887, left=887, height=0.5, color='0.75', hatch='//')
    ax1.text((887 + 1200) / 2, 1, 'no class has fmin ≥ f\n(phase-5 replica skips;\nx-cap usually cuts)',
             ha='center', va='center', fontsize=7.5)
    ax1.text(95, 1, 'f < 192 GHz: smallest available class is\npicked -> runs, but grossly over-resolved',
             ha='left', va='center', fontsize=7.5)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(['pick per (size,f)', 'edge cases'])
    ax1.set_ylim(-0.5, 1.7); ax1.set_xlim(0, 1200)
    ax1.set_xlabel('frequency of the *requested* calculation [GHz]')
    ax1.set_title('Resolution rule: coarsest class with fmin ≥ f\n(then ADDA rescales geometry via -eq_rad)')
    ax1.grid(axis='y', alpha=0)

    for (Dk, _lab), col in zip(picks, ('tab:blue', 'tab:orange', 'tab:red')):
        sub = rows[rows[:, 0] == Dk]
        N_of_class = {fm: int(n) for fm, n in zip(sub[:, 1], sub[:, 2])}
        ax2.plot([f'{fm/1e9:g}' for fm in classes],
                 [N_of_class.get(fm, np.nan) for fm in classes], 'o-',
                 color=col, label=f'$D_{{veq}}$={Dk:g} µm')
        if Dk == 10.0:
            for fm in classes:
                ax2.annotate(f'{N_of_class[fm]}', (fm/1e9, N_of_class[fm]),
                             textcoords='offset points', xytext=(0, 6),
                             fontsize=7, ha='center', color=col)
    ax2.set_yscale('log')
    ax2.set_xlabel('resolution class (frequency_min) [GHz]')
    ax2.set_ylabel('N_dipoles (fixed per shape file)')
    ax2.set_title('HongPlate_Id9 file sets: N vs class\n(small sizes degenerate, large ~fmin³)')
    ax2.legend(framealpha=0.7, fontsize=8)
    savefig(fig, 'fig03_resolution_scheme.png')


# ---------------------------------------------------------------- Fig 04 ---
def fig_adda_cost():
    with open(os.path.join(REPO, 'analysis', 'phase4b_bench_results.json')) as fh:
        bench = json.load(fh)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    pts = {}
    for r in bench:
        pts.setdefault(r['x'], []).append(r)
    marks = {1: 'o', 3: 's', 5: 'D', 8: '^'}
    for x, rs in sorted(pts.items()):
        ax1.scatter([r['N_dipoles'] for r in rs], [r['wall_s'] for r in rs],
                    marker=marks[x], label=f'$x$ = {x}', zorder=3)
    N = np.array([100, 1e5])
    ax1.plot(N, 0.42 / 12312**2.3 * N**2.3, 'k-', lw=1, alpha=0.7,
             label=r'guide $\propto N^{2.3}$ (fit)')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('$N$ dipoles')
    ax1.set_ylabel('wall time per solve [s]')
    ax1.set_title('ADDA 1.4.0, single core (ice spheroid, 230 K, 183 GHz)')
    ax1.legend(framealpha=0.7)

    x5 = [r for r in bench if r['x'] == 5 and r['dpl'] is not None]
    ref = [r for r in x5 if r['dpl'] == 7][0]['Cext']
    ax2.semilogy([r['dpl_eff'] for r in x5],
                 [abs(r['Cext'] - ref) / ref for r in x5], 'o-',
                 label=r'$C_\mathrm{ext}$ rel. err vs dpl = 7')
    auto = [r for r in bench if r['x'] == 5 and r['dpl'] is None]
    ax2.axvline(auto[0]['dpl_eff'], color='tab:green', ls='--')
    ax2.text(auto[0]['dpl_eff'] - 0.6, 2e-1, 'ADDA auto-dpl ($\\approx$18)',
             color='tab:green', fontsize=8, ha='right')
    ax2.axhline(1e-3, color='r', ls=':')
    ax2.text(4.2, 1.3e-3, 'solver-noise band (eps 3: $10^{-3}..10^{-11}$)',
             color='r', fontsize=8)
    ax2.set_xlabel('effective dipoles per wavelength ($dpl_\\mathrm{eff}$)')
    ax2.set_ylabel(r'$|C_\mathrm{ext}-C_\mathrm{ext}(\mathrm{dpl}=7)|'
                   r'\;/C_\mathrm{ext}(\mathrm{dpl}=7)$')
    ax2.set_title('discretization error dominates solver error ($x$=5)')
    ax2.legend(framealpha=0.7)
    savefig(fig, 'fig04_adda_cost.png')


# ---------------------------------------------------------------- Fig 05 ---
def fig_solve_reduction():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    levels = ['L0: 12', 'L1: 42', 'L2: 162', 'L3: 642', 'L4: 2562']
    allv = [12, 42, 162, 642, 2562]
    xs = np.arange(len(levels))
    ax.bar(xs, allv, color='0.75', label='all icosphere vertices (no mirror)')
    # measured runs at the production refinement level (report §4.8)
    ax.bar(2 + 0, 27, color='tab:purple', width=0.5,
           label='ellipsoid gate: solves 162→27 (wall 3.0×)')
    ax.bar(3 + 0, 177, color='tab:blue', width=0.5,
           label='HongPlate XY+XZ: solves 642→177 (wall 1.73×)')
    ax.bar(3 + 0.5, 337, color='tab:orange', width=0.5,
           label='HongColumn YZ: solves 642→337 (wall 1.39×)')
    for x, v in zip(xs, allv):
        ax.text(x, v * 1.06, str(v), ha='center', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(levels)
    ax.set_yscale('log'); ax.set_ylim(1, 6000)
    ax.set_xlabel('icosphere refinement level $L$ (vertices $V=10\\cdot4^{L}+2$)')
    ax.set_ylabel('ADDA solves per job')
    ax.set_title('Mirror fill-in reduces solves (measured, report §4.8)')
    ax.legend(framealpha=0.7, fontsize=8)
    savefig(fig, 'fig05_solve_reduction.png')


# ---------------------------------------------------------------- Fig 06 ---
def fig_sym_closure():
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
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1.02, aspect='auto')
    ax.set_xticks(range(len(mirrors))); ax.set_xticklabels(mirrors)
    ax.set_yticks(range(len(habits))); ax.set_yticklabels(habits, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i,j]:.3f}', ha='center', va='center', fontsize=8)
            else:
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8)
    ax.grid(alpha=0.2)
    ax.set_title('Voxel-set mirror closure (analysis/phase4_symmetry.py):\n'
                 '1.000 = flag usable; <1 = fill-in would corrupt the shape')
    fig.colorbar(im, ax=ax, shrink=0.8, label='closure fraction')
    savefig(fig, 'fig06_sym_closure.png')


# ---------------------------------------------------------------- Fig 07 ---
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6),
                                   gridspec_kw={'width_ratios': [1.25, 1]})
    ys = np.arange(len(habits))
    ax1.barh(ys, jobs, color='tab:blue', alpha=0.35)
    ax1.barh(ys + 0.35, [j for j in jobs], color='none')
    for y, j in zip(ys, jobs):
        ax1.text(j + 40, y, f'{j}', va='center', fontsize=8)
    ax1.set_yticks(ys); ax1.set_yticklabels(habits, fontsize=8)
    ax1.set_xlabel('production jobs (T × f × size, after x-cap + resolution rule)')
    ax1.set_title('Phase-5 backlog size (analysis/phase5_backlog_inventory.txt)')
    ax1.invert_yaxis()

    ax2.barh(ys, cpu, color='tab:green', label='with mirror flags')
    ax2.barh(ys, cpu_asym, color='tab:red', alpha=0.28, height=1.5,
             label='without mirrors')
    for y, (c, ca) in enumerate(zip(cpu, cpu_asym)):
        if ca > 0:
            ax2.text(ca * 1.02, y, f'{ca:.0f}→{c:.0f}', va='center', fontsize=8)
        else:
            ax2.text(2, y, 'minutes/job (orient-avg)', va='center', fontsize=8)
    ax2.set_yticks(ys); ax2.set_yticklabels([])
    ax2.set_xscale('log'); ax2.set_xlim(1, 4000)
    ax2.set_xlabel('estimated single-core ARO solve time [CPU-h]')
    ax2.set_title('cost: mirrors vs no mirrors')
    ax2.invert_yaxis()
    ax2.legend(framealpha=0.7, fontsize=8)
    savefig(fig, 'fig07_backlog.png')


# --------------------------------------------------------------- Fig 08 ---
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
            color='tab:blue', lw=1.2, ls='--', alpha=0.8)

    # incidence direction k-hat: theta_inc from z, azimuth fixed to 0 (x-z plane)
    t_inc = np.deg2rad(115)
    kx, kz = np.sin(t_inc), np.cos(t_inc)
    ax.quiver(0, 0, 0, kx, 0, kz, color='tab:red', lw=2.0,
              arrow_length_ratio=0.09)
    ax.text(kx * 1.08 - 0.05, -0.28, kz * 1.08 - 0.10, 'incidence $\\hat{k}$',
            fontsize=9, color='tab:red', ha='center')

    # theta_inc arc in the x-z plane (from +z down to k-hat)
    tt = np.linspace(0, t_inc, 60)
    ax.plot(np.sin(tt), np.zeros_like(tt), np.cos(tt), color='tab:red', lw=1.3)
    ta = np.deg2rad(78)
    ax.text(0.60 * np.sin(ta), -0.16, 0.60 * np.cos(ta), '$\\theta_{inc}$',
            color='tab:red', fontsize=12)

    # particle symmetry axis: tilt beta off z, own azimuth (the averaged Euler angles)
    b = np.deg2rad(55)
    az = np.deg2rad(-35)
    px, py, pz = np.sin(b) * np.cos(az), np.sin(b) * np.sin(az), np.cos(b)
    ax.quiver(0, 0, 0, px, py, pz, color='tab:blue', lw=2.0,
              arrow_length_ratio=0.09)
    ax.text(px * 1.20 + 0.15, py * 1.20, pz * 1.20 + 0.12, 'particle axis',
            fontsize=9, color='tab:blue', ha='center')

    # beta arc: great-circle from +z to the particle axis, in its own vertical plane
    tt = np.linspace(0, b, 60)
    ax.plot(np.sin(tt) * np.cos(az), np.sin(tt) * np.sin(az), np.cos(tt),
            color='tab:blue', lw=1.3)
    tb = np.deg2rad(44)
    ax.text(0.68 * np.sin(tb) * np.cos(az) + 0.22,
            0.68 * np.sin(tb) * np.sin(az) - 0.04,
            0.68 * np.cos(tb) - 0.02, '$\\beta$', color='tab:blue', fontsize=12)

    ax.legend(handles=[
        Line2D([], [], color='tab:red', lw=2,
               label='incidence $\\hat{k}$: $\\theta_{inc}$ from lab-$z$,\n'
                     'incidence azimuth fixed (0°)'),
        Line2D([], [], color='tab:blue', lw=2,
               label='particle symmetry axis: tilt $\\beta$ from $z$'),
        Line2D([], [], color='tab:blue', lw=1.4, ls='--',
               label='ARO azimuth ring: $\\alpha,\\gamma$ averaged at fixed $\\beta$\n'
                     '(dashed ring: gravity-horizontal state $\\beta=90°$)')],
        loc='upper left', fontsize=7.5, framealpha=0.85,
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
    ax2.quiver(0, 0, -0.15, 0, 0, 1.55, color='k', lw=1.8,
               arrow_length_ratio=0.05)
    ax2.text(0.62, -0.38, 1.02, 'propagation $\\hat{k}$  →  $\\theta=0°$\n'
             'is straight ahead at the pole', fontsize=8.5, ha='left', va='center')
    ax2.scatter(0, 0, 1.0, color='0.3', s=22, zorder=5)
    ax2.scatter(0, 0, -1.0, color='tab:red', s=30, zorder=5)
    ax2.text(0.12, 0, -1.04, '$\\theta=180°$\nexact backscatter', fontsize=8,
             color='tab:red', va='top')

    # theta = cone circle (latitude) around k
    th_c = np.deg2rad(60)
    ax2.plot(np.sin(th_c) * np.cos(u), np.sin(th_c) * np.sin(u),
             np.cos(th_c) * np.ones_like(u), color='tab:orange', lw=1.8)

    # phi = sweep circle at one theta
    ph_c = np.deg2rad(115)
    r_p = np.sin(ph_c)
    ax2.plot(r_p * np.cos(u), r_p * np.sin(u),
             np.cos(ph_c) * np.ones_like(u), color='tab:green', lw=1.8)

    # a sample scattering direction n-hat(theta, phi)
    n_th, n_ph = np.deg2rad(115), np.deg2rad(140)
    nx, ny, nz = (np.sin(n_th) * np.cos(n_ph), np.sin(n_th) * np.sin(n_ph),
                  np.cos(n_th))
    ax2.quiver(0, 0, 0, nx, ny, nz, color='tab:purple', lw=1.7,
               arrow_length_ratio=0.12)

    ax2.legend(handles=[
        Line2D([], [], color='tab:orange', lw=2,
               label='$\\theta$-rings ($\\theta$=const): 181 pts, 0…180°'),
        Line2D([], [], color='tab:green', lw=2,
               label='$\\varphi$-sweep at one $\\theta$: 361 pts, 0…360°'),
        Line2D([], [], color='tab:purple', lw=2,
               label='one grid point $\\hat{n}(\\theta,\\varphi)$ of\n'
                     'ampl/mueller_scatgrid'),
        Line2D([], [], color='tab:red', marker='o', ls='', ms=6,
               label='Phase-6 backscatter slice:\n$\\Theta$=180°, $\\Phi$=0° row/column')],
        loc='upper left', fontsize=7.5, framealpha=0.85,
        handlelength=1.6, borderpad=0.7)

    ax2.set_title("(b) every solve's $181\\times361$ scattering grid lives on a sphere\n"
                  r"attached to $\hat{k}$ — mirror transforms remap only $\varphi$, never $\theta$",
                  fontsize=9.5)
    ax2.view_init(elev=18, azim=32)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)

    fig.subplots_adjust(wspace=-0.25)
    savefig(fig, 'fig08_orientation_geometry.png')


# ------------------------------------------------- Fig 13-16 (Phase 9) ---
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
    cols = dict(zip(xs, plt.cm.viridis(np.linspace(0.15, 0.9, len(xs)))))
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
    ax1.legend(framealpha=0.7, ncol=2, fontsize=8)
    ax1.set_title('(a) whole sphere: the forward peak is a\nlarge-x phenomenon '
                  '(Rayleigh x=1 has NO spike)')
    ax2.set_xlim(0, 50)
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\theta$ [deg]')
    ax2.legend(framealpha=0.7, fontsize=8)
    for xv, ls in ((5.0, '--'), (10.0, ':')):
        w = 219.9 / xv                       # first sphere minimum, a = Dveq/2
        ax2.axvline(w, color=cols[xv], ls=ls, lw=1.0)
        ax2.text(w + 1, 1.5, r'$\approx 220°/x$', color=cols[xv], fontsize=8,
                 rotation=90, va='bottom')
    ax2.set_title('(b) zoom: peak narrows as $\\lambda/a$ — resolving\nit costs '
                  r'$\propto x$ spectral order')
    # measured SH demand (production expansion rule, HongPlate scan cells)
    xx = np.linspace(0.3, 11, 200)
    ax3.plot(xx, xx + 4 * xx ** (1 / 3) + 2, color='0.45', lw=1.4, ls='-',
             label='Mie partial-wave demand\n'
                   r'$n_{\max}=x+4x^{1/3}+2$ (Wiscombe)')
    ax3.axhline(180, color='tab:red', lw=1.0, ls='--')
    ax3.text(0.42, 205, 'Nyquist cap of the static 361-$\\varphi$ ring',
             color='tab:red', fontsize=7.5)
    cells = _p9_results()['cells']
    def pick(suf):
        cc = [c for c in cells if c['tag'].startswith('HongPlate')
              and c['tag'].endswith(suf)]
        return sorted(cc, key=lambda c: c['x'])
    for suf, marker, name in (('_face', 'o', 'face-on (eikonal valid)'),
                              ('_t60', 's', '60° tilt (mixed validity)')):
        cc = pick(suf)
        ax3.scatter([c['x'] for c in cc], [c['ntrunc_full'] for c in cc],
                    marker=marker, s=46, color='tab:blue',
                    label='full Mueller ntrunc, ' + name)
        ax3.scatter([c['x'] for c in cc],
                    [c['ntrunc_res_fulleps'] for c in cc], marker=marker, s=46,
                    facecolors='none', edgecolors='tab:green',
                    label='residual ntrunc (full-eps), ' + name)
    ax3.set_xlabel('size parameter x = $\\pi D_{veq}/\\lambda$')
    ax3.set_ylabel('achieved SH ntrunc')
    ax3.set_yscale('log')
    ax3.set_ylim(2, 400)
    ax3.set_title('(c) measured cost: ntrunc grows like x — and the\nanalytic-'
                  'model RESIDUAL breaks the growth (9A)')
    ax3.legend(framealpha=0.7, fontsize=6.8, loc='lower right')
    savefig(fig, 'fig13_forward_peak.png')


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
    """Schematic of the eikonal screen model + phase thickness vs incidence
    tilt marched through the REAL HongPlate voxels (validity knob)."""
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
    fig = plt.figure(figsize=(11.8, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.24)

    # ---------- (a) schematic on the slab defined by the real shape
    ax = fig.add_subplot(gs[0])
    ax.set_aspect('equal')
    ax.add_patch(Rectangle((-Dmax / 2, -t / 2), Dmax, t, fc='tab:blue',
                           ec='tab:blue', alpha=0.30, hatch='++', lw=1.2))
    # incident plane wave from below (face-on)
    for xc in np.linspace(-0.42, 0.42, 7) * Dmax:
        ax.annotate('', xy=(xc, -t / 2 - 4), xytext=(xc, -t / 2 - 130),
                    arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=1.2))
    ax.text(0, -t / 2 - 158, r'incident plane wave $e^{iks}$',
            color='tab:red', fontsize=8.5, ha='center')
    # highlighted voxel-marched ray + chord
    xr = 0.16 * Dmax
    ax.plot([xr, xr], [-t / 2 - 130, t / 2 + 140], color='k', lw=1.0, ls='--')
    ax.annotate('', xy=(xr, t / 2 + 140), xytext=(xr, t / 2 + 20),
                arrowprops=dict(arrowstyle='-|>', color='k', lw=1.0))
    ax.plot([xr, xr], [-t / 2, t / 2], color='tab:orange', lw=4,
            solid_capstyle='round')
    ax.annotate('chord $\\ell(\\rho)$ — marched through\nthe exact voxels',
                xy=(xr, 0), xytext=(xr + 0.10 * Dmax, 0.16 * Dmax),
                fontsize=8, color='tab:orange',
                arrowprops=dict(arrowstyle='->', color='tab:orange', lw=0.8))
    # exit phase screen + diffracted fan
    ax.plot([-Dmax / 2, Dmax / 2], [t / 2 + 100] * 2, color='tab:purple',
            lw=1.2, ls=':')
    ax.text(Dmax / 2, t / 2 + 112, r'screen $\rho$', fontsize=8,
            color='tab:purple', ha='right')
    for ang in (-25, 0, 25):
        aa = np.deg2rad(90 - ang)
        L = 0.15 * Dmax
        ax.annotate('', xy=(L * np.cos(aa), t / 2 + 100 + L * np.sin(aa)),
                    xytext=(0, t / 2 + 100),
                    arrowprops=dict(arrowstyle='-|>', color='tab:purple',
                                    lw=1.1))
    ax.text(0.11 * Dmax, t / 2 + 0.22 * Dmax,
            r'$\propto e^{-ik\,\hat{n}\cdot\rho}$', color='tab:purple',
            fontsize=8.5)
    # dimension arrows
    ax.annotate('', xy=(-Dmax / 2, t / 2 + 30), xytext=(Dmax / 2, t / 2 + 30),
                arrowprops=dict(arrowstyle='<->', color='0.3', lw=0.9))
    ax.text(0, t / 2 + 42, '$D_{max}$ = %.0f µm' % Dmax, fontsize=8,
            ha='center', color='0.3')
    ax.text(-Dmax / 2 - 12, 0, 't = %.0f µm' % t, fontsize=8, color='0.3',
            ha='right', va='center', rotation=90)
    ax.text(0, -t / 2 - 260,
            '$\\varphi(s)=e^{ik[s+(m-1)\\,\\ell(\\rho)]}$'
            '  (single-pass eikonal)      '
            '$G(\\rho)=\\int(m^{2}-1)\\,\\varphi\\,ds$\n'
            '$S_{model}=\\frac{k^{3}}{4\\pi}\\iint G(\\rho)\\,'
            'e^{-ik\\hat{n}\\cdot\\rho}\\,d^{2}\\rho$'
            '   × one calibrated constant c',
            fontsize=8.5, ha='center',
            bbox=dict(fc='w', ec='0.6', alpha=0.92))
    ax.set_xlim(-0.62 * Dmax, 0.62 * Dmax)
    ax.set_ylim(-t / 2 - 315, t / 2 + 0.30 * Dmax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_title('(a) the general-case model: ray-march the exact voxel '
                 'geometry,\nthen radiate analytically from the exit phase '
                 'screen', fontsize=9.5)

    # ---------- (b) phase thickness vs incidence tilt (real voxel march)
    ax2 = fig.add_subplot(gs[1])
    m_ice = np.sqrt(acf.eps_ice_matzler06(887e9, 230.0))
    k = 2 * np.pi / (p9.C0 / 887e9 * 1e6)
    betas = np.arange(0.0, 91.0, 15.0)
    psi = np.array([abs(m_ice - 1) * k
                    * _chord_max(grid, d, np.array(
                        [np.sin(np.deg2rad(b)), 0, np.cos(np.deg2rad(b))]))
                    for b in betas])
    ax2.plot(betas, psi, lw=1.6, color='tab:blue',
             label=r'$\psi_{max}(\beta)$ marched through the real x=5 HongPlate voxels')
    ax2.axhline(np.pi, color='tab:green', lw=1.1, ls='--')
    ax2.text(2, np.pi + 0.5, r'$\psi=\pi$: single-pass eikonal limit',
             color='tab:green', fontsize=8)
    for b, lab, col, off in ((0, 'face-on: scan model WINS\n'
                              '(ntrunc 25→6, 55→41)', 'tab:green', (7, 5.5)),
                             (60, 'tilted: already failing\n'
                              '(Cext_model/Cext = −0.29)', 'tab:red', (5, 2.2)),
                             (90, 'edge-on: worst failure\n'
                              '(TIR at exit faces, $\\psi\\approx$ %.0f rad)'
                              % psi[-1], 'tab:red', (-52, -1.5))):
        i = int(np.argmin(np.abs(betas - b)))
        ax2.scatter([betas[i]], [psi[i]], s=55, color=col, zorder=4)
        ax2.annotate(lab, xy=(betas[i], psi[i]),
                     xytext=(betas[i] + off[0], max(psi[i] + off[1], 1.0)),
                     fontsize=7.5, color=col,
                     arrowprops=dict(arrowstyle='->', color=col, lw=0.8))
    ax2.set_xlabel('incidence tilt $\\beta$ vs plate axis [deg]')
    ax2.set_ylabel(r'$\psi_{max}=|(m-1)k|\;\ell_{max}$  [rad]')
    ax2.set_xlim(-3, 95)
    ax2.set_ylim(0, max(psi[-1] * 1.12, np.pi * 2.2))
    ax2.legend(fontsize=7.5, framealpha=0.7, loc='upper left')
    ax2.set_title('(b) why the model has an activation domain: grazing '
                  'incidence\nmultiplies chord (→ TIR, multi-pass) and '
                  r'$\psi$; ice $n$ = %.2f at 887 GHz' % m_ice.real,
                  fontsize=9.5)
    savefig(fig, 'fig14_eikonal_model.png')


def fig_delta_scaling():
    """Real angular grids: full field vs model vs residual, and the ntrunc bars
    (win / poster child / harmful cells) — data from phase9_dump_grids.py."""
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
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 9.2))
    for r, ((tag, rtitle), npz) in enumerate(zip(tags, npzs)):
        dd = np.load(npz)
        th = dd['thetas']
        sl = slice(None), 0                      # phi = 0 scattering-plane slice
        f0 = np.abs(dd['s_add'][:, :, 0][sl])
        m0 = np.abs(dd['s_mod'][:, :, 0][sl])
        g0 = np.abs(dd['g'][:, :, 0][sl])
        axf, axg, axb = axes[r]
        c = th <= 40
        axf.semilogy(th[c], f0[c], lw=1.3, color='tab:blue', label='ADDA full field')
        axf.semilogy(th[c], m0[c], lw=1.3, ls='--', color='tab:orange',
                     label='eikonal model (calibrated)')
        axf.set_xlim(0, 40)
        axf.set_xlabel(r'$\theta$ [deg]')
        if r == 0:
            axf.set_ylabel(r'$|s_{11}(\theta)|$  (fwd, log)')
        axf.legend(fontsize=6.5, framealpha=0.7)
        axf.set_title('%s: peak shape captured?' % rtitle, fontsize=9)
        axg.semilogy(th, f0, lw=1.1, color='tab:blue', label='full |f|')
        axg.semilogy(th, g0, lw=1.1, color='tab:red', label='residual |g|=|f−F|')
        axg.set_xlim(0, 180)
        axg.set_xlabel(r'$\theta$ [deg]')
        if r == 0:
            axg.set_ylabel(r'$|s_{11}(\theta)|$  (sphere, log)')
        axg.set_title(r'whole sphere: $\,$ $|g|/|f|$ (L2) = %.3f'
                      % float(dd['ratio_total']), fontsize=9)
        axg.legend(fontsize=6.5, framealpha=0.7)
        vals = [int(dd['ntrunc_full']), int(dd['ntrunc_res']),
                int(dd['ntrunc_res_fulleps'])]
        lab = ['full\nfield', 'residual\nown-eps', 'residual\nfull-eps']
        colr = ['tab:blue', '0.6', 'tab:green']
        axb.barh(range(3), vals, color=colr)
        axb.set_yticks(range(3), lab)
        axb.invert_yaxis()
        axb.set_xlim(0, max(vals) * 1.35)
        axb.set_xlabel('SH ntrunc (production rule)')
        for i, v in enumerate(vals):
            axb.text(v + max(vals) * 0.02, i, str(v), va='center', fontsize=8)
        axb.set_title('spectral order: %s'
                      % ('model HARMFUL — not usable'
                         if float(dd['ratio_total']) > 1.02
                         else ('REDUCED %.1fx' % (vals[0] / vals[2])
                               if vals[2] < vals[0] else 'NOT reduced')),
                      fontsize=9,
                      color='tab:red' if float(dd['ratio_total']) > 1.02
                      else 'k')
    savefig(fig, 'fig15_delta_scaling.png')


def fig_validity_map():
    """When is the model safe? Cext sanity vs residual fraction, per scan cell."""
    d = _p9_results()
    cells = d['cells']
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    mk = {'face': 'o', 't60': 's', 'edge': '^', 'along': 'D'}
    sc = None
    for c in cells:
        xr = c['Cext_model'] / c['Cext_adda']
        yv = c['ratio_total']
        gain = c['ntrunc_res_fulleps'] / max(c['ntrunc_full'], 1)
        mkr = 'o'
        for k_, v in mk.items():
            if c['tag'].endswith(k_):
                mkr = v
        sc = ax.scatter([xr], [yv], s=24 + c['ntrunc_full'] * 1.3, marker=mkr,
                        c=[gain], cmap='RdYlGn_r', vmin=0.2, vmax=1.6,
                        edgecolors='k', linewidths=0.6, zorder=3)
    ax.axhline(1.0, color='k', lw=1.0, ls='--')
    ax.text(0.02, 1.03, r'above: $|g|>|f|$ — model actively HARMFUL',
            fontsize=8)
    ax.axvspan(0.6, 1.5, color='tab:green', alpha=0.08, zorder=0)
    ax.text(1.05, 1.78, 'model Cext plausible\n(0.6–1.5 ×)', ha='center',
            fontsize=8, color='tab:green')
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
    ax.set_ylabel(r'residual fraction $|g|/|f|$')
    ax.set_ylim(-0.65, 2.05)
    ax.set_xlim(-0.7, 2.0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('ntrunc gain  res(full-eps)/full\n(green = reduction)')
    ax.set_title('Phase-9A scan, all 15 cells: marker = orientation '
                 '(circle face,\nsquare 60°, triangle edge, diamond along-column)'
                 '\nsize = full-field ntrunc', fontsize=9.5)
    savefig(fig, 'fig16_validity_map.png')


# ---------------------------------------------------------- Fig 09–12 ---
def fig_baseline_set():
    """Appendix E figures: regenerate the baseline report figures (data:
    frozen golden runs under baselines/adda_1.2/) and mirror them into
    docs/figures as fig09..fig12. Rendering is delegated to the baseline
    tooling itself (baselines/plot_baseline_figs.py) so the appendix can
    never drift from the reference report."""
    import shutil
    import subprocess
    script = os.path.join(REPO, 'baselines', 'plot_baseline_figs.py')
    subprocess.run([sys.executable, script], check=True, cwd=REPO)
    src = os.path.join(REPO, 'baselines', 'report')
    for srcname, dstname in [
            ('fig1_crosssections.png', 'fig09_baseline_crosssections.png'),
            ('fig2_sphere_mie.png', 'fig10_baseline_sphere_mie.png'),
            ('fig3_scattering_maps.png', 'fig11_baseline_scattering_3d.png'),
            ('fig4_ellipsoid_phase.png', 'fig12_baseline_ellipsoid_phase.png')]:
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
    fig_delta_scaling()
    fig_validity_map()
    fig_baseline_set()
    print('done.')
