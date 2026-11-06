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
def fig_orientation_geometry():
    fig = plt.figure(figsize=(10.4, 4.0))
    # panel (a): Euler/ARO/TRO geometry in the lab frame
    ax = fig.add_subplot(121, projection='3d')
    ax.set_axis_off()
    L = 1.0
    # lab z axis (gravity)
    ax.quiver(0, 0, 0, 0, 0, 1.25 * L, color='k', lw=1.5)
    ax.text(0, 0, 1.32 * L, 'lab $z$ (gravity)', fontsize=9)
    # incidence direction: polar theta_inc in the x-z plane
    t_inc = np.deg2rad(115)
    ix, iz = np.sin(t_inc), np.cos(t_inc)
    ax.quiver(0, 0, 0, ix, 0, iz, color='tab:red', lw=1.6)
    ax.text(ix * 1.15, 0, iz * 1.15, r'incidence $\hat{k}$'
            '\n($\\theta_{inc}$ from $z$, azimuth fixed)', fontsize=8, color='tab:red')
    # particle axis tilted by beta (rotated about y so it leaves x-z plane too)
    b = np.deg2rad(55)
    px = np.sin(b) * np.cos(np.deg2rad(35))
    py = np.sin(b) * np.sin(np.deg2rad(35))
    pz = np.cos(b)
    ax.quiver(0, 0, 0, px, py, pz, color='tab:blue', lw=1.6)
    ax.text(px * 1.15, py * 1.15, pz * 1.15, 'particle symmetry axis\n'
            '(tilt $\\beta$ from $z$)', fontsize=8, color='tab:blue')
    # beta arc
    tt = np.linspace(np.deg2rad(55), 0, 30)
    ax.plot(np.sin(tt) * np.cos(np.deg2rad(35)),
            np.sin(tt) * np.sin(np.deg2rad(35)), np.cos(tt), color='tab:blue', lw=1)
    ax.text(0.32, 0.22, 0.95, '$\\beta$', color='tab:blue', fontsize=11)
    tt = np.linspace(np.deg2rad(115), 0, 30)
    ax.plot(np.sin(tt), np.zeros_like(tt), np.cos(tt), color='tab:red', lw=1)
    ax.text(0.42, 0, 0.88, '$\\theta_{inc}$', color='tab:red', fontsize=11)
    # reference sphere: equator + one meridian, light
    u = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(u) * .6, np.sin(u) * .6, np.zeros_like(u), color='0.75', lw=0.6)
    ax.plot(np.zeros_like(u), np.cos(u) * .6, np.sin(u) * .6, color='0.75', lw=0.6)
    ax.set_title('(a) Euler zyz: $R(\\alpha,\\beta,\\gamma)=R_z(\\alpha)R_{z\'}(\\beta)R_{z\'\'}(\\gamma)$\n'
                 'ARO: $\\beta$ = tilt of particle axis, azimuth ($\\alpha$, $\\gamma$) averaged', fontsize=9)
    ax.view_init(elev=12, azim=28)
    ax.set_box_aspect([1, 1, 1])
    # panel (b): scattering grid attached to k
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal'); ax2.axis('off')
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(th), np.sin(th), color='0.4', lw=1.2)
    for a in range(15, 180, 15):  # theta cones seen edge-on
        r = np.cos(np.deg2rad(a))
        ax2.plot(np.cos(th), r * np.sin(th) * .28 + np.sin(th) * (1 - abs(r)) * 0,
                 color='0.8', lw=0.5)
    ax2.annotate('', xy=(1.35, 0), xytext=(-0.05, 0),
                 arrowprops=dict(arrowstyle='->', lw=1.6))
    ax2.text(0.85, 0.07, 'propagation $\\hat{k}$', fontsize=9)
    tta = np.linspace(0, np.deg2rad(120), 30)
    ax2.plot(1.15 * np.cos(tta), 1.15 * np.sin(tta), color='tab:orange', lw=1.4)
    ax2.text(1.02, 0.55, '$\\theta$', color='tab:orange', fontsize=11)
    ph = np.linspace(np.deg2rad(200), np.deg2rad(340), 40)
    ax2.plot(0.45 * np.cos(ph), 0.45 * np.sin(ph) * .3 - 0.42, color='tab:green', lw=1.4)
    ax2.text(0.12, -0.55, '$\\varphi$', color='tab:green', fontsize=11)
    ax2.plot([np.cos(np.deg2rad(120))], [np.sin(np.deg2rad(120))], 'o', ms=6)
    ax2.text(np.cos(np.deg2rad(120)) - .18, np.sin(np.deg2rad(120)) + .05,
             'scattering direction $\\hat{n}(\\theta,\\varphi)$', fontsize=8)
    ax2.text(-1.1, -0.85, 'grid attached to $\\hat{k}$:  $\\theta\\in[0^\\circ,180^\\circ]$ (181 pts),  '
             '$\\varphi\\in[0^\\circ,360^\\circ]$ (361 pts)  —  $\\theta=0°$ is straight ahead', fontsize=8)
    ax2.set_title('(b) the ADDA scattering grid of every solve\n(why the mirror transform only remaps $\\varphi$)', fontsize=9)
    ax2.set_xlim(-1.15, 1.5); ax2.set_ylim(-0.95, 1.3)
    savefig(fig, 'fig08_orientation_geometry.png')


if __name__ == '__main__':
    fig_dielectric()
    fig_size_parameter()
    fig_resolution_scheme()
    fig_adda_cost()
    fig_solve_reduction()
    fig_sym_closure()
    fig_backlog()
    fig_orientation_geometry()
    print('done.')
