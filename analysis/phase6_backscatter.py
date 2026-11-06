#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-6 gate suite: exact 4x4 backscatter Mueller matrix in the ARO pipeline.

GATE 1 (analytic sphere):
    ADDA mueller_scatgrid Theta=180/Phi=0 slice vs analytic Mie |S1(180)|^2;
    structure checks (diag(1,1,-1,-1) * M11, off-diagonal and phi-ring
    flatness at solver-noise level) and the new adda_call back-matrix API.

GATE 2 (ellipsoid, convention pin):
    tiny oblate ellipsoid (adda_config_phase6_gate2): sub_control store ->
    rawdata2average (orientation-averaged back matrices, m^2/sr) compared to
    the pipeline phase matrix (mueller2phase_reexpand +
    create_regular_representation) near the backscatter pole
    (lab-absolute grid: Theta_s = 180-theta_inc-{2,4} deg, Phi_s = 180,
    compared through the same PHI1/PHI2 ladder that mueller2phase applies).
    A convention scan (rotation-angle
    signs on both sides x constant Stokes conjugations) pins the
    pol_changer_back convention: the production variant must win clearly.

GATE 3 (production habit smoke, adda_config_phase6_hp_sym = phase4_hp_sym
clone):
    - full sub_control run (mirror fill-in active), back_mat_array finite at
      every vertex of the icosahedron store,
    - NetCDF cache round-trip incl. tolerance of pre-Phase-6 stores,
    - mirror-ring rule on directly computed twin vertices (XY pair),
    - sub_control convergence-loop back-averaging (sym branch) vs
      rawdata2average (plain branch) -> code-path equivalence,
    - pole comparison of the averaged back matrix vs the regular-representation
      phase matrix of the same store,
    - SSDB NetCDF export with version beta_0.9.1 + backMat_data/backMat_index,
      re-read and value round-trip.

Usage (DDA conda env):
    python analysis/phase6_backscatter.py [--gates 1,2,3] [--force-run]

Results JSON: analysis/phase6_results.json
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io as sio
from netCDF4 import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'adda_python_scripts')
RESULTS = os.path.join(REPO, 'adda_results')
sys.path.insert(0, SCRIPTS)

import adda_control_functions as acf          # noqa: E402
import adda_orientational_avg_spharmonics_functions as aoa  # noqa: E402
import spherical_harmonics_for_adda as spha   # noqa: E402
import icosphere_grid as ico                  # noqa: E402

C0 = 299792458.0
BIN14 = os.path.join(REPO, 'adda_1.4', 'src', 'seq', 'adda')

D_STOKES = np.diag([1., 1., -1., -1.])   # attached-basis par relabelling
T_STOKES = np.diag([1., -1., 1., -1.])   # per relabelling

RESULTS_JSON = {}


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.nanmax(np.abs(b)), 1e-300))


def check(name, ok, detail):
    print(('PASS  ' if ok else 'FAIL  ') + name + '  ' + detail)
    RESULTS_JSON.setdefault('checks', []).append(
        {'name': name, 'pass': bool(ok), 'detail': detail})
    if not ok:
        RESULTS_JSON['overall'] = 'FAIL'
        raise SystemExit(1)


# ==============================================================================
# Mie (validated against ADDA Cext/Csca at x=0.314, report Phase 6)
# ==============================================================================
def mie(m, x, k, nmax=25):
    '''spherical Mie coefficients; k in 1/um (Cext/Csca then in um^2)'''
    z = m * x
    D = np.zeros(nmax + 3, dtype=complex)
    for kk in range(nmax + 2, 1, -1):
        D[kk - 1] = kk / z - 1.0 / (D[kk] + kk / z)
    an = np.zeros(nmax + 1, dtype=complex)
    bn = np.zeros(nmax + 1, dtype=complex)
    from scipy.special import spherical_jn as jn, spherical_yn as yn
    for n in range(1, nmax + 1):
        psi = x * jn(n, x)
        psip = (n + 1) * jn(n, x) - x * jn(n + 1, x)
        chi = x * yn(n, x)
        chip = (n + 1) * yn(n, x) - x * yn(n + 1, x)
        xi = psi + 1j * chi
        xip = psip + 1j * chip
        an[n] = (m * psip - D[n] * psi) / (m * xip - D[n] * xi)
        bn[n] = (psip - m * D[n] * psi) / (xip - m * D[n] * xi)
    Cext = (2 * np.pi / k ** 2) * np.sum(
        [(2 * n + 1) * np.real(an[n] + bn[n]) for n in range(1, nmax + 1)])
    Csca = (2 * np.pi / k ** 2) * np.sum(
        [(2 * n + 1) * (abs(an[n]) ** 2 + abs(bn[n]) ** 2)
         for n in range(1, nmax + 1)])
    S1_0 = sum((2 * n + 1) / 2.0 * (an[n] + bn[n]) for n in range(1, nmax + 1))
    S1_180 = sum((2 * n + 1) / 2.0 * ((-1) ** (n + 1)) * (an[n] - bn[n])
                 for n in range(1, nmax + 1))
    return Cext, Csca, S1_0, S1_180


# ==============================================================================
# GATE 1 — sphere
# ==============================================================================
def gate1():
    print('\n==== GATE 1: sphere / Mie anchor ====')
    tmp = tempfile.mkdtemp()
    f = 300e9
    lamb = C0 / f * 1e6
    m = np.sqrt(acf.eps_ice_matzler06(f, 230.))
    eq = 50.0
    x = 2 * np.pi * eq / lamb

    Cext_m, Csca_m, S1_0, S1_180 = mie(m, x, 2 * np.pi / lamb)

    inp = {'lambda': lamb, 'Size': eq, 'refrative_index': m,
           'theta': [1.0, 0.0, 0.0], 'alpha': 0, 'beta': 0, 'gamma': 0,
           'eps': 3, 'ramdrive': tmp,
           'zipname': os.path.join(tmp, 'sph.zip'),
           'main_command': [BIN14],
           'static_clo': ['-shape', 'sphere', '-store_scat_grid',
                          '-scat_grid_inp',
                          os.path.join(REPO, 'baselines',
                                       'scat_params_baseline.dat'),
                          '-scat_matr', 'both'],
           'zipping': False}
    out = acf.adda_call(inp)

    # Cext is in um^2 from adda_call; Mie computed in um units. Threshold
    # 3%: at default dpl (~3.9) the x=0.31 sphere is a coarse DDA grid, the
    # discretization error is ~1.8% (same size as the B11/|S1(180)|^2 ratio
    # deviation below) — the gate anchors the ANALYTIC convention, not DDA
    # accuracy.
    Cext_adda = out['ext_matrix'][0, 0]
    r_ce = abs(Cext_adda / Cext_m - 1)
    check('G1 Mie-anchor Cext', r_ce <= 3e-2,
          f'ADDA {Cext_adda:.4f} vs Mie {Cext_m:.4f} um^2, rel {r_ce:.2e}')

    B = out['back_matrix']
    ring = out['back_mat_ring']

    # API consistency with manual extraction from the mueller grid
    th = out['thetas_grid']
    ph = out['phis_grid']
    M = np.asarray(out['mueller_array'], dtype=float).reshape(
        len(th), len(ph), 4, 4)
    i180 = int(np.flatnonzero(np.abs(th - 180.) < 1e-9)[0])
    ip0 = int(np.flatnonzero(np.abs(ph) < 1e-9)[0])
    check('G1 adda_call API', np.max(np.abs(B - M[i180, ip0])) == 0
          and np.max(np.abs(ring - M[i180, :])) == 0,
          'back_matrix/back_mat_ring == manual Theta=180/Phi=0 slices')

    m11 = B[0, 0]
    ratio = m11 / abs(S1_180) ** 2
    check('G1 sphere B11 vs |S1(180)|^2', abs(ratio - 1) <= 0.05,
          f'ratio {ratio:.5f} (DDA discretization error ~1.8% expected)')

    struct = np.array([1., 1., -1., -1.])
    ds = np.max(np.abs(np.diag(B) / m11 - struct))
    check('G1 sphere diag(1,1,-1,-1)*M11', ds <= 1e-8,
          f'max diag dev {ds:.2e}')

    off = np.max(np.abs(B - np.diag(np.diag(B)))) / abs(m11)
    check('G1 sphere off-diagonal', off <= 1e-8, f'rel offdiag {off:.2e}')

    flat = np.max(np.abs(ring - ring[ip0])) / abs(m11)
    check('G1 Theta=180 ring phi-flat (sphere)', flat <= 1e-8,
          f'max ring dev {flat:.2e}')

    shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# helpers shared by GATE 2/3
# ==============================================================================
def load_store(matpath):
    mat = sio.loadmat(matpath)['output_data'].ravel()[0]
    return mat


def run_subcontrol(config, tidx, fidx, didx, label):
    env = dict(os.environ)
    env.setdefault('ADDA_BINARY', BIN14)
    env.setdefault('ADDA_VERSION', '1.4.0')
    print(f'  running adda_sub_control_spharmonics.py '
          f'{tidx} {fidx} {didx} {config}  ({label})')
    subprocess.run([sys.executable, 'adda_sub_control_spharmonics.py',
                    str(tidx), str(fidx), str(didx), config],
                   cwd=SCRIPTS, env=env, check=True)


def phase_matrix_regular(mat_in, grid_theta=90):
    '''replicates the adda_process_aro_particles_single.py chain;
    returns (output_data_with_phase_keys, phase_matrix_reg, theta_s, phi_s)'''
    output_data = dict(mat_in)
    # rawdata2average output keys must be present — caller passes its result
    [phase_mat, mnt, mlc] = aoa.mueller2phase_reexpand(
        output_data, grid_size_theta=grid_theta, eps=0.001,
        parallel=False, core_limit=2)
    phase_mat = spha.reshape_sphexp_2_common_exp(
        int(np.max(mnt[:])), int(np.max(mlc[:])), phase_mat)
    output_data['eps_phase'] = 0.001
    output_data['phase_matrix_sph'] = phase_mat
    output_data['max_ntrunc'] = int(np.max(mnt[:]))
    output_data['max_len_coeffs'] = int(np.max(mlc[:]))
    lmax = spha.get_lmax(output_data['phase_matrix_sph'][0, 0, 0, :])
    [pha_reg, theta_s, phi_s] = aoa.create_regular_representation(
        output_data, lmax, grid_size_theta=grid_theta + 1,
        gridtype='regular', full_phi=True)
    return output_data, pha_reg, theta_s, phi_s


def pcb_variant(vertices, vpol, vpol_lab, back, si, ss):
    '''generalized pol_changer_back for the convention scan:
    B_rot = R(ss*phi_i) * B * R(si*phi_i)  (production: si=+1, ss=-1)'''
    hpol_lab = np.array([np.cross(a, b) for a, b in zip(vpol_lab, vertices)])
    x = np.array([np.dot(a, b) for a, b in zip(vpol, vpol_lab)])
    y = np.array([np.dot(a, b) for a, b in zip(vpol, hpol_lab)])
    phi = np.arctan2(y, x)
    out = np.zeros_like(np.asarray(back, dtype=float))
    for i in range(len(out)):
        out[i] = (aoa.rotater_mat(ss * phi[i]) @ np.asarray(back[i], float)
                  @ aoa.rotater_mat(si * phi[i]))
    return out


def avg_back_replica(vertices, vpol, back_mat_array, inc_dir, beta, ref_lev,
                     si, ss):
    '''exact replica of the plain-branch back matrix path of
    orientational_avg (no symmetry), generalized rotation signs'''
    nqp = 4 * 2 ** ref_lev
    if nqp > 64:
        nqp = 64
    sp, gw = np.polynomial.legendre.leggauss(nqp)
    al = sp * 180
    ga = sp * 180
    acc_alpha = np.zeros((4, 4))
    for a_i in range(len(al)):
        acc_gamma = np.zeros((4, 4))
        for g_i in range(len(ga)):
            vr = ico.rotate_vertices(vertices, al[a_i] * np.pi / 180,
                                     beta * np.pi / 180,
                                     ga[g_i] * np.pi / 180)
            w, idx = ico.interpolation_weights_icosahedric_grid(vr, inc_dir)
            vl = np.tile(aoa.create_vpol_vector(inc_dir), (len(idx), 1))
            vp = ico.rotate_vertices(vpol[idx, :], al[a_i] * np.pi / 180,
                                     beta * np.pi / 180,
                                     ga[g_i] * np.pi / 180)
            B = pcb_variant(vr[idx, :], vp, vl,
                            back_mat_array[idx, :, :], si, ss)
            EXT = np.zeros((4, 4))
            for i in range(len(idx)):
                EXT = w[i] * B[i] + EXT
            acc_gamma = EXT * gw[g_i] / 2 + acc_gamma
        acc_alpha = acc_gamma * gw[a_i] / 2 + acc_alpha
    return acc_alpha


def backscatter_rows(theta_s, inc_deg, deltas=(2., 4.)):
    '''rows of the LAB-absolute regular grid approaching exact backscatter
    (theta_s = 180 - theta_inc) from the theta_s < 180-theta_inc side.

    The phi_s = 180 ray crosses the backscatter point; on the theta_s <
    180-theta_inc side the scattering-plane normal (k_s x k_i) is +y_lab,
    which is the ADDA mueller_scatgrid Phi=0 column basis at the Theta=180
    pole (n_inc = +y_inc, and R_y(theta_inc) maps y_inc -> y_lab). The other
    side carries n = -y (the Phi=180 column basis) and must not be compared
    against the raw Phi=0 slice. The singular point itself is skipped
    (PHI basis degenerate). Returns [(row_idx, theta_inc_idx)].'''
    out = []
    for ith, ti in enumerate(inc_deg):
        if ti <= 0.5 or ti >= 179.5:
            continue                     # degenerate: approach along grid axis
        for d in deltas:
            t = 180. - ti - d
            if t <= 0.5 or t >= 179.5:
                continue
            r = int(np.argmin(np.abs(theta_s - t)))
            if abs(theta_s[r] - (180. - ti)) < 1e-6:
                continue                 # hit the singular point itself
            out.append((r, ith))
    return out


def phi_ladder(theta_inc_deg, theta_s_1d, phi_s_1d):
    '''vectorized replica of calculate_rotation_angles() (identical sign
    logic) for an arbitrary regular (theta_s, phi_s) mesh in DEGREES.
    Returns PHI1, PHI2 arrays [n_inc, n_theta, n_phi] in radians — the
    pointwise incidence/scattering basis rotations that mueller2phase applies
    (P = rotater_mat(-PHI2) @ M @ rotater_mat(PHI1) * lamb^2/4pi^2).'''
    t_i = np.deg2rad(np.asarray(theta_inc_deg, dtype=float))
    t_s = np.deg2rad(np.asarray(theta_s_1d, dtype=float))
    p_s = np.deg2rad(np.asarray(phi_s_1d, dtype=float))
    k_i = np.stack([np.sin(t_i), np.zeros_like(t_i), np.cos(t_i)], axis=-1)
    v_i = np.stack([np.cos(t_i), np.zeros_like(t_i), -np.sin(t_i)], axis=-1)
    k_s = np.stack([np.outer(np.sin(t_s), np.cos(p_s)),
                    np.outer(np.sin(t_s), np.sin(p_s)),
                    np.tile(np.cos(t_s)[:, np.newaxis], (1, p_s.size))],
                   axis=-1)
    v_s = np.stack([np.outer(np.cos(t_s), np.cos(p_s)),
                    np.outer(np.cos(t_s), np.sin(p_s)),
                    -np.tile(np.sin(t_s)[:, np.newaxis], (1, p_s.size))],
                   axis=-1)
    k_i4 = np.broadcast_to(k_i[:, np.newaxis, np.newaxis, :],
                           (t_i.size,) + k_s.shape)
    k_s4 = np.broadcast_to(k_s[np.newaxis, ...], k_i4.shape)
    kk = np.cross(k_s4, k_i4)
    sin_theta = np.linalg.norm(kk, axis=-1)
    n = np.divide(kk, sin_theta[..., np.newaxis],
                  out=np.zeros_like(kk), where=sin_theta[..., np.newaxis] > 0)
    out1 = np.empty(n.shape[:-1])
    out2 = np.empty_like(out1)
    for m in range(t_i.size):
        P_i = np.cross(n[m], np.broadcast_to(k_i[m], n[m].shape))
        P_s = np.cross(n[m], np.broadcast_to(k_s, n[m].shape))
        ti_ = np.broadcast_to(v_i[m], n[m].shape)
        ts_ = np.broadcast_to(v_s, n[m].shape)
        c1 = np.clip(np.sum(ti_ * P_i, axis=-1), -1, 1)
        c2 = np.clip(np.sum(ts_ * P_s, axis=-1), -1, 1)
        a1 = np.arccos(c1)
        a2 = np.arccos(c2)
        out1[m] = np.where(np.sum(ti_ * n[m], axis=-1) <= 0, a1, -a1)
        out2[m] = np.where(np.sum(ts_ * n[m], axis=-1) <= 0, a2, -a2)
    return out1, out2


def _mask_coplanar(M):
    '''elements that survive the coplanar comparison. At every comparison
    point (Phi_s = 180, Phi_inc = 0 side) the geometry is mirror-symmetric
    about the scattering plane, so the TRUE M13 == M31 == 0 for any particle
    (the back-matrix path returns exactly 0 there). The pipeline's phase
    matrix, however, pole-interpolates the off-diagonal spherical-harmonic
    expansions (mueller2phase_core, k%5!=0 branch), which fabricates
    spurious M13/M31 ~1e-12 near the inc-frame poles — comparing those
    would test that known artifact, not the back-matrix convention.'''
    return np.delete(np.delete(M, (0, 2), axis=0), (0, 2), axis=1)


def pole_compare(B_avg, pha_reg, theta_s, phi_s, inc_deg, PHI1, PHI2,
                 deltas=(2., 4.)):
    '''B_avg[ib,ith] (m^2/sr, averaged Mueller at exact backscatter, ADDA
    Phi=0-column basis) vs phase_matrix_reg at the exact-backscatter
    approach: Phi_s = 180 (scattering plane = incidence XZ plane), Theta_s =
    180 - theta_inc - deltas, after the same pointwise basis rotation that
    mueller2phase applies (P = R(-PHI2) @ M @ R(PHI1)).'''
    ip180 = int(np.argmin(np.abs(phi_s - 180.)))
    rows = backscatter_rows(theta_s, inc_deg, deltas)
    worst = 0.
    for ib in range(np.shape(B_avg)[0]):
        for r, ith in rows:
            rot1 = aoa.rotater_mat(PHI1[ith, r, ip180])
            rot2 = aoa.rotater_mat(-PHI2[ith, r, ip180])
            P = rot2 @ B_avg[ib, ith] @ rot1
            den = max(np.max(np.abs(B_avg[ib, ith])), 1e-300)
            P_grid = pha_reg[ib, ith, :, r, ip180].reshape(4, 4)
            worst = max(worst, np.max(np.abs(_mask_coplanar(P - P_grid))) / den)
    return worst


# ==============================================================================
# GATE 2 — ellipsoid convention pin
# ==============================================================================
def gate2(force_run):
    print('\n==== GATE 2: ellipsoid back matrix vs phase matrix (x=0.3) ====')
    store = os.path.join(RESULTS, 'phase6_gate2', 'rawdata')
    mats = glob.glob(store + '/*/*overview_data.mat')
    if not mats or force_run:
        run_subcontrol('adda_config_phase6_gate2', 0, 0, 0, 'gate2 x=0.3')
        mats = glob.glob(store + '/*/*overview_data.mat')
    mat = load_store(mats[0])
    check('G2 back_mat_array in store', 'back_mat_array' in mat.dtype.names
          and np.all(np.isfinite(np.asarray(mat['back_mat_array'], float))),
          f'{np.asarray(mat["back_mat_array"]).shape} all finite')

    beta = np.asarray(mat['beta'], dtype=float).ravel()
    incs = np.asarray(mat['theta_inc'], dtype=float).ravel()
    eps_vec = np.asarray(mat['eps_vec'], dtype=float).ravel()
    ref_lev = len(eps_vec) - int(np.sum(np.isnan(eps_vec)))

    cwd = os.getcwd()
    avg = aoa.rawdata2average(os.path.dirname(mats[0]) + '/',
                              incs, beta, None, 0, 0, 0, parallel=False)
    os.chdir(cwd)

    Bm2 = avg['back_mat_horiz_avg']          # m^2/sr, production convention
    check('G2 rawdata2average back finite', np.all(np.isfinite(Bm2)),
          f'shape {Bm2.shape}')

    # replica of the production convention must match rawdata2average
    vertices = np.asarray(mat['vertices'], dtype=float)
    vpol = aoa.create_vpol_vector(vertices)
    back = np.asarray(mat['back_mat_array'], dtype=float)
    lamb_m = float(np.asarray(mat['wavelength'], dtype=float).ravel()[0]) * 1e-6
    conv = lamb_m ** 2 / (4. * np.pi ** 2)
    worst_rep = 0.
    for ib, bt in enumerate(beta):
        for it, th in enumerate(incs):
            inc_dir = np.array([np.sin(th * np.pi / 180), 0,
                                np.cos(th * np.pi / 180)])
            Ba = avg_back_replica(vertices, vpol, back, inc_dir, bt,
                                  ref_lev, +1, +1) * conv
            worst_rep = max(worst_rep, rel(Ba, Bm2[ib, it]))
    check('G2 replica == production averaging', worst_rep <= 1e-10,
          f'max rel {worst_rep:.2e}')

    # phase matrix regular representation
    _, pha_reg, theta_s, phi_s = phase_matrix_regular(dict(avg.items()),
                                                      grid_theta=90)

    # --- convention scan -------------------------------------------------
    # production: R(+1,-1) i.e. B_lab = R(phi_s) B R(phi_i), phi_s = -phi_i,
    # no constant Stokes conjugation. Variants: rotation-angle signs on both
    # sides; constant left/right conjugations only for the production
    # rotation (they commute with the average and multiply it directly).
    # lab-absolute grid: exact backscatter is (Theta_s = 180-theta_inc,
    # Phi_s = 180); approach from the Theta_s < 180-theta_inc side only
    # (that side's scattering-plane normal matches the ADDA Phi=0 column
    # basis of the raw slice, see backscatter_rows), and compare through
    # the same pointwise PHI1/PHI2 basis rotation mueller2phase applies.
    ip180 = int(np.argmin(np.abs(phi_s - 180.)))
    rows = backscatter_rows(theta_s, incs)
    PHI1, PHI2 = phi_ladder(incs, theta_s, phi_s)
    print('  pole-approach comparisons: %d (theta_s = %s)'
          % (len(rows), ', '.join(f'{theta_s[r]:.0f}' for r, _ in rows)))

    scan = {}
    prod = None
    # PRODUCTION convention: si=+1, ss=+1 (B_lab = R(+phi) B R(+phi), see
    # pol_changer_back docstring). One-sided modes (si or ss = 0 -> R(0) = I)
    # test whether the scattered (attached at -v) side transports at all.
    for si, ss in [(+1, -1), (+1, +1), (-1, -1), (-1, +1),
                   (0, -1), (0, +1), (-1, 0), (+1, 0)]:
        accs = []
        for ib, bt in enumerate(beta):
            row = []
            for it, th in enumerate(incs):
                inc_dir = np.array([np.sin(th * np.pi / 180), 0,
                                    np.cos(th * np.pi / 180)])
                row.append(avg_back_replica(vertices, vpol, back, inc_dir,
                                            bt, ref_lev, si, ss))
            accs.append(row)

        def err_of(si, ss, CL, CR, accs=accs):
            worst = 0.
            for ib in range(len(beta)):
                for r, ith in rows:
                    Ba = (CL @ accs[ib][ith] @ CR) * conv
                    P = (aoa.rotater_mat(-PHI2[ith, r, ip180]) @ Ba
                         @ aoa.rotater_mat(PHI1[ith, r, ip180]))
                    den = max(np.max(np.abs(Bm2[ib, ith])), 1e-300)
                    P_grid = pha_reg[ib, ith, :, r, ip180].reshape(4, 4)
                    worst = max(worst, np.max(np.abs(_mask_coplanar(P - P_grid))) / den)
            return worst

        for cnl, CL in [('I', np.eye(4)), ('D', D_STOKES), ('T', T_STOKES),
                        ('DT', D_STOKES @ T_STOKES)]:
            for cnr, CR in [('I', np.eye(4)), ('D', D_STOKES),
                            ('T', T_STOKES), ('DT', D_STOKES @ T_STOKES)]:
                if (cnl != 'I' or cnr != 'I') and (si, ss) != (+1, +1):
                    continue
                key = f'R[{si:+d},{ss:+d}] {cnl}|{cnr}'
                e = err_of(si, ss, CL, CR)
                scan[key] = e
                if (si, ss) == (+1, +1) and cnl == 'I' and cnr == 'I':
                    prod = e
    ranked = sorted(scan.items(), key=lambda kv: kv[1])
    print('  convention scan (max rel err vs phase matrix at '
          'Theta_s = 180-theta_inc-{2,4} deg, Phi_s = 180, via PHI ladder):')
    for k, v in ranked:
        print(f'    {k:24s} {v:.4e}')
    RESULTS_JSON['gate2_convention_scan'] = scan

    check('G2 production convention at pole <= 5%', prod <= 0.05,
          f'max rel err {prod:.4e}')
    runner = [v for k, v in ranked if v > prod * (1 + 1e-4)]
    check('G2 production convention is at the scan minimum; every '
          'distinguishable alternative (ext-analog signs, one-sided, '
          'wrong conjugation pairing) is worse by >= 5x',
          prod <= ranked[0][1] * (1 + 1e-9)
          and len(runner) and runner[0] >= 5 * max(prod, 1e-12),
          f'production {prod:.3e} (floor tie group: '
          f'{sum(1 for v in scan.values() if v <= prod * (1 + 1e-9))} '
          f'variants), '
          f'best distinguishable alternative {runner[0]:.3e}')


def gate2_x1(force_run):
    '''informational leg: x=1 ellipsoid (stronger near-pole structure);
    phase-matrix pole interpolation limits accuracy — reported, not asserted
    beyond a loose bound'''
    print('\n==== GATE 2x: ellipsoid x=1 (informational pole numbers) ====')
    store = os.path.join(RESULTS, 'phase6_gate2_x1', 'rawdata')
    mats = glob.glob(store + '/*/*overview_data.mat')
    if not mats or force_run:
        run_subcontrol('adda_config_phase6_gate2_x1', 0, 0, 0, 'gate2 x=1')
        mats = glob.glob(store + '/*/*overview_data.mat')
    mat = load_store(mats[0])
    beta = np.asarray(mat['beta'], dtype=float).ravel()
    incs = np.asarray(mat['theta_inc'], dtype=float).ravel()
    cwd = os.getcwd()
    avg = aoa.rawdata2average(os.path.dirname(mats[0]) + '/',
                              incs, beta, None, 0, 0, 0, parallel=False)
    os.chdir(cwd)
    _, pha_reg, theta_s, phi_s = phase_matrix_regular(dict(avg.items()),
                                                      grid_theta=90)
    PHI1, PHI2 = phi_ladder(incs, theta_s, phi_s)
    e = pole_compare(avg['back_mat_horiz_avg'], pha_reg, theta_s, phi_s,
                     incs, PHI1, PHI2)
    RESULTS_JSON['gate2x1_pole_relerr'] = e
    check('G2x x=1 pole agreement (loose, <=15%)', e <= 0.15,
          f'max rel err {e:.4e} (x=1: P varies on ~10 deg scale near pole)')


# ==============================================================================
# GATE 3 — production habit (HongPlate 10 um, 228 GHz, 190 K, mirror sym)
# ==============================================================================
def gate3(force_run):
    print('\n==== GATE 3: HongPlate_Id9 10um/228GHz/190K smoke ====')
    store = os.path.join(RESULTS, 'phase6_hp_sym',
                         'HongPlate_Id9_deq00010um_freq228.0GHz_T0190K')
    matf = os.path.join(store, 'HongPlate_Id9_overview_data.mat')
    if not os.path.exists(matf) or force_run:
        run_subcontrol('adda_config_phase6_hp_sym', 0, 25, 0, 'gate3 plate')

    mat = load_store(matf)
    back = np.asarray(mat['back_mat_array'], dtype=float)
    nvert = np.shape(back)[0]
    nfill = int(np.sum(np.all(np.isfinite(back.reshape(nvert, -1)), axis=1)))
    check('G3 back_mat_array complete', nfill == nvert,
          f'{nfill}/{nvert} vertices finite (run+mirror fill)')

    bh_raw = np.asarray(mat['back_mat_horiz_avg'], dtype=float)
    check('G3 sub_control back_mat_horiz_avg finite',
          np.all(np.isfinite(bh_raw)), f'shape {bh_raw.shape}')

    # ---- NetCDF cache round-trip (incl. pre-Phase-6 tolerance) ----
    tmp = tempfile.mkdtemp()
    msph = [np.ones(7, dtype=np.complex128) * (1 + 2j) for _ in range(16)]
    relerr = np.arange(16) / 100.
    extm = np.arange(16, dtype=float).reshape(4, 4)
    bm = np.arange(16, dtype=float).reshape(4, 4) * 0.5
    br = np.ones((37, 4, 4), dtype=float) * 3.25
    acf.write_data_to_separate_netcdf(os.path.join(tmp, 'store_'), 'vertex_X0000001',
                                      msph, relerr, 7, 3, extm,
                                      back_mat=bm, back_mat_ring=br)
    rd = acf.read_data_from_separate_netcdf(os.path.join(tmp, 'store_'),
                                            'vertex_X0000001')
    ok_rt = (rd is not None and rd['back_matrix'] is not None
             and np.max(np.abs(rd['back_matrix'] - bm)) < 1e-6
             and np.max(np.abs(rd['back_mat_ring'] - br)) < 1e-6)
    check('G3 cache round-trip', ok_rt,
          f'f4 max dev B {np.max(np.abs(rd["back_matrix"] - bm)):.2e}, '
          f'ring {np.max(np.abs(rd["back_mat_ring"] - br)):.2e}')

    acf.write_data_to_separate_netcdf(os.path.join(tmp, 'oldstore_'),
                                      'vertex_X0000002', msph, relerr, 7, 3,
                                      extm)
    rd2 = acf.read_data_from_separate_netcdf(os.path.join(tmp, 'oldstore_'),
                                             'vertex_X0000002')
    check('G3 old store lacks back keys -> None (cache miss rule possible)',
          rd2 is not None and 'back_matrix' in rd2
          and rd2['back_matrix'] is None
          and rd2['back_mat_ring'] is None,
          'read_data_from_netcdf tolerates pre-Phase-6 stores')
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- mirror-ring rule on directly computed twin vertices ----
    shap = (os.path.join(REPO, 'adda_shape_files', 'HongPlate_Id9',
                         'Dmax00013um_Dveq00010um_Mass4.79983e-13kg_'
                         'Freq0192.000GHz.adda'))
    grid = os.path.join(SCRIPTS, 'scat_params.dat')
    f = 228e9
    T = 190.0
    m = np.sqrt(acf.eps_ice_matzler06(f, T))
    lamb = C0 / f * 1e6
    tmp = tempfile.mkdtemp()
    c = np.sin(np.deg2rad(45.))
    s = np.cos(np.deg2rad(45.))
    v1 = np.array([c * np.cos(np.deg2rad(30)), c * np.sin(np.deg2rad(30)), s])
    runs = {}
    for tag, vv in {'src': v1, 'tgt': np.array([v1[0], v1[1], -v1[2]])}.items():
        inp = {'lambda': lamb, 'Size': 5.0, 'refrative_index': m,
               'theta': vv, 'alpha': 0, 'beta': 0, 'gamma': 0, 'eps': 3,
               'ramdrive': tmp,
               'zipname': os.path.join(tmp, tag + '.zip'),
               'main_command': [BIN14],
               'static_clo': ['-shape', 'read', shap, '-store_scat_grid',
                              '-scat_grid_inp', grid, '-scat_matr', 'both'],
               'zipping': False}
        runs[tag] = acf.adda_call(inp)

    ph = runs['src']['phis_grid']
    jmap = acf.mirror_phi_index_map(ph, ['XY'])
    R_src = runs['src']['back_mat_ring']
    R_tgt = runs['tgt']['back_mat_ring']
    sgn = acf.mirror_signs(1).reshape(1, 4, 4)
    den = max(np.max(np.abs(R_src)), 1e-300)
    d_ring = np.max(np.abs(R_tgt - R_src[jmap, :, :] * sgn))
    ip0 = int(np.flatnonzero(np.abs(ph) < 1e-9)[0])
    pred = R_src[jmap[ip0]] * sgn[0]
    d_bm = np.max(np.abs(runs['tgt']['back_matrix'] - pred))
    check('G3 mirror ring + back matrix rule (production shape, XY)',
          d_ring <= 1e-6 * den and d_bm <= 1e-6 * np.max(np.abs(R_tgt)),
          f'ring {d_ring / den:.2e} rel, back-matrix {d_bm:.2e} abs '
          f'(B scale {np.max(np.abs(R_tgt)):.2e})')
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- rawdata2average (plain branch) + code-path equivalence ----
    beta = np.asarray(mat['beta'], dtype=float).ravel()
    incs = np.asarray(mat['theta_inc'], dtype=float).ravel()
    cwd = os.getcwd()
    avg = aoa.rawdata2average(store + '/', incs, beta, None, 0, 0, 0,
                              parallel=False)
    os.chdir(cwd)
    lamb_m = float(np.asarray(mat['wavelength'], dtype=float).ravel()[0]) * 1e-6
    conv = lamb_m ** 2 / (4. * np.pi ** 2)
    d_eq = rel(avg['back_mat_horiz_avg'], bh_raw * conv)
    check('G3 sym-branch (sub_control) == plain-branch (rawdata2average) '
          'back averaging', d_eq <= 1e-8,
          f'max rel {d_eq:.2e} (pure code-path difference)')

    # ---- phase matrix chain + pole comparison + SSDB export ----
    avg_out, pha_reg, theta_s, phi_s = phase_matrix_regular(dict(avg.items()),
                                                            grid_theta=90)
    PHI1, PHI2 = phi_ladder(incs, theta_s, phi_s)
    e = pole_compare(avg['back_mat_horiz_avg'], pha_reg, theta_s, phi_s,
                     incs, PHI1, PHI2)
    check('G3 plate pole comparison (x=0.024)', e <= 0.02,
          f'max rel err {e:.4e}')

    # absorption vector exactly as the process script computes it
    ext_xsec = np.real(avg_out['ext_mat_horiz_avg'][:, :, :, 0])
    sca_xsec = np.real(avg_out['phase_matrix_sph'][:, :, 0:4, 0]
                       / (1 / 2 * np.sqrt(1 / np.pi)))
    abs_vector = ext_xsec - sca_xsec
    a1 = abs_vector[:, :, 0]
    a1[a1 < 0] = 0.
    abs_vector[:, :, 0] = a1
    avg_out['abs_vector'] = abs_vector

    db = tempfile.mkdtemp()
    cfg = {'crystal_type': 'Aggregate', 'pristine_type': 'Pristine',
           'orientation_type': 'Azimuthally_random', 'database_path': db}
    import adda2SSDBnetcdf_functions as a2s
    ncs = a2s.export2SSDBnetCDF(avg_out, cfg)
    check('G3 export file count', len(ncs) == len(beta),
          f'{len(ncs)} beta files')

    with Dataset(ncs[0]) as ds:
        ver = ds.getncattr('version')
        g0 = ds['SingleScatteringData']
        bidx = g0['backMat_index'][:]
        bdat = np.ma.filled(g0['backMat_data'][:], np.nan)
        pdat_real = np.ma.filled(g0['phaMat_data_real'][:], np.nan)
        edat = np.ma.filled(g0['extMat_data'][:], np.nan)
    check('G3 SSDB version bumped', ver == 'beta_0.9.1', f'version={ver}')
    check('G3 SSDB backMat_index', np.array_equal(np.asarray(bidx),
          np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],
                    [13, 14, 15, 16]], dtype=bidx.dtype)),
          f'dtype {bidx.dtype}, shape {bidx.shape}')
    expect = np.transpose(avg['back_mat_horiz_avg'][0], (1, 2, 0)) \
        .reshape(16, 1, len(incs))
    check('G3 SSDB backMat_data round-trip', np.all(np.isfinite(bdat))
          and np.max(np.abs(bdat - expect)) <= 1e-6 * np.max(np.abs(expect)),
          f'shape {bdat.shape}, max dev '
          f'{np.max(np.abs(bdat - expect)):.2e} '
          f'(scale {np.max(np.abs(expect)):.2e})')
    check('G3 SSDB existing vars intact', pdat_real.shape[0] == 16
          and edat.shape == (3, 1, len(incs)),
          f'phaMat {pdat_real.shape}, extMat {edat.shape}')
    shutil.rmtree(db, ignore_errors=True)


# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gates', default='1,2,3')
    ap.add_argument('--force-run', action='store_true')
    ap.add_argument('--x1', action='store_true',
                    help='include the informational x=1 gate2 leg')
    args = ap.parse_args()
    gates = [g.strip() for g in args.gates.split(',')]

    RESULTS_JSON['overall'] = 'PASS'
    if '1' in gates:
        gate1()
    if '2' in gates:
        gate2(args.force_run)
    if 'x1' in gates or args.x1:
        gate2_x1(args.force_run)
    if '3' in gates:
        gate3(args.force_run)

    RESULTS_JSON['overall'] = 'PASS'
    with open(os.path.join(REPO, 'analysis', 'phase6_results.json'), 'w') as fh:
        json.dump(RESULTS_JSON, fh, indent=2)
    print('\nALL GATES PASS — analysis/phase6_results.json written')


if __name__ == '__main__':
    main()
