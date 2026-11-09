#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 8 export twin gate: x-dependent SSDB phase-matrix truncation
(export_sph_margin in adda_process_aro_particles_single.py).

Runs adda_process_single_AroParticle twice on the SAME probe store
(phase8_hp_adaptive, HongPlate_Id9 10um @228GHz/T190K, adaptive 22x42
scat grid) — once with the historic worst-cell padding
(export_sph_margin=None) and once with the x-dependent cap
(margin 10 -> ntrunc = ceil(10*(x+4x^(1/3)+2))) — writing to separate
avg-data and SSDB-database trees.

Gates:
  * overlap coefficients (l <= cap) bitwise identical between the two
    stores (truncation must not touch kept content);
  * per (beta, theta_inc, element) L2 norm of the REMOVED tail bounded by
        eps_phase * |int M11 dOmega|(cell)  +  1.2*sqrt(n_removed) * N_store
    where N_store is the store's own processing-noise plateau: the max
    |coefficient| over ALL cells/elements in the l-band [0.78*Lfull, Lfull]
    of the UNCAPPED store (physical content cannot reach that band at any
    production x <= 10 — Wiscombe limit ~21 there — so those coefficients
    are pure mueller->phase transform noise). Normalizing per-element
    against the element's OWN norm is meaningless: off-diagonal /
    difference channels (M22-M11, M33...) vanish in the small-x Rayleigh
    limit and are dominated by transform noise, so any purely relative
    gate explodes on them (same pitfall as ratio gates in
    phase8_scatgrid_twin); the intensity element k=0 (which carries C_sca,
    g and the angular energy pattern) is protected by the eps_phase *
    scale term on its own;
  * abs_vector + ext_mat_horiz_avg relative difference <= 1e-6;
  * report SSDB netCDF size reduction.

Usage (DDA env, from repo root):
  python analysis/phase8_export_twin.py [--compare-only]
"""

import os
import sys
import glob
import numpy as np
import scipy.io as sio

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..'))
sys.path.insert(0, os.path.join(REPO, 'adda_python_scripts'))

import adda_control_functions as acf
import adda_process_aro_particles_single as aps

RESULTS = os.path.join(REPO, 'adda_results')
PROBE = os.path.join(RESULTS, 'phase8_hp_adaptive')
FOLDER = 'HongPlate_Id9_deq00010um_freq228.0GHz_T0190K'

EPS_PHASE = 0.001
NBETA_OUT = 5      # output beta grid (SSDB output coords, not solve count)
NTHETA_OUT = 5


def build_config(avgpath, dbpath, margin):
    cfg = {}
    cfg['p_orientation_beta'] = np.linspace(0, 180, NBETA_OUT)
    cfg['incidence_angles'] = np.linspace(0, 180, NTHETA_OUT)
    cfg['regular_phasematrix'] = False      # keep phase_matrix_sph in the mat
    cfg['export_netdf'] = True
    cfg['grid_size'] = 181                  # historic re-expansion grid
    cfg['alpha_0'] = 0
    cfg['beta_0'] = 0
    cfg['gamma_0'] = 0
    cfg['rawdatapath'] = PROBE + os.sep
    cfg['avgdatapath'] = avgpath
    cfg['database_path'] = dbpath
    cfg['ncores'] = 4
    cfg['parallel'] = False
    cfg['eps_phase'] = EPS_PHASE
    cfg['export_sph_margin'] = margin
    cfg['crystal_type'] = 'Aggregate'
    cfg['pristine_type'] = 'Pristine'
    cfg['orientation_type'] = 'Azimuthally_random'
    return cfg


def load_avg(tree):
    p = glob.glob(os.path.join(tree, '*', '*_horiz_avg_phasematrix.mat'))[0]
    ov = sio.loadmat(p, squeeze_me=True)['output_data']
    def arr(x):
        if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
            x = x.item()
        return np.asarray(x)
    return {'pha': arr(ov['phase_matrix_sph']).astype(complex),
            'abs': arr(ov['abs_vector']).astype(float),
            'ext': arr(ov['ext_mat_horiz_avg']).astype(float),
            'maxntr': int(ov['max_ntrunc']), 'path': p}


def nc_tree_size(dbpath):
    files = glob.glob(os.path.join(dbpath, '**', '*.nc'), recursive=True)
    return len(files), sum(os.path.getsize(f) for f in files) / 1024.0


def main():
    trees = {tag: (os.path.join(RESULTS, 'phase8_export_%s' % tag) + os.sep,
                   os.path.join(RESULTS, 'phase8_export_%s_SSD' % tag) + os.sep)
             for tag in ('capped', 'uncapped')}
    if '--compare-only' not in sys.argv:
        for tag, margin in (('uncapped', None), ('capped', 10.0)):
            aps.adda_process_single_AroParticle(
                build_config(trees[tag][0], trees[tag][1], margin), FOLDER)

    A = load_avg(trees['capped'][0])       # capped
    B = load_avg(trees['uncapped'][0])     # uncapped (reference)
    print('\ncapped   pha shape: %s  max_ntrunc=%d' % (A['pha'].shape, A['maxntr']))
    print('uncapped pha shape: %s  max_ntrunc=%d' % (B['pha'].shape, B['maxntr']))

    cap = A['maxntr']

    # ---- store noise plateau N_store: coefficients in the top l-band of the
    # uncapped store are pure mueller->phase transform noise (no physical
    # content at production x <= 10 reaches l >= 0.78*Lfull)
    Lfull = int(round((-3 + np.sqrt(1 + 8 * B['pha'].shape[-1])) / 2))
    band_lo = int(np.ceil(0.78 * Lfull))
    N_store = 0.0
    for nb in range(B['pha'].shape[0]):
        for nt in range(B['pha'].shape[1]):
            for k in range(16):
                cb = B['pha'][nb, nt, k, :]
                ls, ms = acf.sph_lm_indices(len(cb))
                sel = ls >= band_lo
                if np.any(sel):
                    N_store = max(N_store, np.abs(cb[sel]).max())
    print('noise plateau band l>=%d of Lfull=%d -> N_store = %.3e'
          % (band_lo, Lfull, N_store))

    # ---- gate 1: kept content bitwise identical over the overlap (l<=cap)
    worst_bitwise = 0.0
    # worst removed-tail relative to its combined tolerance (want <= 1)
    worst_tail_ratio = 0.0
    # informational: tail relative to the element's own norm (noisy elements)
    worst_tail_own = 0.0
    # sqrt(n_removed)+margin: L2 of a pure-noise tail over all removed
    # coefficients cannot exceed sqrt(#removed)*N_store
    n_removed = B['pha'].shape[-1] - A['pha'].shape[-1]
    noise_gain = 1.2 * np.sqrt(max(n_removed, 1))
    for nb in range(A['pha'].shape[0]):
        for nt in range(A['pha'].shape[1]):
            cb11 = B['pha'][nb, nt, 0, :]
            scale = np.sqrt(4.0 * np.pi) * abs(cb11[0])   # |int M11 dOmega|
            tol = EPS_PHASE * scale + noise_gain * N_store
            for k in range(16):
                ca = A['pha'][nb, nt, k, :]
                cb = B['pha'][nb, nt, k, :]
                dA = {(int(l), int(m)): c
                      for l, m, c in zip(*acf.sph_lm_indices(len(ca)), ca)}
                dB = {(int(l), int(m)): c
                      for l, m, c in zip(*acf.sph_lm_indices(len(cb)), cb)}
                # overlap: values must be bitwise identical
                for key in dA:
                    if key not in dB or dA[key] != dB[key]:
                        worst_bitwise = max(worst_bitwise,
                                            abs(dA[key] - dB.get(key, 0j)))
                # removed tail: coefficients with l > cap present only in B
                tail = 0.0
                own = 0.0
                for (l, m), c in dB.items():
                    w = (2.0 if m > 0 else 1.0) * abs(c) ** 2
                    own += w
                    if l > cap:
                        tail += w
                tail = np.sqrt(tail)
                worst_tail_ratio = max(worst_tail_ratio, tail / max(tol, 1e-300))
                worst_tail_own = max(worst_tail_own,
                                     tail / max(np.sqrt(own), 1e-300))

    print('max |bitwise overlap diff|      : %.3e   [gate 0]' % worst_bitwise)
    print('max tail L2 / (eps*scale+gain*N): %.3e   [gate 1.0, gain=%.0f]'
          % (worst_tail_ratio, noise_gain))
    print('(info) max tail L2 / own elem  : %.3e   (noise-level elements)'
          % worst_tail_own)

    # ---- gate 2: abs_vector + ext unchanged
    dabs = np.abs(A['abs'] - B['abs']).max() / max(np.abs(B['abs']).max(), 1e-300)
    dext = np.abs(A['ext'] - B['ext']).max() / max(np.abs(B['ext']).max(), 1e-300)
    print('d_abs_vector (rel)              : %.3e   [gate 1e-6]' % dabs)
    print('d_ext_mat_horiz_avg (rel)       : %.3e   [gate 1e-6]' % dext)

    # ---- size bookkeeping
    nA, sA = nc_tree_size(trees['capped'][1])
    nB, sB = nc_tree_size(trees['uncapped'][1])
    print('SSDB netcdf: uncapped %d files %.1f MB | capped %d files %.1f MB'
          '  -> %.1fx smaller' % (nB, sB / 1024, nA, sA / 1024, sB / max(sA, 1)))

    ok = (worst_bitwise == 0.0 and worst_tail_ratio <= 1.0
          and dabs <= 1e-6 and dext <= 1e-6)
    print('GATE:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
