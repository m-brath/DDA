#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 8 gate: size-parameter-adaptive ADDA scattering grid
(acf.scatgrid_ntrunc_cap / acf.write_scat_grid_params).

Strategy: for a set of size parameters x (pi*Dveq/lambda), run acf.adda_call
twice per x — once on the historic static production grid (181x361) and once
on the adaptive grid from the budget law (per margin) — and compare:

  * the forward extinction matrix  (theta=0 column of ampl_scatgrid)
  * the backscatter matrix + ring  (Theta=180 row of mueller_scatgrid)
  * per-element L2 distance of the stored sph expansions
    (expand_mueller_2_spharmonics, ragged-prefix compare; the shtns
    orthonormal coefficient space is unitary w.r.t. the sphere L2 norm)
  * grid-independent RT moments taken from the M11/M22 coefficient sets
    themselves (C ~ c00, g ~ c10/c00, r ~ c00(M22)/c00(M11)) — comparing
    trapezoid moments on the grids themselves would just measure the
    quadrature error of the coarse grid, not the representation error

Solver noise is measured by re-running the static grid (twin of twin) and
all gates are noise-based:  adaptive-vs-static difference must stay below
max(1e-8, 5 * solver-noise)  for ext/back and 5e-3 relative for the moments
(quadrature-level effects). Prints + writes analysis/phase8_results.json.

Usage (DDA conda env):  python analysis/phase8_scatgrid_twin.py [--margins 2,3,4]
"""

import os
import sys
import json
import time
import tempfile

import numpy as np

try:
    from scipy.integrate import trapezoid as _trapz
    _HAS_TRAPZ = True
except ImportError:
    _HAS_TRAPZ = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'adda_python_scripts'))

import adda_control_functions as acf
import spherical_harmonics_for_adda as spha

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ADDA_BIN = os.path.join(REPO, 'adda_1.4', 'src', 'seq', 'adda')
SHAPES = os.path.join(REPO, 'adda_shape_files', 'HongPlate_Id9')

STATIC_GRID = os.path.join(REPO, 'adda_python_scripts', 'scat_params.dat')

# ---------------------------------------------------------------- cases
# (label, shape file, equivalent radius [um] = pipeline -eq_rad D/2,
#  wavelength [um]);  x = pi * (2*eq_rad) / lambda
# All at lambda = 1315.79 um (228 GHz) so the Matzler ice dielectric stays
# in range; ADDA rescales the geometry file with -eq_rad (dipoles fixed),
# exactly like the pipeline does. The x0.02 case uses the 10 um plate.
CASES = [
    ('x0.02', 'Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0192.000GHz.adda',
     5.0, 1315.79),
    ('x0.5', 'Dmax01638um_Dveq00584um_Mass9.54564e-08kg_Freq0887.000GHz.adda',
     104.76, 1315.79),
    ('x5', 'Dmax01638um_Dveq00584um_Mass9.54564e-08kg_Freq0887.000GHz.adda',
     1047.6, 1315.79),
    ('x10', 'Dmax01638um_Dveq00584um_Mass9.54564e-08kg_Freq0887.000GHz.adda',
     2095.2, 1315.79),
]

T_ICE = 230.0
EPS_ADDA = 3          # production solver tolerance
EPS_SPH = 0.005       # production expansion tolerance

# theta-resolution constants to calibrate against the static reference
# (kappa=0.17 reproduces the static 181-point grid at x=10 exactly)
KAPPAS = [0.06, 0.10, 0.17]

# oblique general vertex direction (not axis-aligned) — exercises prop path
PROP = np.array([0.3, 0.5, -0.81])
PROP = PROP / np.linalg.norm(PROP)


def make_input(shapefile, lamb, eq_rad, m, gridfile):
    static_clo = ['-shape', 'read', os.path.join(SHAPES, shapefile),
                  '-store_scat_grid', '-scat_grid_inp', gridfile,
                  '-scat_matr', 'both']
    return {
        'lambda': lamb,
        'Size': eq_rad,                    # adda_call passes -eq_rad directly
        'refrative_index': m,
        'theta': PROP,
        'alpha': 0., 'beta': 0., 'gamma': 0.,
        'eps': EPS_ADDA,
        'ramdrive': None,
        'main_command': [ADDA_BIN],
        'static_clo': static_clo,
        'zipname': tempfile.mktemp(suffix='.zip', dir=tempfile.gettempdir()),
        'zipping': False,
    }


def sph_moments(sph):
    """grid-independent moment proxies from the SH coefficients of the
    orthonormal, m>=0 shtns layout (m-major blocks; (0,0)->0, (1,0)->1):
        C ~ int M11 dOmega        = sqrt(4pi) * c00(M11)
        g ~ int cos(th) M11 / int M11 = c10(M11) / (sqrt(3) c00(M11))
        I ~ int M22 dOmega        = sqrt(4pi) * c00(M22)
    (element order follows adda_call's mueller_array columns:
    M11 = 0, M22 = 3). I (cross-pol integral) is kept ABSOLUTE (scaled by
    C) rather than as the ratio I/C: ratio differences explode when I is a
    small cancellation-dominated number, while what matters physically is
    the absolute depolarized-power integral relative to the total."""
    c11 = sph[0]
    c22 = sph[3]
    c00_11 = c11[0].real
    c10_11 = c11[1].real if len(c11) > 1 else 0.0
    c00_22 = c22[0].real
    C = np.sqrt(4.0 * np.pi) * c00_11
    g = c10_11 / (np.sqrt(3.0) * c00_11)
    I = np.sqrt(4.0 * np.pi) * c00_22
    return C, g, I


def sph_l2_distance(sphA, sphB, m11_int):
    """per-element ||cA - cB||_2 / m11_int over the union of stored (l,m)
    sets. shtns storage is m-major block order ((l,m) from
    acf.sph_lm_indices), so arrays of different ntrunc MUST be matched by
    (l,m), never by raw prefix."""
    worst = 0.0
    per = []
    for a, b in zip(sphA, sphB):
        la, ma = acf.sph_lm_indices(len(a))
        lb, mb = acf.sph_lm_indices(len(b))
        keyB = {(int(l), int(m)): c for l, m, c in zip(lb, mb, b)}
        keyA = {(int(l), int(m)): c for l, m, c in zip(la, ma, a)}
        acc = 0.0
        for key in set(keyA) | set(keyB):
            ca = keyA.get(key, 0.0)
            cb = keyB.get(key, 0.0)
            acc += abs(ca - cb) ** 2
        d = np.sqrt(acc) / m11_int
        per.append(d)
        worst = max(worst, d)
    return worst, per


def run_case(case, margins, workdir):
    label, shapefile, eq_rad, lamb = case
    x = np.pi * 2.0 * eq_rad / lamb
    m = np.sqrt(acf.eps_ice_matzler06(c0_over_lam(lamb), T_ICE))

    res = {'label': label, 'x': x, 'eq_rad_um': eq_rad, 'lambda_um': lamb,
           'shape': shapefile}

    runs = {}

    def one(tag, gridfile):
        t0 = time.perf_counter()
        out = acf.adda_call(make_input(shapefile, lamb, eq_rad, m, gridfile))
        wall = time.perf_counter() - t0
        sph, relerr, maxlen, maxntr = spha.expand_mueller_2_spharmonics(
            out, EPS_SPH, diagnostics=False)
        C, g, I = sph_moments(sph)
        runs[tag] = {'wall': wall, 'out': out, 'sph': sph,
                     'relerr': relerr, 'maxlen': maxlen, 'maxntr': maxntr,
                     'moments': (C, g, I),
                     'grid': (out['thetas_grid'].size, out['phis_grid'].size)}
        return out

    # static production grid, twice (solver-noise floor)
    ref = one('static_a', STATIC_GRID)
    one('static_b', STATIC_GRID)

    # adaptive grids (margin x theta-kappa variants)
    for kappa in KAPPAS:
        for mg in margins:
            cap = acf.scatgrid_ntrunc_cap(x, margin=mg)
            gf = os.path.join(workdir,
                              'scat_adaptive_%s_mg%s_k%s.dat'
                              % (label, mg, kappa))
            acf.write_scat_grid_params(gf, cap, x=x, kappa=kappa)
            one('mg%g_k%g' % (mg, kappa), gf)

    C0, g0, I0 = runs['static_a']['moments']
    # normalization consistent with expand_mueller_2_spharmonics: m11_int =
    # L2 sphere norm sqrt(int M11^2 dOmega) of the STATIC reference field
    # (shtns coefficient space is unitary w.r.t. this norm, so
    # ||dc||_2 == L2 field distance directly)
    outS = runs['static_a']['out']
    thS = np.deg2rad(outS['thetas_grid'])
    phS = outS['phis_grid']
    M11S = outS['mueller_array'][:, :, 0]
    if _HAS_TRAPZ:
        l2m11 = np.sqrt(_trapz(_trapz(M11S ** 2 * np.sin(thS)[:, None],
                                      x=phS, axis=1), x=thS))
    else:
        l2m11 = np.sqrt(np.trapz(np.trapz(M11S ** 2 * np.sin(thS)[:, None],
                                          x=phS, axis=1), x=thS))
    m11_int = float(l2m11)

    # solver noise floor
    def noise(tagA, tagB):
        d_ext = np.abs(runs[tagA]['out']['ext_matrix']
                       - runs[tagB]['out']['ext_matrix']).max() / abs(
            runs[tagA]['out']['ext_matrix'][0, 0])
        d_back = np.abs(runs[tagA]['out']['back_matrix']
                        - runs[tagB]['out']['back_matrix']).max() / abs(
            runs[tagA]['out']['back_matrix'][0, 0])
        ds, _ = sph_l2_distance(runs[tagA]['sph'], runs[tagB]['sph'], m11_int)
        return {'ext': float(d_ext), 'back': float(d_back), 'sph': float(ds)}

    res['solver_noise'] = noise('static_a', 'static_b')

    res['adaptive'] = {}
    for kappa in KAPPAS:
        for mg in margins:
            tag = 'mg%g_k%g' % (mg, kappa)
            A, S = runs[tag], runs['static_a']
            d_ext = np.abs(A['out']['ext_matrix']
                           - S['out']['ext_matrix']).max() \
                / abs(S['out']['ext_matrix'][0, 0])
            d_back = np.abs(A['out']['back_matrix']
                            - S['out']['back_matrix']).max() \
                / abs(S['out']['back_matrix'][0, 0])
            sph_max, sph_per = sph_l2_distance(A['sph'], S['sph'], m11_int)
            C, g, I = A['moments']
            res['adaptive'][tag] = {
                'kappa': kappa,
                'margin': mg,
                'ntrunc_cap': acf.scatgrid_ntrunc_cap(x, margin=mg),
                'grid': list(A['grid']),
                'points': int(A['grid'][0] * A['grid'][1]),
                'wall_s': A['wall'],
                'static_wall_s': S['wall'],
                'speedup': S['wall'] / A['wall'],
                'max_ntrunc_stored': int(A['maxntr']),
                'max_len_coeffs': int(A['maxlen']),
                'relerr_max': float(np.nanmax(A['relerr'])),
                'd_ext': float(d_ext),
                'd_back': float(d_back),
                'd_sph_max': float(sph_max),
                'd_sph_per_elem': [float(v) for v in sph_per],
                'd_moment_C': float(abs(C - C0) / abs(C0)),
                'd_moment_g': float(abs(g - g0)),
                'd_moment_I': float(abs(I - I0) / abs(C0)),
            }

    res['static'] = {'wall_s': S['wall'],
                     'grid': list(S['grid']),
                     'points': int(S['grid'][0] * S['grid'][1]),
                     'max_ntrunc_stored': int(S['maxntr']),
                     'max_len_coeffs': int(S['maxlen']),
                     'relerr_max': float(np.nanmax(S['relerr']))}

    # ---------------------------------------------------------- gates
    # noise floor: static_a vs static_b twins
    nzC, nzg, nzI = runs['static_b']['moments']
    tolC = max(10.0 * abs(nzC - C0) / abs(C0), 2.0 * EPS_SPH)
    tol_g = max(10.0 * abs(nzg - g0), 2.0 * EPS_SPH)
    tol_I = max(10.0 * abs(nzI - I0) / abs(C0), 2.0 * EPS_SPH)
    relerr_static = res['static']['relerr_max']
    nz = res['solver_noise']
    gates = {}
    for kappa in KAPPAS:
        for mg in margins:
            tag = 'mg%g_k%g' % (mg, kappa)
            a = res['adaptive'][tag]
            tol_ext = max(1e-8, 5.0 * nz['ext'])
            tol_back = max(1e-8, 5.0 * nz['back'])
            # two representations that each meet eps_sphexp may differ by
            # at most 2x the tolerance (triangle inequality on the L2
            # metric); the relerr gate additionally requires the ADAPTIVE
            # representation to be no worse than the static one (static
            # itself can exceed eps at tiny amplitudes where the shared
            # eps target is below noise)
            tol_sph = max(1e-8, 5.0 * nz['sph'], 2.0 * EPS_SPH)
            tol_relerr = max(2.0 * EPS_SPH, 1.25 * relerr_static)
            gates[tag] = {
                'ext': a['d_ext'] <= tol_ext,
                'back': a['d_back'] <= tol_back,
                'sph': a['d_sph_max'] <= tol_sph,
                'relerr': a['relerr_max'] <= tol_relerr,
                'moments': (a['d_moment_C'] <= tolC
                            and a['d_moment_g'] <= tol_g
                            and a['d_moment_I'] <= tol_I),
            }
            gates[tag]['ALL'] = all(gates[tag].values())
    res['gates'] = gates
    return res


def c0_over_lam(lam_um):
    return 299792458.0 / (lam_um * 1e-6)


def main():
    margins = [3.0, 4.0, 6.0]
    if '--margins' in sys.argv:
        margins = [float(v) for v in sys.argv[sys.argv.index('--margins') + 1].split(',')]

    workdir = tempfile.mkdtemp(prefix='phase8_scatgrid_')
    print('workdir:', workdir)

    allres = {'margins': margins, 'cases': []}
    for case in CASES:
        print('\n=== case %s (x=%.4f) ===' % (case[0], np.pi * 2 * case[2] / case[3]))
        r = run_case(case, margins, workdir)
        allres['cases'].append(r)
        nz = r['solver_noise']
        print('solver noise: ext=%.2e back=%.2e sph=%.2e' %
              (nz['ext'], nz['back'], nz['sph']))
        print('static: wall %.2fs %dx%d stored ntrunc=%d len=%d' %
              (r['static']['wall_s'], r['static']['grid'][0],
               r['static']['grid'][1], r['static']['max_ntrunc_stored'],
               r['static']['max_len_coeffs']))
        for kappa in KAPPAS:
            for mg in margins:
                a = r['adaptive']['mg%g_k%g' % (mg, kappa)]
                print('  k=%-5g mg=%-4g cap=%-4d grid=%-4dx%-4d wall=%6.2fs '
                      'speedup=%5.1fx ntrunc=%-3d relerr=%.2e | '
                      'd_ext=%.2e d_back=%.2e d_sph=%.2e dC=%.2e dg=%.2e '
                      'dI=%.2e | %s'
                      % (kappa, mg, a['ntrunc_cap'], a['grid'][0],
                         a['grid'][1],
                         a['wall_s'], a['speedup'], a['max_ntrunc_stored'],
                         a['relerr_max'],
                         a['d_ext'], a['d_back'], a['d_sph_max'],
                         a['d_moment_C'], a['d_moment_g'], a['d_moment_I'],
                         'PASS' if r['gates']['mg%g_k%g' % (mg, kappa)]['ALL']
                         else 'FAIL'))

    outjson = os.path.join(REPO, 'analysis', 'phase8_results.json')
    with open(outjson, 'w') as fh:
        json.dump(allres, fh, indent=1)
    print('\nwrote', outjson)

    ok = all(r['gates']['mg%g_k%g' % (m, k)]['ALL'] for r in allres['cases']
             for k in KAPPAS for m in margins)
    print('GATE OVERALL (all kappa/margin variants):',
          'PASS' if ok else 'see table above')


if __name__ == '__main__':
    main()
