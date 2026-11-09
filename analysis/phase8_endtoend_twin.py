#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 8 end-to-end twin: adaptive-scattering-grid ARO run (phase8_hp_adaptive,
22x42 grid) vs historic static-grid run (phase6_hp_sym, 181x361 grid) of the
SAME case (HongPlate_Id9, Dveq=10um, 228 GHz, T=190 K, same icosphere/delta/
symmetry settings, ADDA 1.4).

Compares, matched by vertex coordinates (vertex_index is per-run, so vertices
are matched geometrically):
  * per-vertex extinction matrices and backscatter matrices
  * per-vertex mueller sph-coefficient sets (union over stored (l,m))
  * the orientation-averaged ext_mat_horiz_avg from the overview mats
Gates are noise-based: production eps_sphexp tolerance + solver-noise slack.
"""
import os
import sys
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'adda_python_scripts'))
import adda_control_functions as acf

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..'))
# twin pair: adaptive vs static run of the SAME pipeline vintage
# (phase8_hp_static re-generated with current code isolates the grid change;
# comparing against the phase6_hp_sym store adds old-code vintage effects)
A = os.path.join(REPO, 'adda_results', 'phase8_hp_adaptive',
                 'HongPlate_Id9_deq00010um_freq228.0GHz_T0190K')
B = os.path.join(REPO, 'adda_results', 'phase8_hp_static',
                 'HongPlate_Id9_deq00010um_freq228.0GHz_T0190K')
if len(sys.argv) == 3:
    A, B = (os.path.join(REPO, 'adda_results', *p.split('/')) for p in sys.argv[1:3])

EPS_SPH = 0.005
SOLVER_SLACK = 50.0   # tiny-particle solver noise at -eps 3 can be ~eps^2-ish


def _arr(x):
    """unwrap a scipy loadmat(squeeze_me=True) 0-d object cell"""
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    return np.asarray(x)


def load_run(path):
    ov = sio.loadmat(os.path.join(path, 'HongPlate_Id9_overview_data.mat'),
                     squeeze_me=True)['output_data']
    v = _arr(ov['vertices']).astype(float)
    ext = _arr(ov['ext_mat_array']).astype(float)
    back = _arr(ov['back_mat_array']).astype(float)
    mavg = _arr(ov['ext_mat_horiz_avg']).astype(float)
    sph = acf.load_mueller_array_from_netcdf(
        os.path.join(path, 'HongPlate_Id9_mueller_data.nc'))
    return {'v': v, 'ext': ext, 'back': back, 'mavg': mavg, 'sph': sph}


def rel(a, b, ref):
    return np.abs(a - b).max() / max(abs(ref).max(), 1e-300)


def main():
    ra, rb = load_run(A), load_run(B)
    print('adaptive: n_vertices=%d computed=%d' %
          (ra['v'].shape[0], sum(s is not None for s in ra['sph'])))
    print('static  : n_vertices=%d computed=%d' %
          (rb['v'].shape[0], sum(s is not None for s in rb['sph'])))
    if ra['v'].shape[0] != rb['v'].shape[0]:
        print('DIFFERENT icosphere level -> compare common vertices only')

    # exact geometric match by rounding vertex coordinates to grid-id precision
    def vid(vv):
        return {(round(c * 1e5), round(d * 1e5), round(e * 1e5)): i
                for i, (c, d, e) in enumerate(vv)}
    idA, idB = vid(ra['v']), vid(rb['v'])
    common = [k for k in idA if k in idB]
    print('geometrically matched vertices: %d (A=%d B=%d)'
          % (len(common), len(idA), len(idB)))

    dext = dback = dsph = 0.0
    ncmp = 0
    for k in common:
        ia, ib = idA[k], idB[k]
        sa, sb = ra['sph'][ia], rb['sph'][ib]
        if sa is None or sb is None:
            continue
        ncmp += 1
        dext = max(dext, rel(ra['ext'][ia], rb['ext'][ib], rb['ext'][ib]))
        dback = max(dback, rel(ra['back'][ia], rb['back'][ib],
                               abs(rb['back'][ib][0, 0])))
        worst = 0.0
        m11_scale = 0.0
        for ea, eb in zip(sa, sb):
            la, ma = acf.sph_lm_indices(len(ea))
            lb, mb = acf.sph_lm_indices(len(eb))
            dB = {(int(l), int(m)): c for l, m, c in zip(lb, mb, eb)}
            acc = 0.0
            for l, m, c in zip(la, ma, ea):
                d = c - dB.pop((int(l), int(m)), 0.0)
                acc += abs(d) ** 2
            for c in dB.values():
                acc += abs(c) ** 2
            worst = max(worst, np.sqrt(acc))
        m11_scale = abs(np.sqrt(4.0 * np.pi) * sb[0][0].real)  # |int M11 dOmega|
        dsph = max(dsph, worst / max(m11_scale, 1e-300))
    print('compared vertices: %d' % ncmp)
    print('d_ext (4x4 matrix, rel)      : %.3e   [gate %.1e]'
          % (dext, max(1e-8, SOLVER_SLACK * 1e-11)))
    print('d_back (rel to M11)          : %.3e   [gate %.1e]'
          % (dback, 1e-2))
    print('d_sph (rel to |int M11 dOm|) : %.3e   [gate %.1e]'
          % (dsph, 1e-2))
    print('d_ext_horiz_avg (rel)        : %.3e   [gate %.1e]'
          % (rel(ra['mavg'], rb['mavg'], rb['mavg']), 1e-2))

    ok = (dext <= max(1e-8, SOLVER_SLACK * 1e-11)
          and dback <= 1e-2 and dsph <= 1e-2
          and rel(ra['mavg'], rb['mavg'], rb['mavg']) <= 1e-2)
    print('GATE:', 'PASS' if ok else 'FAIL')

    # data-size bookkeeping
    for tag, p in (('adaptive', A), ('static  ', B)):
        nc = os.path.getsize(os.path.join(p, 'HongPlate_Id9_mueller_data.nc'))
        print('%s mueller_data.nc: %.1f kB' % (tag, nc / 1024.0))


if __name__ == '__main__':
    main()
