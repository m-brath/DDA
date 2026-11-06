#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mueller-NetCDF migration gates (plans/DDA-mueller-netcdf-migration-plan.md,
verification items 2 + 3): end-to-end ARO pipeline case + double-writer /
double-reader equivalence.

Pipeline case: adda_config_mueller_nc_probe (tiny oblate ellipsoid, ARO,
mueller_artifact_pickle=True -> the run writes BOTH the new *_mueller_data.nc
and a legacy .pkl twin from the SAME computation).

  GATE W (writer / e2e):
    sub_control run produces *_mueller_data.nc (atomic: no .tmp leftover),
    the per-vertex _temp_data_vertex_*.nc cache is cleaned up, the pkl probe
    twin exists, and main_control's completed-case glob finds the .nc.

  GATE X (artifact content vs overview .mat + pkl twin):
    nc root attrs mirror the mat provenance (shape/T/f/lambda/m/adda_*/
    max_len/max_ntrunc/n_vertices); group ids == mat-vertex zipname_vid ids;
    vertex_index covers exactly the computed vertices; every coefficient
    array == pkl coefficients cast to complex64, BITWISE.

  GATE R (double-reader equivalence):
    aoa.rawdata2average on the same run, once via *_mueller_data.nc and once
    via the legacy .pkl fallback (.nc temporarily hidden) -> every array of
    both returned dicts must be BIT-IDENTICAL (raw stored data; the complex64
    round trip is lossless; both readers hand the same complex64 lists to
    the identical averaging code).

  GATE E (downstream chain + SSDB export bit-equal):
    E1: full chain (mueller2phase_reexpand -> rescale/abs_vector) on both
        averaged dicts of the probe run -> bitwise identical.
    E2: SSDB export (a2s.export2SSDBnetCDF, which imports the habit's
        adda_config module — real habit only) on the Phase-6 HongPlate
        store: legacy .pkl auto-migrated to .nc first; both reader branches
        exported and the SSDB NetCDF trees compared variable-wise, BITWISE.

Usage (DDA conda env):
    python analysis/mueller_nc_gate.py [--force-run]

Run log: analysis/mueller_nc_gate_run.log
Exit code 0 = PASS.
"""

import argparse
import copy
import glob
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from netCDF4 import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'adda_python_scripts')
BIN14 = os.path.join(REPO, 'adda_1.4', 'src', 'seq', 'adda')
CONFIG = 'adda_config_mueller_nc_probe'
RAWDATA = os.path.join(REPO, 'adda_results', 'mueller_nc_probe', 'rawdata')
RUNLOG = os.path.join(REPO, 'analysis', 'mueller_nc_gate_run.log')

sys.path.insert(0, SCRIPTS)

import adda_control_functions as acf   # noqa: E402
import adda_orientational_avg_spharmonics_functions as aoa  # noqa: E402
import adda2SSDBnetcdf_functions as a2s                    # noqa: E402
import spherical_harmonics_for_adda as spha                # noqa: E402

FAILURES = []


def check(ok, name, info=''):
    print(('PASS  ' if ok else 'FAIL  ') + name + ((' — ' + info) if info
                                                  else ''))
    if not ok:
        FAILURES.append(name)


def bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def find_store():
    stores = [d for d in glob.glob(os.path.join(RAWDATA, '*_deq*'))
              if os.path.isdir(d)]
    if len(stores) != 1:
        raise RuntimeError('expected exactly one probe store in %s, found %d'
                           % (RAWDATA, len(stores)))
    return stores[0]


# ==============================================================================
# GATE W — run + writer-side checks
# ==============================================================================

def gate_w(force_run):
    store = find_store() if os.path.isdir(RAWDATA) else None
    has_artifact = store is not None and glob.glob(store
                                                   + '/*_mueller_data.nc')
    if force_run or not has_artifact:
        env = dict(os.environ,
                   ADDA_BINARY=BIN14, ADDA_VERSION='1.4.0')
        print('  running adda_sub_control_spharmonics.py 0 0 0 '
              + CONFIG + ' (ADDA 1.4.0 seq) ...')
        with open(RUNLOG, 'wb') as log:
            rc = subprocess.run(
                [sys.executable, 'adda_sub_control_spharmonics.py',
                 '0', '0', '0', CONFIG],
                cwd=SCRIPTS, env=env, stdout=log,
                stderr=subprocess.STDOUT).returncode
        check(rc == 0, 'W0 sub_control run exit 0',
              'log: ' + RUNLOG + (' (rc=%d)' % rc if rc else ''))
        if rc:
            return None
    else:
        print('  probe run already present (use --force-run to redo)')

    store = find_store()
    ncs = glob.glob(store + '/*_mueller_data.nc')
    pkls = glob.glob(store + '/*_mueller_data.pkl')
    check(len(ncs) == 1, 'W1 exactly one *_mueller_data.nc', store)
    check(len(glob.glob(store + '/*_mueller_data.nc.tmp')) == 0,
          'W1 no *_mueller_data.nc.tmp leftover (atomic os.replace)')
    check(len(glob.glob(store + '/*_temp_data_vertex_*.nc')) == 0,
          'W1 per-vertex temp cache cleaned')
    check(len(pkls) == 1,
          'W1 pkl probe twin written (mueller_artifact_pickle)',
          os.path.basename(pkls[0]) if pkls else 'missing')

    # main_control completed-case glob (production part: nc only; the probe
    # twin additionally matches the legacy .pkl pattern)
    datalist = (glob.glob(store + '/*mueller_data.xml')
                + glob.glob(store + '/*mueller_data.pkl')
                + glob.glob(store + '/*mueller_data.nc'))
    ncs_only = glob.glob(store + '/*mueller_data.nc')
    check(len(ncs_only) == 1 and len(datalist) >= 1,
          'W1 main_control skip-glob accepts the .nc artifact')

    return store


# ==============================================================================
# GATE X — nc content vs overview .mat + pkl twin (bitwise)
# ==============================================================================

def gate_x(store):
    matf = glob.glob(store + '/*_overview_data.mat')[0]
    nc = glob.glob(store + '/*_mueller_data.nc')[0]
    pkl = glob.glob(store + '/*_mueller_data.pkl')[0]

    mat = acf.loadmat(matf)['output_data']
    vertices = np.asarray(mat['vertices'], dtype=float)
    n = vertices.shape[0]

    with Dataset(nc, 'r') as rg:
        att = rg.__dict__
        ok_att = (att['shape_name'] == str(mat['shape_name'])
                  and float(att['temperature'])
                  == float(np.ravel(mat['Temperature'])[0])
                  and float(att['frequency'])
                  == float(np.ravel(mat['Frequency'])[0])
                  and float(att['wavelength'])
                  == float(np.ravel(mat['wavelength'])[0])
                  and complex(*att['refr_index'])
                  == complex(np.ravel(mat['refr_index'])[0])
                  and att['adda_version'] == str(mat['adda_version'])
                  and att['adda_binary'] == str(mat['adda_binary'])
                  and int(att['n_vertices']) == n
                  and int(att['max_len_coeffs'])
                  == int(np.ravel(mat['max_len_coeffs'])[0])
                  and int(att['max_ntrunc'])
                  == int(np.ravel(mat['max_ntrunc'])[0]))
        check(ok_att, 'X1 nc root attrs mirror the overview .mat provenance',
              'T=%s K f=%.3e Hz n_vertices=%d'
              % (att['temperature'], att['frequency'], n))
        check(att['mueller_units'] == acf.MUELLER_UNITS_NOTE,
              'X1 mueller_units provenance attr')

        ids = {}
        for i in range(n):
            ids[acf.mueller_vertex_id(vertices[i])] = i
        check(set(rg.groups.keys()) == set(ids.keys()),
              'X2 group ids == mat-vertex ids (%d groups)' % n)

        idx = {gname: int(grp.vertex_index)
               for gname, grp in rg.groups.items()}
        check(sorted(idx.values()) == list(range(n)),
              'X2 vertex_index covers all computed vertices exactly once')

        # per-vertex max_len/max_ntrunc sanity (float32 relerror/ext_mat are
        # covered bitwise by the pkl comparison below)
        ok_diag = all(rg[g]['relerror'][:].shape == (16,)
                      and rg[g]['ext_mat'][:].shape == (4, 4)
                      for g in rg.groups)
        check(ok_diag, 'X2 relerror(16)/ext_mat(4x4) in every group')

    # nc vs pkl twin: bitwise after the complex64 storage cast
    pklarr = pickle.load(open(pkl, 'rb'))
    ncarr = acf.load_mueller_array_from_netcdf(nc, n)
    ncoef = 0
    bad = 0
    for i in range(n):
        if pklarr[i] is None:
            if ncarr[i] is not None:
                bad += 1
            continue
        for a, b in zip(pklarr[i], ncarr[i]):
            ncoef += 1
            if not bitwise_equal(a.astype(np.complex64), b):
                bad += 1
    check(bad == 0,
          'X3 nc == pkl twin coefficients, bitwise after complex64 cast '
          '(%d arrays / %d vertices)' % (ncoef, n))


# ==============================================================================
# GATE R / E — double reader + full downstream chain
# ==============================================================================

def average_once(store):
    """rawdata2average with cwd hygiene; returns (avg dict, beta array)."""
    mat = acf.loadmat(glob.glob(store + '/*_overview_data.mat')[0]
                      )['output_data']
    incs = np.asarray(mat['theta_inc'], dtype=float).ravel()
    beta = np.asarray(mat['beta'], dtype=float).ravel()
    cwd = os.getcwd()
    try:
        avg = aoa.rawdata2average(store + '/', incs, beta, None, 0, 0, 0,
                                  parallel=False)
    finally:
        os.chdir(cwd)
    return avg, beta


def downstream(avg, grid_size=90, eps=0.02):
    """exact chain of adda_process_aro_particles_single (minus mat save)."""
    [phase_mat, max_ntrunc_pha, max_len_coeffs_pha] = \
        aoa.mueller2phase_reexpand(avg, grid_size_theta=grid_size, eps=eps,
                                   parallel=False, core_limit=2)
    phase_mat = spha.reshape_sphexp_2_common_exp(
        np.max(max_ntrunc_pha[:]), np.max(max_len_coeffs_pha[:]), phase_mat)

    avg['eps_phase'] = eps
    avg['phase_matrix_sph'] = phase_mat
    avg['max_ntrunc'] = int(np.max(max_ntrunc_pha[:]))
    avg['max_len_coeffs'] = int(np.max(max_len_coeffs_pha[:]))

    ext_xsec = np.real(avg['ext_mat_horiz_avg'][:, :, :, 0])
    sca_xsec = np.real(avg['phase_matrix_sph'][:, :, 0:4, 0]
                       / (1 / 2 * np.sqrt(1 / np.pi)))
    abs_vector = ext_xsec - sca_xsec

    a1 = abs_vector[:, :, 0]
    logic_a1 = a1 < 0
    frac = sum(logic_a1.ravel()) / np.size(logic_a1)
    rescale = 1. + a1 / sca_xsec[:, :, 0]
    if frac > 0:
        print('  phase matrix rescaled at %d (beta,incidence) cells'
              % sum(logic_a1.ravel()))
        for i in range(np.size(a1, axis=0)):
            for j in range(np.size(a1, axis=1)):
                if logic_a1[i, j]:
                    phase_mat[i, j, :, :] = phase_mat[i, j, :, :] * rescale[i, j]
        avg['phase_matrix_sph'] = phase_mat.copy()
        ext_xsec = np.real(avg['ext_mat_horiz_avg'][:, :, :, 0])
        sca_xsec = np.real(avg['phase_matrix_sph'][:, :, 0:4, 0]
                           / (1 / 2 * np.sqrt(1 / np.pi)))
        abs_vector = ext_xsec - sca_xsec

    a1 = abs_vector[:, :, 0]
    a1[a1 < 0] = 0.
    abs_vector[:, :, 0] = a1
    avg['rescale'] = rescale
    avg['abs_vector'] = abs_vector
    return avg


def export(avg, beta, db):
    cfg = {'crystal_type': 'Aggregate', 'pristine_type': 'Pristine',
           'orientation_type': 'Azimuthally_random', 'database_path': db}
    return a2s.export2SSDBnetCDF(avg, cfg)


def compare_dicts(a, b, tag):
    check(set(a.keys()) == set(b.keys()), tag + ' — identical dict keys')
    bad, cmp_n = [], 0
    for k in a:
        if k not in b:
            bad.append(k + ' (missing key)')
            continue
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            cmp_n += 1
            if not bitwise_equal(x, y):
                bad.append(k)
        elif isinstance(x, (int, float, complex, str, bool)) or \
                isinstance(x, np.generic):
            cmp_n += 1
            if not (x == y):
                bad.append(k)
    check(not bad, tag + ' bitwise identical (%d arrays/scalars compared)'
          % cmp_n, 'differing: %s' % bad if bad else '')


def compare_ssdb(files_a, files_b):
    """variable-wise bitwise comparison of two SSDB export trees."""
    if len(files_a) != len(files_b):
        return False, 'file count %d vs %d' % (len(files_a), len(files_b))
    nvar = 0
    for fa, fb in zip(sorted(files_a), sorted(files_b)):
        with Dataset(fa) as da, Dataset(fb) as db:
            va = {}
            vb = {}

            def walk(g, store):
                for name, var in g.variables.items():
                    data = var[:]
                    store[name] = (np.ma.getmaskarray(data),
                                   np.ma.getdata(data))
                for sub in g.groups.values():
                    walk(sub, store)
            walk(da, va)
            walk(db, vb)
            if set(va.keys()) != set(vb.keys()):
                return False, 'variable sets differ in %s' % fa
            for k in va:
                nvar += 1
                (ma, xa), (mb, xb) = va[k], vb[k]
                if not (np.array_equal(ma, mb) and bitwise_equal(xa, xb)):
                    return False, 'variable %s differs in %s' % (k, fa)
    return True, '%d files, %d variables bitwise' % (len(files_a), nvar)


def gate_re(store):
    # ---- reader A: NetCDF artifact ----
    avg_nc, beta = average_once(store)
    print('  R1 averaged via .nc: keys=%d' % len(avg_nc))

    # ---- reader B: legacy pkl fallback (nc temporarily hidden) ----
    store_nc = glob.glob(store + '/*_mueller_data.nc')[0]
    hidden = store_nc + '.hidden'
    os.rename(store_nc, hidden)
    try:
        avg_pkl, _ = average_once(store)
    finally:
        os.rename(hidden, store_nc)
    print('  R1 averaged via .pkl fallback: keys=%d' % len(avg_pkl))

    compare_dicts(avg_nc, avg_pkl, 'R2 nc- vs pkl-averaged dict')

    # ---- E1: full downstream chain (phase matrix + rescale/abs_vector) ----
    #      (SSDB export itself needs a habit config module — covered for a
    #      real habit in gate_e_ssdb below)
    ch_nc = downstream(copy.deepcopy(avg_nc))
    ch_pkl = downstream(copy.deepcopy(avg_pkl))
    compare_dicts(ch_nc, ch_pkl, 'E1 nc- vs pkl-phase-matrix chain')


# ==============================================================================
# GATE E2 — SSDB export for a REAL habit (config import works there), on a
# Phase-6 store whose legacy .pkl was migrated to .nc by migrate_pkl_to_nc
# ==============================================================================

HP_GLOB = os.path.join(REPO, 'adda_results', 'phase6_hp_sym',
                       'HongPlate_Id9_deq*')


def gate_e_ssdb():
    stores = [d for d in glob.glob(HP_GLOB) if os.path.isdir(d)]
    if len(stores) != 1:
        print('  SKIP  E2: phase6_hp_sym HongPlate store not found (%d)'
              % len(stores))
        return

    store = stores[0]
    pkls = glob.glob(store + '/*_mueller_data.pkl')
    ncs = glob.glob(store + '/*_mueller_data.nc')
    if not pkls:
        print('  SKIP  E2: store has no .pkl twin to compare against')
        return
    if not ncs:
        print('  E2: migrating %s with migrate_pkl_to_nc ...'
              % os.path.basename(store))
        rc = subprocess.run(
            [sys.executable, os.path.join(REPO, 'analysis',
                                          'migrate_pkl_to_nc.py'), store],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            check(False, 'E2 auto-migration of the HongPlate store')
            return
        ncs = glob.glob(store + '/*_mueller_data.nc')
        if not ncs:
            check(False, 'E2 auto-migration produced no .nc')
            return
    print('  E2 store: %s' % os.path.basename(store))

    avg_nc, beta = average_once(store)
    store_nc = glob.glob(store + '/*_mueller_data.nc')[0]
    hidden = store_nc + '.hidden'
    os.rename(store_nc, hidden)
    try:
        avg_pkl, beta2 = average_once(store)
    finally:
        os.rename(hidden, store_nc)
    compare_dicts(avg_nc, avg_pkl, 'E2 nc- vs pkl-averaged dict (HongPlate)')

    ch_nc = downstream(copy.deepcopy(avg_nc), grid_size=90, eps=0.02)
    ch_pkl = downstream(copy.deepcopy(avg_pkl), grid_size=90, eps=0.02)

    db1 = tempfile.mkdtemp()
    db2 = tempfile.mkdtemp()
    try:
        ncs1 = export(ch_nc, beta, db1)
        ncs2 = export(ch_pkl, beta2, db2)
        check(len(ncs1) == len(beta) and len(ncs2) == len(beta2),
              'E2 SSDB export proceeds from both artifacts',
              '%d beta files each' % len(ncs1))
        ok, info = compare_ssdb(ncs1, ncs2)
        check(ok, 'E2 SSDB NetCDF trees nc- vs pkl-branch bitwise identical',
              info)
    finally:
        shutil.rmtree(db1, ignore_errors=True)
        shutil.rmtree(db2, ignore_errors=True)


# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force-run', action='store_true')
    args = ap.parse_args()

    store = gate_w(args.force_run)
    if store is None:
        print('\nRESULT: FAIL (sub_control run failed — see ' + RUNLOG + ')')
        return 1
    gate_x(store)
    gate_re(store)
    gate_e_ssdb()

    print()
    if FAILURES:
        print('RESULT: FAIL (%d): %s' % (len(FAILURES), FAILURES))
        return 1
    print('RESULT: PASS — e2e writer, artifact-vs-mat cross-check, double-'
          'reader equivalence and SSDB export are bitwise clean')
    return 0


if __name__ == '__main__':
    sys.exit(main())
