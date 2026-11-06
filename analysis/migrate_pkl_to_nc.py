#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate legacy *_mueller_data.pkl artifacts to the NetCDF4 format
(plans/DDA-mueller-netcdf-migration-plan.md, change 5).

For every pickle artifact with a *_overview_data.mat sibling (the .mat
supplies vertices + provenance + per-vertex ext/back arrays):

  1. load pkl + mat,
  2. derive vertex_index from the pkl list order + mat vertices via
     acf.mueller_vertex_id (same ids the pipeline writer uses),
  3. write the new *_mueller_data.nc via acf.write_mueller_array_to_netcdf,
  4. VERIFY: reload with acf.load_mueller_array_from_netcdf and assert
     BITWISE equality against the pkl after the complex64 storage cast
     (exact check, not noise-based: raw stored data. The pkl's complex128
     coefficients are already downcast to complex64 by rawdata2average, so
     the storage cast loses nothing the pipeline ever used),
  5. never touch the .pkl unless --prune is given (then rename to
     .pkl.migrated, never delete).

Existing .nc siblings: skipped as already-migrated when content-equal,
refused without --force otherwise.

Note: per-vertex max_len_coeffs group attrs are re-derived from the
coefficient lengths; per-vertex max_ntrunc is not stored in the legacy
.mat, so the run-global max_ntrunc is used for every group (diagnostic
only — not used by any reader).

Usage (DDA conda env):
    python analysis/migrate_pkl_to_nc.py DIR [DIR ...] [--prune] [--force]
    python analysis/migrate_pkl_to_nc.py --check-only DIR ...

Exit code 0 = all OK (or nothing to do), 1 = any failure.
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'adda_python_scripts')
sys.path.insert(0, SCRIPTS)

import adda_control_functions as acf   # noqa: E402


def bitwise_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def nc_matches_existing(ncfile, pklarr):
    """True if an existing .nc already carries exactly this pkl's content."""
    try:
        loaded = acf.load_mueller_array_from_netcdf(ncfile, len(pklarr))
    except Exception:
        return False
    if len(loaded) != len(pklarr):
        return False
    for a, b in zip(pklarr, loaded):
        if a is None or b is None:
            if a is not b:
                return False
            continue
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not bitwise_equal(np.asarray(x).astype(np.complex64), y):
                return False
    return True


def migrate_one(pklf, matf, prune=False, force=False, check_only=False):
    store = os.path.dirname(pklf)
    base = os.path.basename(pklf)[:-len('.pkl')]
    ncfile = os.path.join(store, base + '.nc')

    pklarr = pickle.load(open(pklf, 'rb'))
    if not isinstance(pklarr, list):
        print('  SKIP  %s: pickle is not a vertex list' % pklf)
        return None

    mat = acf.loadmat(matf)['output_data']
    vertices = np.asarray(mat['vertices'], dtype=float)
    if vertices.shape[0] != len(pklarr):
        print('  SKIP  %s: %d pkl entries vs %d mat vertices (misaligned)'
              % (pklf, len(pklarr), vertices.shape[0]))
        return None

    ext_mat_array = np.asarray(mat['ext_mat_array'], dtype=np.float32)
    if ext_mat_array.shape[0] != len(pklarr):
        print('  SKIP  %s: ext_mat_array misaligned' % pklf)
        return None

    # pre-Phase-6 overview files carry no back_mat_array
    try:
        back_mat_array = np.asarray(mat['back_mat_array'], dtype=np.float32)
        if back_mat_array.shape[0] != len(pklarr):
            back_mat_array = None
    except KeyError:
        back_mat_array = None

    meta = {}
    meta['shape_name'] = str(np.ravel(mat['shape_name'])[0])
    meta['temperature'] = float(np.ravel(mat['Temperature'])[0])
    meta['frequency'] = float(np.ravel(mat['Frequency'])[0])
    meta['wavelength'] = float(np.ravel(mat['wavelength'])[0])
    meta['refr_index'] = complex(np.ravel(mat['refr_index'])[0])
    try:
        meta['adda_version'] = str(np.ravel(mat['adda_version'])[0])
        meta['adda_binary'] = str(np.ravel(mat['adda_binary'])[0])
    except KeyError:
        pass  # very old stores: provenance keys absent in the .mat
    meta['max_len_coeffs'] = int(np.ravel(mat['max_len_coeffs'])[0])
    meta['max_ntrunc'] = int(np.ravel(mat['max_ntrunc'])[0])

    if os.path.exists(ncfile):
        if nc_matches_existing(ncfile, pklarr):
            print('  OK    %s already migrated (content-verified)' % base)
            return True
        if not force:
            print('  SKIP  %s: existing .nc differs from pkl '
                  '(use --force to overwrite)' % pklf)
            return None

    if check_only:
        print('  TODO  %s: not yet migrated' % pklf)
        return None

    acf.write_mueller_array_to_netcdf(
        ncfile, pklarr, vertices,
        np.zeros((len(pklarr), 16), dtype=np.float32),  # relerror not in mat
        ext_mat_array, back_mat_array=back_mat_array, meta=meta)

    # --- verification: bitwise vs the pkl (after the complex64 cast) ---
    loaded = acf.load_mueller_array_from_netcdf(ncfile, len(pklarr))
    bad = 0
    ncoef = 0
    for i, (a, b) in enumerate(zip(pklarr, loaded)):
        if a is None:
            if b is not None:
                bad += 1
            continue
        if b is None or len(a) != len(b):
            bad += 1
            continue
        for x, y in zip(a, b):
            ncoef += 1
            if not bitwise_equal(np.asarray(x).astype(np.complex64), y):
                bad += 1
    if bad:
        print('  FAIL  %s: verification against pkl failed at %d places '
              '(.pkl kept, remove the bad .nc)' % (pklf, bad))
        return False

    nvert = sum(1 for x in loaded if x is not None)
    print('  OK    %s: %d vertices / %d coeff arrays bitwise-verified'
          % (base, nvert, ncoef))

    if prune:
        os.rename(pklf, pklf + '.migrated')
        print('        (renamed to %s.pkl.migrated)' % base)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+', help='trees to scan recursively')
    ap.add_argument('--prune', action='store_true',
                    help='rename migrated .pkl to .pkl.migrated '
                         '(never deletes)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing content-differing .nc')
    ap.add_argument('--check-only', action='store_true',
                    help='report what would be migrated, write nothing')
    args = ap.parse_args()

    n_ok = n_skip = n_fail = 0
    for d in args.dirs:
        for pklf in sorted(glob.glob(os.path.join(d, '**',
                                                  '*_mueller_data.pkl'),
                                     recursive=True)):
            print('migrating %s' % pklf)
            mats = glob.glob(os.path.join(os.path.dirname(pklf),
                                          '*_overview_data.mat'))
            if len(mats) != 1:
                print('  SKIP  %d overview .mat siblings in %s'
                      % (len(mats), os.path.dirname(pklf)))
                n_skip += 1
                continue
            r = migrate_one(pklf, mats[0], prune=args.prune, force=args.force,
                            check_only=args.check_only)
            if r is True:
                n_ok += 1
            elif r is None:
                n_skip += 1
            else:
                n_fail += 1

    print('\nmigrated/verified: %d   skipped: %d   failed: %d'
          % (n_ok, n_skip, n_fail))
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
