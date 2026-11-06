#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-1 gate of plans/DDA-mueller-netcdf-migration-plan.md:
round-trip unit check for acf.write_mueller_array_to_netcdf /
acf.load_mueller_array_from_netcdf (no ADDA run needed).

Checks
  1. synthesized jagged mueller_array (real refined-icosphere vertices,
     None holes, 16 ragged complex64 coeff lists/vertex, relerror, ext/back
     matrices, back_mat rings) -> write nc -> reload:
     - BITWISE equality of all complex64 coefficients (raw stored data,
       lossless complex64 <-> float_vlen round trip — not noise-based),
     - vertex-index ordering identical to the pkl list order,
     - None holes stay None (no groups),
     - group names match the temp-cache zipname_vid ids,
     - root provenance attrs (incl. complex refr_index as [re, im]) and
       group attrs vertex_index / max_len_coeffs / max_ntrunc exact,
     - relerror / ext_mat / back_mat / back_mat_ring bitwise (float32).
  2. error paths: duplicate / missing / out-of-range vertex_index ->
     ValueError (group order is never trusted).

Usage (DDA conda env):
    python analysis/mueller_nc_roundtrip_check.py [output.nc]

Exit code 0 = PASS.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
from netCDF4 import Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'adda_python_scripts')
sys.path.insert(0, SCRIPTS)

import adda_control_functions as acf   # noqa: E402
import icosphere_grid as ico           # noqa: E402

FAILURES = []


def check(ok, name):
    print(('PASS  ' if ok else 'FAIL  ') + name)
    if not ok:
        FAILURES.append(name)


def bitwise_equal(a, b):
    """dtype+shape+bit-pattern equality (NaN-safe: compares raw bytes)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def synth_run(n_keep=23, seed=20261106):
    """Synthesize one ARO-run store: real (refined) icosphere vertices,
    only some of them computed (None holes), jagged complex64 SHT coeffs."""
    rng = np.random.default_rng(seed)

    vertices, triangles = ico.make_icosahedric_grid()
    vertices, triangles = ico.refine_icosahedric_grid(vertices, triangles, 1)
    n = vertices.shape[0]

    # deterministic subset of computed vertices (incl. both index extremes)
    keep = set(np.unique(np.append(rng.choice(n, n_keep - 2, replace=False),
                                   [0, n - 1])))

    mueller_array = [None] * n
    back_mat_ring_array = [None] * n
    rel_err_array = np.full((n, 16), np.nan, dtype=np.float32)
    ext_mat_array = np.full((n, 4, 4), np.nan, dtype=np.float32)
    back_mat_array = np.full((n, 4, 4), np.nan, dtype=np.float32)
    max_len_array = np.zeros(n, dtype=np.int64)
    max_ntrunc_array = np.zeros(n, dtype=np.int64)

    ring_len = 17  # one phi grid -> equal ring lengths within a run (shared
                   # root dim 'backring', same as in the per-vertex caches)

    for i in sorted(keep):
        ntrunc = int(rng.choice([2, 3, 5]))
        max_ntrunc_array[i] = ntrunc
        sph = []
        for k in range(16):
            m = int(rng.integers(1, 3 * (ntrunc + 1) ** 2))
            sph.append((rng.standard_normal(m)
                        + 1j * rng.standard_normal(m)).astype(np.complex64))
        mueller_array[i] = sph
        max_len_array[i] = max(len(c) for c in sph)
        rel_err_array[i] = rng.random(16).astype(np.float32) * 1e-2
        ext_mat_array[i] = rng.standard_normal((4, 4)).astype(np.float32)
        back_mat_array[i] = rng.standard_normal((4, 4)).astype(np.float32)
        if i % 3 == 0:
            back_mat_ring_array[i] = (
                rng.standard_normal((ring_len, 4, 4)).astype(np.float32))

    meta = {
        'shape_name': 'RoundtripHabit_Dmax100_Dveq80_Mass1e-9_Freq220',
        'temperature': 223.15,
        'frequency': 220e9,
        'wavelength': 1.362692991e-3,
        'refr_index': complex(-0.0215, 0.00032),
        'adda_version': '1.4.0',
        'adda_binary': '/x/y/adda_1.4/src/seq/adda',
        'max_len_coeffs': int(max_len_array.max()),
        'max_ntrunc': int(max_ntrunc_array.max()),
    }
    return dict(vertices=vertices, mueller_array=mueller_array,
                rel_err_array=rel_err_array, ext_mat_array=ext_mat_array,
                back_mat_array=back_mat_array, max_len_array=max_len_array,
                max_ntrunc_array=max_ntrunc_array,
                back_mat_ring_array=back_mat_ring_array,
                keep=keep, ring_len=ring_len, meta=meta)


def test_roundtrip(run, ncfile):
    acf.write_mueller_array_to_netcdf(
        ncfile, run['mueller_array'], run['vertices'], run['rel_err_array'],
        run['ext_mat_array'], back_mat_array=run['back_mat_array'],
        meta=run['meta'], max_len_array=run['max_len_array'],
        max_ntrunc_array=run['max_ntrunc_array'],
        back_mat_ring_array=run['back_mat_ring_array'])

    n = len(run['mueller_array'])

    # explicit n_vertices and root-attr-derived default must agree
    loaded = acf.load_mueller_array_from_netcdf(ncfile, n)
    loaded_auto = acf.load_mueller_array_from_netcdf(ncfile)
    check(len(loaded) == n and len(loaded_auto) == n,
          'list length == n_vertices (explicit + root attr)')

    holes = [i for i in range(n) if run['mueller_array'][i] is None]
    check(all(loaded[i] is None and loaded_auto[i] is None for i in holes),
          'None holes stay None (%d holes, no padding groups)' % len(holes))
    check(all(loaded[i] is not None for i in run['keep']),
          'computed vertices present at identical indices')

    ncoef = 0
    for i in sorted(run['keep']):
        a, b = run['mueller_array'][i], loaded[i]
        if len(a) != len(b) or not all(bitwise_equal(x, y)
                                       for x, y in zip(a, b)):
            check(False, 'coeff bitwise equality (vertex %d)' % i)
            return
        ncoef += len(a)
    check(True, 'coeff bitwise equality: %d complex64 arrays / %d vertices'
          % (ncoef, len(run['keep'])))
    check(all(loaded_auto[i] is None or
              all(bitwise_equal(x, y) for x, y in
                  zip(loaded_auto[i], run['mueller_array'][i]))
              for i in range(n)),
          'root-attr loader == explicit loader')

    # --- file-level structure / attrs (raw NetCDF4 read) ---
    with Dataset(ncfile, 'r') as rg:
        check(rg.dimensions['stokes'].size == 4
              and rg.dimensions['stokes2'].size == 16,
              'root dims stokes=4 / stokes2=16')
        check(rg.vltypes['float_vlen'].dtype == np.dtype(np.float32),
              'float_vlen type is float32')
        check(rg.n_vertices == n
              and int(rg.max_len_coeffs) == run['meta']['max_len_coeffs']
              and int(rg.max_ntrunc) == run['meta']['max_ntrunc'],
              'root attrs n_vertices / max_len_coeffs / max_ntrunc')
        check(rg.mueller_units == acf.MUELLER_UNITS_NOTE
              and rg.shape_name == run['meta']['shape_name']
              and rg.adda_version == run['meta']['adda_version']
              and rg.adda_binary == run['meta']['adda_binary']
              and rg.created.startswith('20'),
              'root attrs mueller_units / shape_name / adda_* / created')
        check(float(rg.temperature) == run['meta']['temperature']
              and float(rg.frequency) == run['meta']['frequency']
              and float(rg.wavelength) == run['meta']['wavelength'],
              'root attrs temperature / frequency / wavelength (float64 exact)')
        ri = np.asarray(rg.refr_index)
        check(ri.dtype == np.float64
              and ri[0].tobytes()
              == np.float64(run['meta']['refr_index'].real).tobytes()
              and ri[1].tobytes()
              == np.float64(run['meta']['refr_index'].imag).tobytes(),
              'root attr refr_index = [re, im] float64 bitwise')

        ids = {acf.mueller_vertex_id(run['vertices'][i]): i
               for i in sorted(run['keep'])}
        check(set(rg.groups.keys()) == set(ids.keys()),
              'group names == temp-cache zipname_vid ids (%d groups)'
              % len(ids))

        ok_aux = ok_grp = ok_ring = True
        for gname, grp in rg.groups.items():
            i = ids[gname]
            if int(grp.vertex_index) != i:
                ok_grp = False
            if (int(grp.max_len_coeffs) != int(run['max_len_array'][i])
                    or int(grp.max_ntrunc) != int(run['max_ntrunc_array'][i])):
                ok_grp = False
            if not (bitwise_equal(grp['relerror'][:], run['rel_err_array'][i])
                    and bitwise_equal(np.float32(grp['ext_mat'][:]),
                                      run['ext_mat_array'][i])
                    and bitwise_equal(np.float32(grp['back_mat'][:]),
                                      run['back_mat_array'][i])):
                ok_aux = False
            has_ring = 'back_mat_ring' in grp.variables
            if has_ring:
                if not bitwise_equal(grp['back_mat_ring'][:],
                                     run['back_mat_ring_array'][i]):
                    ok_ring = False
            elif run['back_mat_ring_array'][i] is not None:
                ok_ring = False
        check(ok_grp, 'group attrs vertex_index(int32)/max_len/max_ntrunc')
        check(ok_aux, 'relerror/ext_mat/back_mat float32 bitwise')
        check(ok_ring, 'back_mat_ring present+bitwise (variable coverage)')
        check('backring' in rg.dimensions
              and rg.dimensions['backring'].size == run['ring_len'],
              "shared 'backring' dim")


def expect_valueerror(name, fn):
    try:
        fn()
    except ValueError as err:
        check(True, '%s -> ValueError (%s)' % (name, str(err)[:60]))
    except Exception as err:
        check(False, '%s -> wrong exception %s: %s'
              % (name, type(err).__name__, err))
    else:
        check(False, '%s -> no exception' % name)


def fill_vlen(rg, group, name):
    arr = np.ones(16, dtype=object)
    for k in range(16):
        arr[k] = np.array([1.0, 0.0], dtype=np.float32)
    group.createVariable(name, rg.vltypes['float_vlen'], ('stokes2',))[:] = arr


def test_error_paths(run, ncfile):
    """each case runs on its own pristine copy so the FIRST injected defect
    is the one the loader must catch (and group iteration order must never
    be relied on to reach it)"""
    n = len(run['mueller_array'])

    def fresh():
        f = ncfile + '.err%d' % test_error_paths.k
        test_error_paths.k += 1
        shutil.copy(ncfile, f)
        return f

    test_error_paths.k = 0

    def dup():
        f = fresh()
        with Dataset(f, 'a') as rg:
            g = rg.createGroup('vertex_X9999999_Y9999999_Z9999999')
            fill_vlen(rg, g, 'mueller_sph')
            g.vertex_index = np.int32(0)  # duplicate of an existing index
        acf.load_mueller_array_from_netcdf(f, n)

    def missing():
        f = fresh()
        with Dataset(f, 'a') as rg:
            g = rg.createGroup('vertex_X8888888_Y8888888_Z8888888')
            fill_vlen(rg, g, 'mueller_sph')
        acf.load_mueller_array_from_netcdf(f, n)

    def outrange():
        f = fresh()
        with Dataset(f, 'a') as rg:
            for g in rg.groups.values():
                g.vertex_index = np.int32(n + 5)
                break
        acf.load_mueller_array_from_netcdf(f, n)

    expect_valueerror('duplicate vertex_index', dup)
    expect_valueerror('missing vertex_index attr', missing)
    expect_valueerror('out-of-range vertex_index', outrange)

    try:
        acf.load_mueller_array_from_netcdf(ncfile + '.nonexistent', n)
    except FileNotFoundError:
        check(True, 'missing file -> FileNotFoundError')
    except Exception as err:
        check(False, 'missing file -> wrong exception %s' % type(err).__name__)
    else:
        check(False, 'missing file -> no exception')


def main():
    if len(sys.argv) > 1:
        ncfile = sys.argv[1]
    else:
        ncfile = os.path.join(tempfile.mkdtemp(
            prefix='mueller_nc_roundtrip_'), 'test_mueller_data.nc')
    print('test file:', ncfile)

    run = synth_run()
    print('synthesized: %d vertices, %d computed, jagged complex64 coeffs'
          % (run['vertices'].shape[0], len(run['keep'])))

    test_roundtrip(run, ncfile)
    test_error_paths(run, ncfile)

    print()
    if FAILURES:
        print('RESULT: FAIL (%d): %s' % (len(FAILURES), FAILURES))
        return 1
    print('RESULT: PASS — mueller_array NetCDF4 round trip is bitwise '
          'lossless, ordering + error gates hold')
    return 0


if __name__ == '__main__':
    sys.exit(main())
