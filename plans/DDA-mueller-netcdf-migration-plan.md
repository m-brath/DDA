# Plan: Replace `*_mueller_data.pkl` with NetCDF4 (`*_mueller_data.nc`)

## Goal

The internal ADDA-results artifact (spherical-harmonic-expanded Mueller matrices per
icosphere vertex) is currently stored as a Python **pickle** (`*_mueller_data.pkl`), a
replacement for the old ARTS binary XML that broke when typhon dropped `ty.arts.xml`
(report Phase 2). Pickle is Python-version- and platform-coupled and opaque. Replace it
with a **single NetCDF4 file per (T, f, size) run**, using the schema that already exists
in this project for the per-vertex cache files.

Requirements met:
- **Platform independent** — NetCDF4/HDF5, self-describing, readable by any language.
- **Fast** — same path the pipeline already uses for its per-vertex cache files; arrays
  are small (complex64 vlen), overhead negligible.
- **Easy to use** — zero new dependencies (`netcdf4` already in the DDA conda env),
  inspectable with `ncdump`, consistent with the temp-cache and SSDB-export formats.

## Current state (verified)

| Piece | Location | Notes |
|---|---|---|
| pickle write | `adda_sub_control_spharmonics.py:1010-1012` | dumps `mueller_array`: list of N_vertices entries, each `None` or list of 16 ragged `complex64` coefficient arrays |
| unused name | `adda_sub_control_spharmonics.py:954` | `netcdf_name = ..._mueller_data.nc` already defined, never written |
| existing NC schema | `adda_control_functions.py`: `append_data_to_netcdf` / `write_variables` (1418+) | one group per particle; `mueller_sph` = vlen float32 arrays (complex64 `.view(np.float32)`), `relerror`, `ext_mat`, optional `back_mat` + `back_mat_ring`, group attrs `max_len_coeffs`/`max_ntrunc`; dims `stokes`(4)/`stokes2`(16) |
| existing NC readers | same file: `read_data_from_netcdf` (1341), `read_fulldata_from_netcdf` (1379) | `read_fulldata…` returns groups in **file order** — must not be reused naively for ordering |
| pickle read | `adda_orientational_avg_spharmonics_functions.py:1633` (`rawdata2average`) | expects list **indexed by vertex order** (same order as `vertices` in the `.mat` overview); lines 1636-1644 crash on `None` entries, but `orientational_avg` tolerates them (line 1405) |
| idempotency glob | `adda_main_control.py:191-193` | accepts `*mueller_data.xml` + `*mueller_data.pkl`; must also accept `.nc` |
| temp cache | `netcdf_tempname = ..._temp_data_` + `vertex_*.nc` | one file per vertex, DELETED at end of run (sub_control:1036); keep as-is |

## Format specification (final artifact `*_mueller_data.nc`)

NetCDF4, one file per (T, f, size) run, reusing the exact temp-cache variable layout:

- Root attrs (provenance, mirrors the `.mat` overview): `shape_name`, `temperature`,
  `frequency`, `wavelength`, `refr_index`, `adda_version`, `adda_binary`,
  `mueller_units` ('raw k-normalized mueller-grid units, m^2/sr conversion in
  rawdata2average'), `n_vertices`, `max_len_coeffs`, `max_ntrunc`, creation timestamp.
- Dimensions (root, shared): `stokes=4`, `stokes2=16`; vlen type `float_vlen` (float32).
- One **group per computed vertex**, group name = the existing `zipname_vid` id
  (`vertex_X{XV:07.0f}_Y{YV:07.0f}_Z{ZV:07.0f}`, coords ×1e5 µm) — same as cache files —
  containing `mueller_sph` (16 vlen float arrays), `relerror`, `ext_mat`,
  `back_mat`, `back_mat_ring` (present when the run has them).
- Group attr **`vertex_index` (int32)** = row index into the run's `vertices` array.
  This is the ordering key: the reader rebuilds the N_vertices list (filled at
  `vertex_index`, `None` elsewhere) so all downstream indexing stays identical.
  Group-name-sorted iteration is explicitly NOT trusted for order.
- Uncomputed vertices simply have no group (no padding, no wasted space).

Complex convention: keep the existing `complex64 → .view(np.float32)` on write and
`vlen → .view(np.complex64)` on read — both writer and reader sides already do this in
`write_variables` / `read_data_from_netcdf`; the round-trip is lossless.

## Changes

### 1. `adda_control_functions.py` — new save/load helpers
- `write_mueller_array_to_netcdf(filename, mueller_array, vertices, rel_err_array,
  ext_mat_array, back_mat_array, meta)`: single `Dataset(..., 'w')`, loop over non-None
  vertices calling the existing `write_variables` with absolute group paths, set
  `vertex_index` + root attrs.
- `load_mueller_array_from_netcdf(filename, n_vertices)`: returns the vertex-ordered
  Python list (`None` for missing), using group attr `vertex_index`; raises
  `ValueError` on missing/duplicate indices.
- Do not touch the per-vertex temp-cache functions (still used as intra-run cache).

### 2. `adda_sub_control_spharmonics.py`
- Replace the `pickle.dump` block (~1010) with
  `acf.write_mueller_array_to_netcdf(netcdf_name, ...)` using the already-defined
  `netcdf_name`; write to a temp name + `os.replace` for atomicity (interrupted runs
  must not leave a half-written final artifact).
- Remove the now-unused `import pickle` (keep `copy.deepcopy` usage intact).
- Check the sub-control's own "already done / skip" logic: if it tests for artifact
  existence, make it accept `*_mueller_data.nc` (and keep legacy pkl/xml acceptance).

### 3. `adda_orientational_avg_spharmonics_functions.py` (`rawdata2average`)
- Replace `pickle.load(...)` at 1633 with `acf.load_mueller_array_from_netcdf(...)`:
  `n_vertices` comes from `overviewdata['vertices']` (already read above).
- Keep the complex64 cast loop harmless/consistent (the loader already yields
  complex64→complex128 views exactly like the pkl path did — verify no double downcast
  changes results; the loop at 1636 stays as-is).
- Legacy fallback: if `*.nc` is absent but a `.pkl` sibling exists, load the pkl and
  print a deprecation warning (old artifacts on HPC keep working).

### 4. `adda_main_control.py`
- Line 193 glob: add `*mueller_data.nc` to the completed-case patterns (keep `.xml`/
  `.pkl` for old results). Make sure the glob does not match the `_temp_data_*` files
  (it cannot — different suffix).

### 5. Migration script `analysis/migrate_pkl_to_nc.py`
- For every `*_mueller_data.pkl` next to its `*_overview_data.mat` (dir arg or repo
  results tree): load pkl + mat, derive `vertex_index` from the pkl list order + mat
  `vertices` (via `zipname_vid`), write the new `.nc`.
- Verify: reload with `load_mueller_array_from_netcdf`, assert element-wise `==`
  against the pkl (lossless; this is exact, not noise-based — it is raw stored data,
  not solver output).
- Keep the `.pkl` in place (rename to `.pkl.migrated` only on explicit `--prune`);
  never delete automatically.

### 6. Documentation
- New section in `plans/DDA-modernization-report.md` (format decision + verification
  numbers).
- Update `agents.md`: mueller artifacts are NetCDF4 again ("internal Mueller artifact"
  lines), note `.nc` schema and legacy fallback.

## Verification / gate

1. **Round-trip unit check** (no ADDA run needed): synthesize a random jagged
   `mueller_array` (incl. `None` holes + back_mat rings), write nc, reload, assert
   bitwise equality of all complex64 coefficients and attrs.
2. **Pipeline e2e**: run one small ARO case (e.g. `adda_config_ellipsoid_oblate.py` or
   `adda_config_phase3_test.py`) with the change; confirm `_mueller_data.nc` exists,
   temp `vertex_*.nc` are cleaned, no pickle left, SSDB export proceeds.
3. **Equivalence gate**: run `rawdata2average` on the same case twice — once reading
   the new `.nc`, once reading a pkl produced from the same run (keep both writers on in
   a probe run) — downstream averaged/SSDB outputs must be **bit-identical** (safe
   expectation: unlike ADDA re-runs, this artifact is stored raw data; the nc round-trip
   of complex64 is lossless).
4. Run existing golden-baseline check (`baselines/check_baselines.py`) to confirm
   nothing else moved — the baselines cover ADDA outputs, so no baseline regeneration
   is needed (artifact format is internal, noise gates unaffected).

## Out of scope / notes

- `*_overview_data.mat` stays (already platform-independent).
- Per-vertex `_temp_data_vertex_*.nc` cache files stay as-is (inter-run cache could be
  folded into the single file later — not now).
- Old HPC `.pkl`/`.xml` artifacts: handled by the legacy read fallback + migration
  script when they become accessible again.
- `adda_process_aro_particles.py` reaches the data only through `rawdata2average`, so
  no changes there.

## Suggested execution order

1. Helpers + round-trip unit check (acf).
2. Writer switch (sub_control) + reader switch (rawdata2average, with pkl fallback)
   + main_control glob.
3. e2e pipeline case + double-writer equivalence gate.
4. Migration script over existing local artifacts.
5. Docs (report + agents.md).
