# Backscatter Matrix for ARO Particles — Implementation Plan

Status: **implemented & gated 2026-11-06** — all 23 gate checks PASS
(`analysis/phase6_results.json`); findings in `plans/DDA-modernization-report.md`
§6. Decisions taken: pipeline/scattering-plane convention; SSDB version bump
to `beta_0.9.1`; TRO deferred. One plan assumption corrected during
implementation: the raw mueller grid is k-normalized (not µm²/sr) — the
m²/sr conversion happens in `rawdata2average` via the same lambda²/4π²
factor `mueller2phase` uses.

## Motivation

So far every ARO ADDA call stores the extinction matrix (exact forward
direction, optical theorem) and the Mueller matrix as a spherical-harmonic
(SH) expansion over scattering directions. For radar applications the exact
**backscatter** direction matters most, but as stated in Brath et al. 2020
(AMT 13, 2309, Sect. 4/5): "In the spherical harmonics representation of the
Mueller matrix, the polarization at the poles, which are in the forward and
backward direction, is not well represented. ... there can be some residual
effects for the polarization at the backscattering direction. This will be
revised for the next iteration of the database."

Remedy: store the exact 4x4 backscatter Mueller matrix per run and carry it
through orientation averaging and SSDB export — directly analogous to the
extinction matrix, bypassing the SH expansion entirely.

## Key finding (verified in code)

No new ADDA computation is needed:

- `adda_python_scripts/scat_params.dat` uses theta range 0–180 deg, N=181 →
  `mueller_scatgrid` already contains the rows **Theta=180** (all 361 Phi
  values), next to the Theta=0 rows that feed the extinction matrix.
- ADDA's scat grid is **propagation-ATTACHED**: `adda_1.2/src/crosssec.c::
  CalcScatGrid` builds `observer = cos(Theta)*prop + sin(Theta)*(cos(Phi)*incPolX
  + sin(Phi)*incPolY)`. So Theta=180 is exact monostatic backscatter for that
  vertex's incidence direction; the Phi-ring at Theta=180 is the
  receiver-plane dependence (receiver basis rigid to the incidence basis,
  at Phi=0: output basis = (-incPolX, -incPolY)).
- This is the same kind of object from which `ext_matrix` is derived at
  Theta=0 → "back matrix analogous to ext matrix" = parse the Theta=180 row
  the way row 0 is parsed for extinction. Free of the SH-pole problem.
- No ADDA CLI change, no extra solves → golden baselines (`baselines/`) stay
  valid; existing solver numerics unchanged.

## Geometry/basis background needed for the implementation

Per-vertex ADDA run: icosphere vertex v = incidence direction in the fixed
particle frame (`-prop v`); mueller grid angles are relative to v. The
Theta=180, Phi=0 value `B_v` is the 4x4 backscatter Mueller in the vertex's
incidence-attached basis (input basis ADDA incPolX/Y at v; output basis
rigid to it). `aoa.create_vpol_vector` + `aoa.pol_changer` map vertex bases
to lab bases via the angle phi_i around the propagation axis; for the ext
matrix pol_changer does `R(phi_s) K R(phi_i)`; the SH mueller path uses
`Zrotate`. For a raw 4x4 back matrix BOTH input and output Stokes bases
rotate (same phi_i since the receiver basis is propagation-rigid) —
new branch needed (see Step 3).

Sign-convention anchor: ADDA Theta=180/Phi=0 output basis = -incPolX/-incPolY
(a pi rotation of both bases = the diag(1,1,-1,-1) `D` already used by the
Phase-4 mirror machinery `acf.MIRROR_D` / `construct_by_symmetry`). The
pipeline-internal convention is defined as: exact lab-frame phase-matrix
value Z_aro(theta_inc, theta_s, dphi) evaluated AT backscatter, i.e. exactly
consistent with the exported phase matrix by construction (only the SH
interpolation error is removed). Convention note + ARTS assp2backcoef
(4Pi) consumption remark go into the NetCDF description/attributes.

## Implementation steps

### Step 1 — extraction: `adda_control_functions.py` (`acf.adda_call`)

- After `mueller_scatgrid` is loaded/reshaped (existing `mueller_array`
  grid), extract the Theta=180 ring: `back_mat_ring` =
  `mueller_array[theta==180, :, :]` (361x4x4 real) and
  `back_matrix` = the Phi=0 slice (4x4 real).
- Add to output dict: `output_data['back_mat_ring']`,
  `output_data['back_matrix']`.
- Extend per-vertex cache: `write_data_to_separate_netcdf` /
  `read_data_from_separate_netcdf` / `write_variables` /
  `read_fulldata_from_netcdf`: new group variables `back_mat` (4x4) and
  `back_mat_ring` (361x4x4 float32). Keep backwards compatibility: readers
  tolerate stores without the new variables (return None/missing key →
  caller treats as "needs re-run" or derives what it can).

### Step 2 — run orchestration: `adda_sub_control_spharmonics.py`

- Parallel to `ext_mat_array`: allocate `back_mat_array` (Nvertex x 4 x 4,
  NaN-filled) in BOTH allocation branches (ref_lev==0 and refinement pass);
  include in the `*_old` bookkeeping for the vertices_old reuse branch
  (copy `back_mat_array[point_i]` from `r_idx` alongside ext).
- Store `output_data['back_matrix']` (+ring for mirror fill) per vertex.
- `mirror_fill()`: the Phase-4 mirror transform applies unchanged to the
  Theta=180 ring: `mueller_S(Theta,Phi) = mueller(Theta, Phi_map(Phi)) *
  outer(D,D)` (see `acf.MIRROR_AXES`, `MIRROR_PHIMAP`, `mirror_signs`,
  `mirror_phi_index_map` and analysis/phase4_grid_symmetry.py verification).
  Fill mirrored vertices' rings + Phi=0 matrix + identical NetCDF cache
  entries, as the current fill-in does for the mueller grid + ext matrix.
- Save `back_mat_array` into the overview `.mat` (`output_data`), like
  `ext_mat_array`. (Averaging happens in post-processing — do NOT duplicate
  the whole alpha/gamma loop inside sub_control; optional cheap addition:
  also average the Phi=0 back matrix in the existing ext-averaging loop for
  the overview file, but it is not the exported quantity.)

### Step 3 — orientation averaging: `adda_orientational_avg_spharmonics_functions.py` (post-processing; this produces the exported averages)

- `pol_changer`: new branch for a raw 4x4 Mueller matrix (list/array of
  per-vertex 4x4): transform BOTH sides — output Stokes rotation R(phi_s)
  left, input R(phi_i) right (mind: ext uses `KxP1 = K·R(phi_i)` then
  `R(phi_s)·(KxP1)`; for the 4x4 Mueller the input side must be the inverse
  rotation, i.e. R(-phi_i) = rotater_mat(-phi_i), applied from the right;
  verify against the sphere test, not by derivation alone).
- `orientational_avg()`: accept `back_mat_array`; interpolate at `inc_dir`
  with the same weights `w_int`, idx + symmetry handling (plain averaging —
  Phase-4 fill-in guarantees complete vertices) and the new pol_changer
  branch; accumulate alpha/gamma GL sums exactly like `ext_mat`; return
  `back_mat_alpha` as a third element (update all call sites unpacking the
  return, incl. the `Parallel(...)` results usage in `rawdata2average`).
- `rawdata2average()`: read `back_mat_array` from the overview mat (tolerate
  old stores: if absent, skip and warn); allocate `back_mat_horiz_avg`
  (n_beta x n_za_inc x 4 x 4, NaN); fill from results; convert um^2/sr →
  m^2/sr (×1e-12) like ext; add `output_data['back_mat_horiz_avg']` (and
  pass through `back_mat_array` like ext_mat_array).

### Step 4 — SSDB export: `definitions_for_netcdf.py`, `adda2SSDBnetcdf_functions.py`

- `definitions_for_netcdf.py`:
  - `version = 'beta_0.9.1'`
  - new Group0 variables `backMat_index` (i1, dims (scatMat_Col,scatMat_row),
    value = same 4x4 index map as `phaMat_index`) and `backMat_data`
    (f4, dims (phaMatElem, aa_inc, za_inc) — flat-16 element order identical
    to `phaMat_data`, single real field, units m^2/sr), plus matching
    RAW_variables entry `back_mat_horiz_avg`, fill value np.nan, and
    descriptions explaining the convention: exact backscatter (Theta=180)
    Mueller matrix, receiver basis attached to the incidence plane (lab
    phase-matrix value at exact backscatter, no SH interpolation); for
    radar Cb use the ARTS/atmlab assp2backcoef 4Pi convention.
- `adda2SSDBnetcdf_functions.py::put_variables_in_group0`: new branch for
  `back_mat_horiz_avg` — index like ext (raw_temp=output_data[...][b_i],
  transpose (1,2,3,0)→(za, elem-flat, aa) order consistent with phaMat_data
  handling), write all 16 elements (do not thin by index).
- Keep the export idempotent/compatible: consumers of old files unaffected
  (new variables are additive).

### Step 5 — verification: new `analysis/phase6_backscatter.py` (gate script)

1. **Mie sphere anchor** (convention pin, analytic): homogeneous sphere,
   known S1(180), S2(180) → analytic 4x4 backscatter Mueller. Run small
   sphere with production binary (ADDA 1.4.0) through `acf.adda_call` and
   compare `back_matrix` ≤ solver noise (gates must be NOISE-based, not
   bit-level: ADDA re-runs are not bit-reproducible at -eps 3). This pins
   all signs incl. Z12/Z21/Z34/Z43.
2. **Oblate ellipsoid** (`adda_config_ellipsoid_oblate.py`, moderate x):
   exact back matrix vs. SH-reconstructed phase matrix evaluated at
   Theta=180 — expect agreement at small x (Rayleigh) and a deviation of
   the kind documented in Brath 2020 at larger x (quantifies the pole
   problem being fixed; report the numbers).
3. **ARO smoke run** (local no-queue mode, small habit/size, sym flags on):
   full pipeline run → check cache netcdf round-trip of back_mat(+ring),
   mirror-filled rings consistent with directly-computed twin vertices,
   `back_mat_horiz_avg` alpha/gamma convergence behavior analogous to
   `ext_mat_horiz_avg`, `backMat_data` present + re-readable in the SSD
   NetCDF tree (SSDB_EXPORT_NETCDF=1).

Update `plans/DDA-modernization-report.md` with a new section (findings,
convention derivation/verification numbers, deferred items).

## Constraints & gotchas

- Environment: mamba env `DDA` (py3.12, numpy 2, scipy 1.18, netcdf4,
  shtns, typhon 0.10.0). ADDA 1.4.0 production binary built locally
  (`adda_1.4/`); local mode = `calc_type='local'` branch (no queue).
- Do NOT change ADDA CLI/options anywhere in the ARO path (no new flags) —
  only parse extra rows from already-produced `mueller_scatgrid`.
- ADDA output parsing is positional — `mueller_scatgrid` row order is
  theta-outer/phi-inner (verified in crosssec.c); Theta=180 block = last
  N_phi rows; Phi=0 = first column of that block. Do not `np.loadtxt`
  re-sort.
- Old completed ARO runs CANNOT be back-processed (raw grids deleted, cache
  stores only mueller_sph + ext_mat) — backMat appears only in new runs.
  Backlog HPC submissions will carry it.
- Do not touch `SSD_scripts_deprecated/`, `old_configs/`, deprecated
  `adda_sub_control_gridref.py` / `adda_orientational_avg_spharmonics_for_testing.py`.
- Keep config contract intact (no new required config globals needed; if a
  toggle like `store_backscatter=True` is added, sub-control must default
  it ON via getattr fallback for old configs).
- mueller_array values are real (Mueller) but stored float; ring cache in
  float32 to keep files small.
- TRO deferred (`-orient avg` has no scat grid; different extraction path).

## Out of scope

- No radar product variables (sigma0, Cb scalars) — consumers derive from
  `backMat_data`.
- No re-run/re-export of old HPC production data.
- No change to golden baselines (`baselines/`) — solver outputs unchanged.
