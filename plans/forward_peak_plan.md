# Phase 9 (candidate) — Analytic forward-peak model + residual storage

Status: **9A DONE (2026-11-09) — conditional GO** (`analysis/phase9_fwdmodel_scan.py`,
results `analysis/phase9_results.json`, table `analysis/phase9_fwdmodel_table.txt`).
Findings written to `plans/DDA-modernization-report.md` §9. Headline: conventions
fully pinned (c≈1.005, origin = scatterer center, Mie triple-validated); the eikonal
screen model wins big where valid (face/near-face plate: full-eps ntrunc 25→6 at
x=5, 179→81 at x=10) but is actively harmful at grazing orientations (|g|/|f| up to
1.73, negative Cext_model) ⇒ 9B must be a **per-cell gated hybrid**, not a blanket
subtraction; SSDB-format hybrid = no-go (internal 9B only).

## Motivation (corrected framing)

The old note in `docs/DDA-manual.md` ("at very small x the forward peak dominates")
is wrong: in the Rayleigh regime (x ≪ 1) the pattern is ≈ cos²θ with NO spike.
The diffraction/forward peak dominates at **x ≫ 1**: C_ext → 2·G (extinction
paradox), peak width Δθ ~ λ/D ~ 1/x, Airy-like rings set by the **projected
silhouette** (not D_veq).

At our production cap x ≤ 10 the peak is only moderately narrow — but it is
exactly what the Phase-8 grid laws are scaled to: `scat_grid_sizes` theta law
ntheta ≈ πx/κ (feature width ~1/x) and the empirically stored SH truncations
(81 at x=5, margin-6 cap at x=10) are the SH signature of the same peak.
Subtracting a model of the peak is the natural next lever: the **residual's**
angular band-limit stops scaling like x.

Analogy: delta-Eddington/delta-M (RT) replaces the peak by a delta function and
preserves only moments. Here we do the opposite: keep the peak **resolved**, but
represent it by an analytic model instead of numeric grid/SH content. The RT
consumer would evaluate model + residual at arbitrary quadrature angles.

## Idea

f = F_model + g (amplitude level). Store g (smooth, small) instead of f;
reconstruct f on demand. Amplitude level, NOT Mueller level: M = |f|² is
quadratic; storing M − |F|² loses the cross term 2Re(F*⊗g) unless the consumer
has F anyway — with (F, g) everything is algebraically exact.

## Forward model for the general (non-spherical) case

Exact Mie exists only for spheres. Candidates:

1. **Eikonal internal-field radiation integral (primary):** assume the internal
   field is eikonal, φ(r) = e^{ik∫m ds} along the incident ray; radiate it:
   f_model = (k²/2π)∫(m²−1)φ(r)e^{−ik n̂·r}d³r. Ray-marched directly on the
   `.adda` voxel geometry — exact 3D phases (no screen-plane shortcut), valid
   for all θ (degrades gracefully away from forward), handles arbitrary
   habits/orientations, computable from data we already ship.
2. Projected-area Fraunhofer (screen at plane through origin) — baseline only.
3. Equivalent-sphere Mie — empirical baseline only (right 1/x width, wrong ring
   structure for plates).

Model accuracy does NOT have to be good in absolute terms: g ≡ f − F is exact
by construction; the model only needs to capture enough of the peak that g is
smoother/smaller. Failure mode is graceful (residual ≈ full field, no gain).

Kill-risks being measured by the scan:
- **Phase/normalization conventions** (cross term is at peak amplitude — a phase
  slip is fatal). Pinned by the exact-Mie sphere pilot (ADDA grid = BH s1..s4
  columns, ampl = −i·k·S-like constant factor, far-field phase origin = lattice
  index (0,0,0) of ADDA's *shifted* dipole box — analytic shapes centered,
  read-shapes bbox-corner; all verified empirically).
- **theta=180 / specular**: eikonal is not a backscatter model; Phase-6
  backscatter ring stays computed (never modeled). Scan quantifies |F| at
  θ=180 to define the exemption rule.

## Where savings appear (honest)

- Internal (unconditional): ADDA far-field grid (relax ntheta∝x law), shtns
  cost + achieved ntrunc in `mueller_data.nc`, per-vertex caches, FFT far-field
  memory. ADDA *solve* cost unchanged (scatgrid is post-processing).
- SSDB export (conditional): `sph_coeffs_scat` stores SH of the FULL Mueller.
  Storing residual-coeffs + model-coeffs is WORSE (the model's SH tail IS the
  expensive content). A real export win requires a hybrid product
  (residual coeffs + model descriptor + consumer-side reconstruction) —
  format change + version bump + `arts-sspdb-interface`/RTTOV work. Decision
  deferred until the scan measures the residual-ntrunc win.

## Scope

ARO path only. TRO production uses ADDA `-orient avg` (no scattering grid at
all), so this machinery does not apply there. No changes to `acf`, sub-control
scripts, configs, exports, or baselines in 9A.

## Phase 9A — scan (analysis only)

`analysis/phase9_fwdmodel_scan.py`, ADDA 1.4.0, production static grid
(181×361), production command mirror (`-shape read`, `-eps 3`, eq_rad=D/2):

- **G1 convention anchor (sphere, exact Mie):** analytic sphere x = 1 and 2;
  fit the complex constant c and the linear-in-n̂ origin phase → recover the
  lattice-index origin; must match the box-corner prediction from ADDA stdout.
- **G2 eikonal scan (voxel shapes):** HongPlate_Id9 at f=887 GHz (the size grid
  hits x = 1, 2, 5, 10 exactly: Dveq = 108/215/541/1078 µm) × orientations
  {normal, 60°, edge-on}; HongColumn_Id7 at x≈2, 5 × 2 orientations; T=230 K,
  Matzler eps. Metrics per cell:
  1. fitted complex calibration (model·c vs full L2) + fitted origin phase —
     should be ≈(1, 0) if conventions hold;
  2. ‖g‖/‖f‖ (amplitude L2, split forward cone θ<30°/rest) and intensity-weighted;
  3. **decisive: achieved SH ntrunc via `spha.expand_mueller_2_spharmonics`
     (production eps_sphexp=0.005 formula) for full M vs M(g)**;
  4. C_ext: ADDA forward-amplitude value vs model-only, and the identity
     Cext(f)=Cext(F+g) (algebraic check, should be exact);
  5. model |F| at θ=180 vs full M11 at θ=180 (backscatter exemption evidence);
  6. cross-pol: |g3|+|g4| vs |full off-diag| near forward.
- **G3 Fraunhofer baseline** (screen at plane through origin) on the best/worst
  cell to bound the ray-march model against the 2D shortcut.

## Decision outputs (documented in report §9, no pipeline code yet)

- go / no-go, activation threshold x_min;
- global vs forward-cone model support and the theta=180 exemption form;
- eps_sphexp renormalization rule for residual runs;
- measured ntrunc / storage / grid savings → feeds Phase 9B (internal
  integration: acf model builder, residual caches + reconstruction in
  `rawdata2average`, new grid laws + twin gate) and the SSDB-hybrid decision.

## Later (only if 9A passes)

- **9B** internal integration (ARO): config switch `scatmodel_*`, residual +
  model stored in vertex cache / `mueller_data.nc` (provenance attrs),
  reconstruction on load, relaxed adaptive grid laws, twin gates (noise-based,
  never bit-level — Phase-3 caveat).
- **9C** export decision: Option 1 reconstruct-before-export (format untouched,
  internal-only wins) vs Option 2 hybrid SSDB product
  (`scatmodel_id`, residual coeffs at low ntrunc, model descriptor; consumer
  evaluates the peak analytically — better angular resolution at equal volume;
  needs ARTS/RTTOV-side implementation + database version bump).
- **9D gates:** round-trip |F+g−f| ≤ solver-noise plateau, ext identical to
  solver noise, end-to-end twins at Phase-8 parity, size-reduction report,
  full existing gate suite re-run.
