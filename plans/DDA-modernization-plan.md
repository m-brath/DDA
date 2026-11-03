# DDA Modernization Plan

*Created: 2026-11-03*

Steps requested by user: 1. analyse existing code, 2. update code to latest Python packages (numpy 2), 3. adjust code to use ADDA 1.4 instead of ADDA 1.2, 4. search for optimization (symmetry exploitation).

## Decisions

- **ADDA:** binary path becomes a first-class config option; both `adda_1.2` (rebuilt) and 1.4/1.5 run side-by-side in equivalence tests; winner selected in Phase 3. Target checkout: `adda/` (1.5.0-alpha3) stays, plus a downloaded stable **1.4.0** tree.
- **Compute:** mixed local + HPC → build a proper `noqueue` local mode early (it doubles as the validation harness for every later phase).
- **No old baselines** (HPC results not accessible) → validation rests on: analytic references (sphere/Mie, oblate ellipsoid), 1.2-vs-1.4/1.5 cross-runs, and internal convergence checks (`eps_sphexp`, Romberg/iterative refinement).

## Findings from quick analysis (already done)

- 37× `np.NaN` across 8 files (`definitions_for_netcdf.py`, `adda_process_aro_particles*.py`, `icosphere_grid.py`, `adda_sub_control_spharmonics.py`, `adda_orientational_avg_spharmonics_functions.py`), `numpy.matlib` in 2 files, and `posix_spawn.py` loads Linux `libc.so.6` → breaks import on macOS.
- Binary is hardcoded to `…/adda/adda_1.2/src/mpi/adda_mpi` (`adda_sub_control_spharmonics.py:107`); repo has an unmodified **ADDA 1.5.0-alpha3** checkout in `adda/`, and 1.4.0 is the last stable.
- ADDA CLI diff 1.2 → 1.4/1.5 (checked against upstream changelog + `param.c` of the 1.5 tree):
  - `-eps <n>` semantics **unchanged** (residual exponent) → existing calls keep working.
  - `-orient`, `-m`, `-lambda`, `-eq_rad`, `-store_scat_grid`, `-scat_grid_inp`, `-scat_matr`, `-Csca`, `-alldir_inp` all still exist.
  - **Hard constraint for TRO:** `-orient avg` cannot be combined with `-scat_grid` or `-alldir`/`-Csca` → the built-in averaging only gives cross sections, never Mueller grids. The self-assembled harmonic-expansion approach stays necessary.
  - Parsing of `ampl_scatgrid`/`mueller_scatgrid` is positional → must be verified against 1.4 output before trusting results.
- **Symmetry is currently being thrown away:** in `adda_config_HongPlate_Id9.py` all `sym_XYobj/XZobj/YZobj = False` even though a Hong plate has a clear XY mirror plane; `sym_Zobj` (n-fold rotation about the particle's own axis) is commented out entirely — a hexagonal plate is ~C6, which is a **factor ~6** saving in γ sampling, mirror another factor 2 in β/incidence. Currently every job does `5 β × 10 incidence angles = 50` full ADDA solves per (T, f, size).

## Phase 0 — Safety net

1. `git init` + initial commit of the whole repo (before any edits).
2. Export env: `mamba env export -n DDA > environment.yml`; commit it.
3. Regenerate a small **frozen baseline set** with the *current* stack (ADDA 1.2 rebuilt locally, current scripts) for 2–3 cheap cases: oblate ellipsoid + smallest HongPlate size, 2 frequencies, 1 T. Commit the outputs as golden files. This replaces the lost HPC results and makes every later change measurable.

## Phase 1 — Code analysis (formalize)

4. Annotate the call graph: `adda_main_control.py` → sbatch → `adda_sub_control_spharmonics.py` → `acf.adda_call` → output parsing; list every ADDA option and output-file column the scripts depend on (positional parsing of `ampl_scatgrid`/`mueller_scatgrid` is the fragile spot).
5. `ruff` + `pyright` pass over `adda_python_scripts/` to catch latent bugs pre-migration (fix only clear defects, note the rest).
6. Inventory sheet: habit × size × frequency × T matrix with status (done / pending / size-param-violating).

## Phase 2 — Python modernization (numpy 2)

7. `np.NaN` → `np.nan` (37 sites, 8 files); `numpy.matlib` → `np.tile` in `adda_sub_control_spharmonics.py` + `adda_orientational_avg_spharmonics_functions.py`.
8. Platform-guard `posix_spawn.py` (Linux ctypes path vs. `subprocess` fallback on macOS) so `adda_control_functions` imports everywhere.
9. Sweep remaining numpy-2/scipy drift (scalar reprs, `copy` semantics).
10. **Gate:** golden outputs from Phase 0 must be bit-for-bit identical. No behavior changes allowed in this phase.

## Phase 3 — ADDA 1.4/1.5 integration (configurable binary)

11. Introduce `adda_binary` + `adda_version` in configs (defaults per hostname), remove the hardcoded `adda_1.2/src/mpi/adda_mpi` from `adda_sub_control_spharmonics.py:107`.
12. Build matrix: `make seq` / `make mpi` locally for 1.2, 1.4.0 (new download), 1.5.0-alpha3; same on the cluster.
13. Add the local `noqueue` path: `adda_main_control.py` gets a `calc_type='local'` branch calling `adda_sub_control_*.py` directly (no sbatch) with local paths — this is the workhorse for all validation runs.
14. Equivalence suite (the real deliverable of this phase):
    - `-shape sphere` vs. Mie at x ≈ 1, 5 (both binaries);
    - oblate ellipsoid vs. analytic/1.2 reference;
    - one small HongPlate case, 1.2 vs 1.4 vs 1.5: verify `ampl_scatgrid`/`mueller_scatgrid` column layouts match the parser, Mueller elements agree within `eps`;
    - `avg_params.dat` compatibility check for the TRO path.
15. Pick the production ADDA (expect 1.5.0-alpha3 or 1.4.0, both fine) and record the choice + rationale in `agents.md`.

## Phase 4 — Optimization (symmetry first)

16. **Symmetry flags per habit** — the big lever:
    - Determine true symmetries per habit (numerically: rotate the `.adda` dipole set by the candidate symmetry operation, compare point sets) instead of hand-guessing;
    - Hong/Liu plates & columns: `sym_XYobj=True` (mirror) → halve β and incidence-angle ranges; hexagonal habits: check C6/C3 `sym_Zobj` → γ ∈ [0°, 60°) — up to **~12× fewer orientation samples** vs. today's setup (all flags currently `False`).
    - Re-run golden HongPlate case with symmetries on, verify agreement within `eps_sphexp` and measure the speedup.
17. **Quadrature:** uniform β grid (5 pts, 0–180°) → Gauss–Legendre nodes on the symmetry-reduced range, with the existing iterative refinement as convergence proof; also revisit `incidence_angles=np.linspace(0,180,10)` → 0–90° where mirror symmetry allows.
18. **ADDA-internal benchmarks** on the ellipsoid (analytic truth): `-eps` 3 vs 2, solver/prescription choices, dipole density vs. `size_parameter_limit`; produce a small tuning table per size-parameter regime.
19. **Job structure:** evaluate running the 50 orientation solves per (T,f,size) with orientation-level parallelism (separate parallel ADDA runs for different β/incidence in one job, or job arrays) — measure against current sequential layout.
20. Document measured speedups in `agents.md`.

## Phase 5 — Production backlog

21. TRO: `HongPlate_Id9_TOR_sizegrid` end-to-end (validates the modernized stack + TRO strategy; decide `-orient avg` for cross sections only vs. own sampling — note 1.4/1.5 forbid `-orient avg` + scattering grids, so Mueller data must come from own sampling anyway).
22. ARO queue: HongColumn_Id7 → IconSnow_Id28 → LiuSectorSnowflake_Id3.
23. Post-processing → SSDB NetCDF (`adda_process_aro_particles.py` → `adda2SSDBnetcdf_functions.py`), then sanity-check against the published ARTS SSPDB habits via `arts-sspdb-interface` where overlapping habits exist.

## Order rationale & risks

**Order:** Phase 2 before 3 isolates Python-vs-ADDA diffs; 3 before 4 so speedups are measured against validated 1.4/1.5 results. Each phase has an explicit numerical gate before the next begins.

**Main risks:**
- Output-format drift between ADDA versions (mitigated by step 14).
- `eps_sphexp` convergence may change when symmetry shrinks sampling ranges (mitigated by re-validation run in step 16).
- shtns rebuild fragility on arm64 (mitigated by env pinning in step 2).
