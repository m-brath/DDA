# DDA Project — Single Scattering Properties of Randomly Orientated Hydrometeors

## Project Goal

Calculate **single scattering properties** (extinction/absorption cross sections and
Mueller matrices) for **azimuthally random oriented (ARO)** and **totally random
oriented (TRO)** particulate habits (ice crystals, snowflakes, liquid water) using the
**Discrete Dipole Approximation (ADDA)**. Results are intended for the **ARTS Microwave
Single Scattering Properties Database (SSDB)** and radiative-transfer use (RTTOV).

Orientation strategies:
- **ARO** — particle orientation is random except that its symmetry/maximum axis stays
  horizontal; handled by integrating over a set of Euler-β tilt angles (and the azimuthal
  incidence angles), then expanding the result in **spherical harmonics** (methodology:
  Brath et al. 2020).
- **TRO** — fully random orientation; handled either by ADDA's built-in
  `-orient avg` (Romberg integration over α,β,γ via `avg_params.dat`) or by an
  icosphere sampling of orientations (`*_RndOrient` scripts, see below).

## Repository Layout

| Directory | Contents |
|---|---|
| `adda/` | ADDA v1.5.0-alpha3 source (latest, unmodified upstream checkout). **No compiled binaries present.** |
| `adda_1.2/` | ADDA v1.2 source tree, previously compiled on a Linux cluster (`src/seq/adda`, `src/mpi/adda_mpi` are ELF x86-64 — **do not run on this Mac, rebuild with `make seq` / `make mpi`**). |
| `adda-gui/` | Official Java GUI for ADDA (Gradle project). Reference only; not used in the automation workflow. |
| `adda_python_scripts/` | The orchestration workflow (see below). Core of the project. |
| `adda_shape_files/` | Particle geometry per habit: `*.adda` (dipole positions) + `*.meta.mat` (metadata: Dmax, volume-equivalent diameter, mass, min frequency, axis ratios). Naming: `Dmax<µm>_Dveq<µm>_Mass<kg>_Freq<GHz>.adda`. |
| `adda_aux_input/` | ADDA `avg_params.dat` files (Euler-angle integration control for built-in `-orient avg`) for EvanSnowflake_aggregate and hexPlateAggregates2. |
| `arts-sspdb-interface/` | Upstream MATLAB/Python interface for browsing/importing the published ARTS SSP database (zenodo 1175572), incl. RTTOV extraction. Read-only reference. |
| `SSD_scripts_deprecated/` | Old/prototype SSDB interface scripts (`assp.py`, `sph.py`, `rttov.py`, …). **Deprecated — do not modify or extend.** |
| `plans/` | Working plans (`DDA-modernization-plan.md`) and the running phase-findings report (`DDA-modernization-report.md` — updated after every phase). |
| `baselines/` | Frozen regression baselines (golden ADDA outputs) + `make_baseline.py` / `check_baselines.py`. See `baselines/README.md`. **Do not regenerate casually; they are the reference for all migrations.** |
| `analysis/` | Phase-1 analysis tools (e.g. `inventory.py` — habit×size×freq×T status sheet). |
| `docs/` | **Project manual** (`DDA-manual.md` — theory, architecture, runbooks, gotcha catalogue; generated `.html`/`.pdf`; data + Mermaid figures). Regenerate figures with `docs/make_figures.py`, HTML/PDF with `docs/build_html.py`; commit regenerated figures. |

### Habits (shape folders / active configs)

| Habit folder | Type | Config in `adda_python_scripts/` |
|---|---|---|
| `HexPlaAggCrystal_Id20` | hexagonal plate aggregate crystal (ID20) | `adda_config_HexPlaAggCrystal_Id20.py` |
| `HongPlate_Id9` | Hong/MoCOLAG plate (ID9) | `adda_config_HongPlate_Id9.py` |
| `HongPlate_Id9_TOR_sizegrid` | same habit, size grid prepared for **TRO** runs | (reuse `adda_config_HongPlate_Id9.py` / new TRO config) |
| `HongColumn_Id7` | Hong/MoCOLAG column (ID7) | `adda_config_HongColumn_Id7.py` |
| `IconSnow_Id28` | Icon Initiative snowflake (ID28) | `adda_config_IconSnow_Id28.py` |
| `LiuSectorSnowflake_Id3` | Liu sector snowflake (ID3) | `adda_config_LiuSectorSnowflake_Id3.py` |
| — (analytic) | oblate ellipsoid (test) | `adda_config_ellipsoid_oblate.py` |

`adda_python_scripts/old_configs/` holds superseded configs (Evans snowflake, Liu shapes
per frequency, `*_RndOrient` configs for the old TRO approach). **Do not edit; use as
reference patterns only.**

## Current Status

- **ARO production runs done (old HPC)** for: `HexPlaAggCrystal_Id20` and `HongPlate_Id9`
  (plate habit). Results lived on the HPC under `…/adda_results/` (not in this repo; see
  note below — currently not accessible).
- **TRO stack ready, runs pending.** `adda_shape_files/HongPlate_Id9_TOR_sizegrid/` is the
  first TRO size grid; the modernized stack is `adda_main_control_rndorient.py` →
  `adda_sub_control_randomorient.py` (9-row liste with temperature) + production config
  `adda_config_HongPlate_Id9_TRO.py` (4608 jobs, `-orient avg` + XY-folded β input,
  validated end-to-end + gate PASS — report §5.2/5.3).
- **Backlog configs READY** (repaired frequency lists, symmetry flags set, smoke-tested;
  HPC submission is the remaining step — runbook report §5.6): HongColumn_Id7,
  IconSnow_Id28, LiuSectorSnowflake_Id3. Job/cost inventory:
  `analysis/phase5_backlog_inventory.txt`.
- Previous HPC ARO result files are **no longer accessible** — regression validation must
  rely on analytic tests (sphere/Mie, oblate ellipsoid) and 1.2-vs-1.4/1.5 cross-runs.
- **Phase 6 (2026-11-06): backscatter Mueller matrix implemented** — exact
  4x4 `mueller_scatgrid` Theta=180/Phi=0 slice extracted per vertex
  (raw k-normalized units, + Phi-ring for mirror fill), cached in vertex
  NetCDFs (`back_mat`/`back_mat_ring`, old stores => cache miss), mirror
  fill-in with the `D=diag(1,1,-1,-1)` sign rule (verified exact on
  computed twins), orientation-averaged via `pol_changer_back`
  (convention: B_lab = R(+phi) B R(+phi) — NOT the ext-branch -phi;
  pinned by gate scan vs the pipeline phase matrix: floor 3.1e-2 vs
  0.96 for the ext analog), overview-mat storage, lambda^2/4pi^2 ->
  m^2/sr conversion, SSDB export `backMat_data`/`backMat_index`,
  database version **beta_0.9.1**. Gate suite `analysis/phase6_backscatter.py`,
  results `analysis/phase6_results.json`, findings report §6.
- **Modernization plan approved** (see `plans/DDA-modernization-plan.md`): code analysis,
  numpy-2 migration, configurable ADDA binary (1.2 / 1.4 / 1.5.0-alpha3), symmetry-based
  optimization, then production backlog. Status: **Phases 0+1+2+3 done, Phase 4 core done**
  (Phase 2: pipeline
  runs natively on numpy 2 / scipy 1.18 / py3.12 / macOS, proven by `analysis/phase2_equiv.py`;
  internal Mueller artifact switched from ARTS-XML to pickle (Phase 7 replaced pickle by
  NetCDF4). Phase 3: config-driven ADDA
  binary + local no-queue mode + TRO repairs; equivalence suite `analysis/phase3_equiv.py`
  PASSES all gates — 1.2/1.4.0/1.5.0-alpha3 identical within solver noise, TRO `-orient avg`
  cross sections bit-identical, auto dpl heuristic identical; **production binary decision:
  ADDA 1.4.0**) — findings in `plans/DDA-modernization-report.md` (read it before touching
  pipeline code; §1.5 breakages, §1.6 dormant bugs, §1.7 backlog reality check, Phase-2/3
  sections list deferred items). Phase 4 + 4b complete (mirror machinery + flags +
  gates; steps 17–19 resolved in report Phase-4b: β/θ grids are SSDB output coords
  not quadrature axes — GL nodes save no solves, half-range reuse provably wrong
  (2.3e-3..7.2e-2 vs 1e-11 noise); ADDA cost table `analysis/phase4b_bench_results.json`
  (error budget is discretization-dominated: eps=3 at noise floor, auto dpl_eff≈18 at
  production x, anchors 12k dipoles→0.4s / 49k→9.5s single-core); vertex loop is
  process-level parallel — HPC recommendation: pool N_cores single-core `adda` solves
  inside the per-(T,f,size) job, keep adaptive refinement sequential).
  `sym_Zobj` parked (not exactly available at model level, §4.6). **Phase 5 done —
  plan complete**: TRO strategy settled (`-orient avg` gives cross sections AND
  orientation-averaged Mueller; production config `adda_config_HongPlate_Id9_TRO`
  with XY-folded β-range avg input, gate vs full-range twin PASS ≤1.35e-5 = Romberg
  d); backlog configs repaired (5-GHz stubs → 35-channel list; inventory
  `analysis/phase5_backlog_inventory.txt`: HC 1524 jobs/141 CPU-h, Icon 1276/1942,
  Liu 1252/257, HongPlate 5235/377, TRO 4608 orient-avg jobs; smoke runs exit 0);
  SSDB export localized + verified end-to-end (mat + full SSD NetCDF tree); golden
  baselines re-anchored to `baselines/adda_1.4.0/`. Remaining is operational
  (HPC submissions per runbook §5.6, optional published-DB cross-check).
  **Phase 6 done** (2026-11-06): exact 4x4 backscatter Mueller matrix (ADDA
  `mueller_scatgrid` Theta=180/Phi=0 slice) extracted per vertex, cached,
  mirror-filled, orientation-averaged and exported to SSDB as
  `backMat_data`/`backMat_index` (database version **beta_0.9.1**); the
  backscatter matrix is stored *outside* the spherical-harmonic route, which
  Brath et al. (2020) flag as unreliable at the exact backscatter direction.
  Plan `plans/backscatter_matrix_plan.md`, report §6, gate suite
  `analysis/phase6_backscatter.py`.
   **Phase 7 done** (2026-11-06): internal Mueller artifact migrated
   pickle → **NetCDF4** (`*_mueller_data.nc`, one file per run; schema:
   report §7.1; writer/reader helpers in `acf`, atomic write,
   `vertex_index` group attr is the sole ordering key). All gates bitwise
   PASS (`analysis/mueller_nc_roundtrip_check.py`,
   `analysis/mueller_nc_gate.py` incl. SSDB export double-reader
   equivalence); all 12 local legacy `.pkl` artifacts migrated +
   bitwise-verified via `analysis/migrate_pkl_to_nc.py`. Report §7.
   **Phase 8 done** (2026-11-07): size-parameter-adaptive ADDA scattering
   grid (`scatgrid_adaptive=True` config switch; budget law
   `ntrunc_cap = ceil(6·(x+4x^{1/3}+2))` for phi + separate theta law
   `ntheta ≤ min(181, max(cap+2, πx/0.1+1))`; even periodic phi ring
   `nphi=2cap+2`, endpoint EXCLUDED; `spha.sph_expansion` initial truncation
   now `(nlons-1)//2`). Per-run `scat_params_adaptive.dat` in storage_path;
   provenance in overview mat + nc root attrs (`size_parameter`,
   `scatgrid_ntrunc_cap`=-1 static, `scatgrid_adaptive`). Gates:
   `analysis/phase8_scatgrid_twin.py` (all PASS margin≥4; margin 3 fails at
   x=5 as expected) + `analysis/phase8_endtoend_twin.py` (22x42 vs 181x361
   full-pipeline twin: d_ext 7.6e-16, d_sph 1.9e-3 — PASS). Per-solve
   speedups ~8-17x at small x, ~2x at x=5, ~1.2x at x=10. Switch still OFF
   in production habit configs (probe configs
   `adda_config_phase8_hp_{adaptive,static}.py` only). **Export-side
   adaptivity done** (same day): SSDB export caps `sph_coeffs_scat` at
   `acf.scatgrid_ntrunc_cap(x, margin=export_sph_margin=10.0)` via
   `acf.sph_array_truncate` (0/None = historic worst-cell padding); gate
   `analysis/phase8_export_twin.py` PASS (bitwise kept content, removed
   tail <= noise budget, **15x smaller** beta files). Report §8.8.
- **Project manual added** (`docs/DDA-manual.md`, 2026-11-06): the maintainer-facing
  single reference — physics theory, pipeline architecture/call graphs, config contract
  walkthrough, local/HPC runbooks, validation gates, gotcha catalogue. Read it alongside
  `plans/DDA-modernization-report.md` (the historical findings log) when operating or
  extending the pipeline.

## The Workflow (`adda_python_scripts/`)

### ARO pipeline (per habit)

1. **`adda_config_<Habit>.py`** — one config file per habit: shape folder, frequency list,
   temperatures, `number_of_betas` (β tilt orientations, currently 5), `incidence_angles`
   (10 values 0–180°), dielectric function (ice: `acf.eps_ice_matzler06(f, T)`, Matzler 2006),
   tolerances (`eps=3` for ADDA, `eps_sphexp=0.005` for spherical-harmonic expansion),
   `size_parameter_limit=10`, HPC settings (`ncores`, Slurm `Partition`).
2. **`adda_main_control.py`** — interactive entry point: lists all `adda_config_*.py`,
   asks for one index, loops over temperatures × frequencies × sizes, skips
   size-parameter violations and finished jobs, and submits one SLURM job per
   (T, f, size) via `sbatch_simple*` → runs
3. **`adda_sub_control_spharmonics.py`** — per job: runs ADDA for each β and incidence
   angle (`acf.adda_call`), stores scattering grids (`-store_scat_grid` /
   `-scat_grid_inp`, `-scat_matr both`), assembles Mueller matrices, expands them in
   spherical harmonics over orientation space (`spherical_harmonics_for_adda.py`,
   using `shtns`; icosphere grid from `icosphere_grid.py`) with iterative refinement to
   `eps_sphexp`.
4. **`adda_process_aro_particles.py`** (batch) / **`adda_process_aro_particles_single.py`**
   (per habit) — post-processing: re-reads harmonic expansions per size/frequency, merges,
   checks relative error, and exports to SSDB NetCDF via
   **`adda2SSDBnetcdf_functions.py`** driven by **`definitions_for_netcdf.py`**
   (SSDB variable/attribute definitions, version `beta_0.9`, groups
   `SingleScatteringData` / `ShapeData` / `CalculationData`).

### TRO pipeline (production-ready since Phase 5)

`adda_main_control_rndorient.py` → `adda_sub_control_randomorient.py` →
`acf.adda_call_randomorient`, using ADDA's built-in `-orient avg` with an
`avg_params.dat` from `adda_aux_input/` (zyz Euler angles, Romberg refinement
`Jmin/Jmax/eps`). Old configs in `old_configs/*_RndOrient.py`. **Phase-3
repairs**: `adda_call_randomorient` now fails loudly (mistral batch path raises
`NotImplementedError`; ADDA exit codes checked; exceptions re-raised instead of
swallowed); `static_clo_temp` no longer unbound for analytic shapes; avg files
verified accepted by ADDA 1.2/1.4/1.5 (report Phase 3). **Phase-5 repairs DONE**:
9-row `liste` carries the temperature (row 8) — `adda_sub_control_randomorient.py`
reads it (legacy 8-row configs still run but warn); new production TRO config
`adda_config_HongPlate_Id9_TRO.py` (+ `_probe` / `_probe_fullr` gate twins) over
`HongPlate_Id9_TOR_sizegrid` with XY-mirror-folded averaging input
(`adda_aux_input/HongPlate_Id9_TOR_sizegrid/avg_params.dat`, beta 0–90°, justified
by the voxel-exact XY closure of that grid); gate vs full-range twin PASS
(≤1.4e-5 = Romberg tolerance; averaged Mueller ≤1e-13) — report §5.3.

`adda_sub_control_gridref.py` + `acf.romberg_*` and
`adda_orientational_avg_spharmonics_for_testing.py` are **deprecated** (header
notes; do not extend — report §1.6-9).

### Support modules

- `adda_control_functions.py` — ADDA CLI wrapper (`adda_call`,
  `adda_call_randomorient`), Romberg integration over α/γ (own quadrature as alternative
  to `-orient avg`), Matzler 2006 ice dielectric function, Slurm job chunking, MATLAB
  `.mat` reader (`loadmat`), NetCDF read/append helpers.
- `icosphere_grid.py` — icosahedral sphere grid generation/refinement for orientation sampling.
- `spherical_harmonics_for_adda.py` — 2-D (θ,φ) harmonic expansion of Mueller matrix
  functions over particle orientation.
- `memory.py`, `posix_spawn.py`, `alldir_params.dat`, `scat_params.dat` — helpers/ADDA input files.
- `sbatch_adda` — legacy Slurm wrapper for `adda_mpi` on Mistral.
- `NetCDF_export_testscript*.py`, `adda_orientational_avg_spharmonics_for_testing.py` —
  test/validation scripts for the export and the harmonic averaging.

## Compute Environment

- **HPC (primary in the past):** Mistral (hostnames `mlogin*`, `mistral*`, `m2*`) and
  Thunder (`thunder`). Configs auto-detect via `socket.gethostname()` and set cluster
  prefixes (`/work/um0878/users/mbrath/`, `/scratch/uni/u237/users/mbrath/`), core counts
  and Slurm partitions. Job submission through `sbatch_simple` wrappers (path hardcoded).
- **Local Mac (this machine):** the scripts as-is will **not** run — hostname `else`
  branches still point at `/scratch/uni/...` and `sbatch_*`. To run small test jobs locally:
  - Build ADDA locally: `cd adda/src && make seq` (or `make mpi` with mpich/openmpi).
    FFTW3 and OpenMP recommended. Never reuse the Linux ELF binaries in `adda_1.2/src/seq|mpi/`.
  - **Proven macOS (arm64) build recipe** (works for both `adda_1.2` and `adda` 1.5.0-alpha3,
    2026-11-03): FFTW3 comes from the DDA conda env; no gfortran on this Mac so build
    without Fortran (only IGT/surface mode need it; the pipeline's default `-int defint`
    does not):
    ```sh
    cd adda_1.2/src && make cleanseq
    make seq "OPTIONS=NO_FORTRAN" \
      FFTW3_INC_PATH=$CONDA_PREFIX/../envs/DDA/include \
      FFTW3_LIB_PATH=$CONDA_PREFIX/../envs/DDA/lib
    # make links without rpath; dyld then cannot find libfftw3.3.dylib. Fix by relinking
    # the object files make produced, with rpath added, e.g.:
    # cd seq && gcc -o adda *.o -L<env>/lib -Wl,-rpath,<env>/lib -lm -lfftw3
    ```
    Stale `.d` files from the old Linux build must be cleaned first (`make cleanseq`),
    otherwise make aborts on `/usr/include/stdc-predef.h` dependencies.
  - A config needs a local branch (new hostname case or `calc_type='noqueue'`-style path
    that calls `adda_sub_control_*.py` directly instead of sbatch).
- **Python:** dedicated mamba env **`DDA`** (`mamba activate DDA`), Python 3.12:
  - conda-forge: `numpy` (2.x), `scipy`, `netcdf4`, `joblib`, `matplotlib`, `fftw`, `pkg-config`
  - pip: `typhon` (not on conda-forge), `shtns` (conda-forge has **no osx-arm64 build**;
    installed as native arm64 wheel from PyPI source — rebuild with
    `pip install --no-build-isolation --no-deps shtns` while `fftw`+`pkg-config` are
    in the env, so it links against the env's fftw/numpy; shtns 3.7.5 still provides all
    API used by `spherical_harmonics_for_adda.py`: `sht_orthonormal`, `SHT_NO_CS_PHASE`,
    `analys`/`synth`, `Zrotate`/`Yrotate`, `SH_to_point`, `gauss_wts`)
  - `adda_main_control.py` uses interactive `input()` to select the config.
  - **numpy 2 migration DONE** (Phase 2, 2026-11-05): all `np.NaN`→`np.nan`,
    `np.complex`→`np.complex128`, `scipy.integrate.trapz`→`trapezoid` (with old-scipy
    fallback), `matlib.repmat`→`np.tile`, `posix_spawn.py` has a platform branch
    (Linux ctypes path unchanged; macOS uses subprocess) — every pipeline module now
    imports and runs locally. `typhon` must stay **0.10.0** (0.8.0 is broken on
    py3.12; `ty.arts.xml` is gone — internal Mueller artifacts went ARTS-XML →
    pickle → **NetCDF4** files `*_mueller_data.nc` (Phase 7, see report), one
    file per (T,f,size) run).

## Conventions & Gotchas

- **Config contract (since Phase 3)**: every active `adda_config_*.py` defines
  `calc_type` (host detection incl. a new **`local`** branch for macOS/darwin —
  repo-relative paths, no queue, direct `sub.check_call` execution),
  `adda_prefix` (ADDA checkout root: `<prefix>/adda` on cluster, the repo itself
  locally), `cluster_name` (SSDB provenance), and the ADDA binary triple
  `adda_binary` / `adda_launcher` / `adda_version` (default 1.2; overridable at
  run time via env `ADDA_BINARY=<exe> ADDA_VERSION=<ver>` for cross-version
  work). Sub-control scripts fall back to the historic 1.2 defaults if a config
  lacks these. The three ADDA trees are now built locally: `adda_1.2/`,
  **`adda_1.4/`** (v1.4.0 tarball from github adda-team/adda, gitignored —
  same NO_FORTRAN+FFTW rpath build recipe), `adda/` (1.5.0-alpha3).
  `-Csca`/`-alldir_inp` were **removed from all ARO configs** (outputs never
  parsed; ARO extinction comes from the forward `ampl_scatgrid` column).
  **Production binary decision (Phase 3): ADDA 1.4.0** — numerically indistinguishable
  from 1.5.0-alpha3 and within solver noise of golden-1.2 (suite `analysis/phase3_equiv.py`,
  `analysis/phase3_results.json`). Caveat: ADDA re-runs of identical commands are *not*
  bit-reproducible at `-eps 3` (own scatter ≤1e-11, but ~8e-4 for x=5/12312 dipoles) →
  regression/golden gates must be noise-based, never bit-level.
- One `adda_config_*.py` per habit; keep the structure/field names of existing configs so
  the control scripts keep working (they read globals via `exec('from ... import *')`
  and `getattr`-style checks).
- Every shape `.adda` file must have a matching `.meta.mat` sibling; metadata drives
  size/frequency bookkeeping (`diameter_vol_eq`, `diameter_max`, `frequency_min`).
- **Shape resolution scheme**: each physical size exists in 5 resolution files with
  `frequency_min ∈ {192, 337, 457, 671, 887} GHz`; `N_dipoles` is fixed per shape file
  (tied to fmin — ADDA rescales geometry, never resamples). The `f < fmin` skip in
  `adda_main_control.py` implements resolution selection (file with smallest
  `fmin >= f` = least-fine discretization still resolved at f). The old backlog
  5-GHz stubs are FIXED (Phase 5): all active configs now carry the production
  channel list (report §1.7 for the historical problem).
- Size parameter x = π·D_veq/λ; default cap `size_parameter_limit = 10`.
- ARO config symmetry flags (`sym_XYobj`, `sym_XZobj`, `sym_YZobj`) must reflect the actual
  habit symmetry. **Phase 4 implemented the full mirror machinery** (write-time fill-in —
  skipped vertices are reconstructed from the mirrored raw ADDA grid of the run vertex,
  transform table + proof in report §4.2) and set the flags from measured voxel closure
  (`analysis/phase4_symmetry.py`): **HongPlate_Id9 + TOR grid: XY+XZ True** (~3.6× fewer
  solves), **HongColumn_Id7: YZ True** (~1.9×), LiuSector/Icon/HexPlaAgg: all False.
  **Closure is resolution-dependent**: a mirror exact at fine resolution may be broken at
  the coarse (192 GHz-class) files of the size grid (e.g. HongPlate YZ 0.986 coarse) —
  flags must use the **coarsest** class; a not-closed mirror fill is a real shape error
  (~1.5e-4 Mueller field error, NOT solver noise). `sym_Zobj` (Cn rotation) still has no
  **Shape-model caveat:** the model is cubes on a cubic lattice — mirrors map
  the lattice to itself (exact closure is what the flags test), but rotations
  other than 90/180° never do, so a voxelized hexagon is never an exactly C6
  *model* (ideal-shape C6 is sub-voxel only; exploiting it costs a boundary-
  layer discretization error). Exact C2z is itself resolution-dependent
  (HongPlate: 1.000 fine, 0.986 coarse) → `sym_Zobj` rotation reduction is
  not exactly available for the production size grid; the XY+XZ fill-in
  already implies exact C2x (mirror composition).
- **ADDA 1.2 → 1.4/1.5 compatibility** (checked against upstream changelog + `param.c` of
  the 1.5.0-alpha3 tree in `adda/`): `-eps <n>` semantics unchanged (residual exponent),
  `-orient`, `-m`, `-lambda`, `-eq_rad`, `-store_scat_grid`, `-scat_grid_inp`, `-scat_matr`,
  `-Csca`, `-alldir_inp` all still exist. **But `-orient avg` cannot be combined with
  `-scat_grid`, `-alldir` or `-Csca`** (1.4/1.5 reject it; 1.2 `param.c` too).
  With `-orient avg` ADDA Romberg-integrates the cross sections AND the
  orientation-averaged **Mueller** matrix (writes `mueller` on the θ-grid +
  `CrossSec`; source: 1.2 `calculator.c`/`Romberg.c`, verified end-to-end with
  1.4.0 in Phase 5) — only the amplitude matrix is never integrated. TRO products
  therefore need NO own orientation sampling (strategy decision report §5.2);
  own icosphere sampling remains the ARO machinery only.
- ADDA output parsing (`ampl_scatgrid`, `mueller_scatgrid` in `acf.adda_call`) is
  **positional** (column indices hardcoded) → re-verify column layouts when switching
  ADDA binary versions.
- ADDA binary path is hardcoded in `adda_sub_control_spharmonics.py` (line ~107:
  `…/adda/adda_1.2/src/mpi/adda_mpi`) — to be replaced by a config-driven `adda_binary`.
- In `avg_params.dat` (TRO `-orient avg`): zyz Euler convention; exploit habit symmetries to
  shrink ranges (e.g. `beta max=90` for an xy-symmetry plane; axysymmetrical → `gamma max=0`).
- `force_new=True` in a config re-runs everything; default `False` skips completed cases —
  idempotency is tracked by looking for existing output (mat/NetCDF) files (the
  completed-case glob accepts `*_mueller_data.nc` (current), plus legacy `.pkl`/`.xml`).
- **Mueller artifact (Phase 7)**: the per-run spherical-harmonic Mueller store is a
  single **NetCDF4** file `<shape>_mueller_data.nc` (replaces Phase-2 `.pkl`, which
  replaced ARTS-XML `.xml`). Schema: root dims `stokes=4`/`stokes2=16` + `float_vlen`
  (float32) + `int32_vlen` (int32), root provenance attrs mirroring the `_overview_data.mat`; one **group per
  computed icosphere vertex** (name = the temp-cache `vertex_X…` id), carrying
  `mueller_sph` (16 vlen arrays, complex64 via `.view(float32)` — lossless),
  `sph_l`/`sph_m` (16 int32 vlen arrays each: the l- and m-index of every stored
  coefficient, positionally aligned with `mueller_sph` — triangular m≥0 shtns
  order, generated by `acf.sph_lm_indices`, cross-verified against shtns
  `l`/`m`/`idx()`; `acf.read_sph_indices` also accepts the short-lived first
  schema (single interleaved `sph_indices` variable); stores predating index
  storage → `None`), `relerror`,
  `ext_mat`, optional `back_mat`; group attr **`vertex_index` (int32)** = row into the
  run's `vertices` array is the ONLY ordering key (group iteration order is never
  trusted); uncomputed vertices have no group (loader yields `None`). Writers/readers:
  `acf.write_mueller_array_to_netcdf` / `acf.load_mueller_array_from_netcdf`; sub_control
  writes atomically (`.tmp` + `os.replace`); config flag `mueller_artifact_pickle=True`
  writes a legacy `.pkl` twin (gates only, never production). `rawdata2average` reads
  `.nc` first, falls back to `.pkl` with a deprecation warning; migrate old artifacts
  with `analysis/migrate_pkl_to_nc.py` (bitwise-verified, keeps the `.pkl`; re-running
  it also UPGRADES existing `.nc` stores that lack `sph_l`/`sph_m` or still carry the
  legacy interleaved `sph_indices` variable). Per-vertex
  `_temp_data_vertex_*.nc` intra-run caches are unchanged. Gates:
  `analysis/mueller_nc_roundtrip_check.py` + `analysis/mueller_nc_gate.py`.
 - **Adaptive scattering grid (Phase 8)**: enabled per config via
   `scatgrid_adaptive=True` (+ `scatgrid_margin` 6.0 / `scatgrid_theta_kappa`
   0.1 / `scatgrid_theta_mode` 'uniform'). Grid-file conventions that MUST
   hold (see `acf.write_scat_grid_params` header): phi is an EVEN periodic
   ring of `nphi=2*cap+2` points with endpoint EXCLUDED (max=360-360/nphi) —
   duplicate phi=0/360 injects FFT aliasing error ∝ Δφ² (3.2e-2 twin failure
   at x=5 before the fix), and odd rings break `mirror_phi_index_map`
   (180° must be a ring node); theta keeps exact 0 and 180 rows (forward
   extinction parse = row 0; Phase-6 backscatter ring); theta never exceeds
   the production 181. `spha.sph_expansion` initial truncation is
   `(nlons-1)//2` (shtns requires nphi > 2*mmax on even rings; same value
   180 for the historic 361-point static grid). Gauss-node theta mode exists
   in `acf` for experiments only — it fails the gates (no near-pole samples).
 - **Twin-gate vintage rule (Phase 8)**: end-to-end comparisons must use runs
   of the SAME pipeline vintage; old frozen stores (e.g. `phase6_hp_sym`)
   differ from fresh reruns by ~4.5e-8 ext at 12/642 vertices for reasons
   unrelated to the tested change. Also: SH coefficient arrays of different
   ntrunc must be compared matched by (l,m) via `acf.sph_lm_indices`, never
   by raw prefix.
 - **Export truncation (Phase 8)**: `adda_process_aro_particles_single.py`
   caps the SSDB `sph_coeffs_scat` dimension at
   `acf.scatgrid_ntrunc_cap(x, margin=export_sph_margin)` (default 10.0,
   set 0/None for the historic worst-cell padding; rationale + gate: report
   §8.8). Unit trap: in overview mats / `rawdata2average` output_data the
   diameter (`diameter_volume_equ`) is in **metres** while `wavelength` is
   in **micrometres** — convert before computing x. Gate metrics must use
   the cell's physical intensity scale + measured store noise plateau, never
   per-element relative norms (Rayleigh-limit channels are pure noise).
- Results/intermediate files go to `<prefix_data>/adda_results/` on the cluster — outside
  this repo. This repo only holds inputs, source and scripts.
- `SSD_scripts_deprecated/` and `old_configs/` are archives; never import from them in new code.
- No version control here (not a git repo). Consider `git init` before large changes.

## Key References

- **Brath et al. 2020, *AMT* 13, 2309–2333, doi:10.5194/amt-13-2309-2020** — **the main
  methodology article for the ARO approach in this project** (TRO/ARO definitions, zyz Euler
  angles, α/γ-averaging at fixed tilt β, the (β, θ_inc) ARO output coordinates, ADDA +
  self-developed orientation averaging; ARO hex-plate/aggregate database: zenodo 3463003).
- **Eriksson et al. 2018, *ESSD* 10, 1301–1326, doi:10.5194/essd-10-1301-2018** — the published
  ARTS Microwave SSPDB (TRO) paper: source of the habit-ID numbering (`*_Id<n>` folder names),
  shape-file lineage, the production frequency-channel selection, T = 190/230/270 K, and the
  SSDB NetCDF format our export targets.
- Yurkin & Hoekstra 2011, *JQSRT* 112, 2234–2247 (ADDA paper). ADDA manual: `adda/doc/manual.pdf`, `adda_1.2/doc/manual.pdf`.
- Matzler 2006 dielectric function of ice (implemented in `acf.eps_ice_matzler06`).
- ARTS Microwave SSP Database: https://doi.org/10.5281/zenodo.1175572
- Liu / Hong-MoCOLAG / Icon habit definitions: particle IDs referenced in shape folder names
  (ID numbering and shapes from Eriksson et al. 2018; see `arts-sspdb-interface` documentation
  for habit naming conventions).
