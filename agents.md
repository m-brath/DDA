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
  incidence angles), then expanding the result in **spherical harmonics**.
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

- **ARO done** for: `HexPlaAggCrystal_Id20` and `HongPlate_Id9` (plate habit).
  Results lived on the HPC under `…/adda_results/` (not in this repo; see note below —
  currently not accessible).
- **TRO not yet done.** `adda_shape_files/HongPlate_Id9_TOR_sizegrid/` was prepared as
  the first TRO size grid. Old TRO machinery exists (`adda_main_control_rndorient.py` +
  `adda_sub_control_randomorient.py` + `acf.adda_call_randomorient` + `*_RndOrient`
  configs using ADDA's `-orient avg`).
- Next habits queued (configs + shape files ready, not yet processed): HongColumn_Id7,
  IconSnow_Id28, LiuSectorSnowflake_Id3.
- Previous HPC ARO result files are **no longer accessible** — regression validation must
  rely on analytic tests (sphere/Mie, oblate ellipsoid) and 1.2-vs-1.4/1.5 cross-runs.
- **Modernization plan approved** (see `plans/DDA-modernization-plan.md`): code analysis,
  numpy-2 migration, configurable ADDA binary (1.2 / 1.4 / 1.5.0-alpha3), symmetry-based
  optimization, then production backlog. Status: **Phases 0+1+2+3 done, Phase 4 core done**
  (Phase 2: pipeline
  runs natively on numpy 2 / scipy 1.18 / py3.12 / macOS, proven by `analysis/phase2_equiv.py`;
  internal Mueller artifact switched from ARTS-XML to **pickle**. Phase 3: config-driven ADDA
  binary + local no-queue mode + TRO repairs; equivalence suite `analysis/phase3_equiv.py`
  PASSES all gates — 1.2/1.4.0/1.5.0-alpha3 identical within solver noise, TRO `-orient avg`
  cross sections bit-identical, auto dpl heuristic identical; **production binary decision:
  ADDA 1.4.0**) — findings in `plans/DDA-modernization-report.md` (read it before touching
  pipeline code; §1.5 breakages, §1.6 dormant bugs, §1.7 backlog reality check, Phase-2/3
  sections list deferred items). Phase 4 core complete (mirror machinery + flags,
  gates passed); plan steps 17–19 (quadrature, ADDA benchmark table, job structure)
  deferred; Phase 5 pending.

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

### TRO pipeline (old, reference)

`adda_main_control_rndorient.py` → `adda_sub_control_randomorient.py` →
`acf.adda_call_randomorient`, using ADDA's built-in `-orient avg` with an
`avg_params.dat` from `adda_aux_input/` (zyz Euler angles, Romberg refinement
`Jmin/Jmax/eps`). Old configs in `old_configs/*_RndOrient.py`. **Phase-3
repairs**: `adda_call_randomorient` now fails loudly (mistral batch path raises
`NotImplementedError`; ADDA exit codes checked; exceptions re-raised instead of
swallowed); `static_clo_temp` no longer unbound for analytic shapes; avg files
verified accepted by ADDA 1.2/1.4/1.5 (report Phase 3). **Open Phase-5 TODO**:
`liste` carries no temperature row → output folders encode the flat index as
`T####K` (FIXME in `adda_sub_control_randomorient.py`) — rebuilt TRO configs
must carry the temperature.

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
    py3.12; `ty.arts.xml` is gone — internal Mueller artifacts are **pickle** files
    `*_mueller_data.pkl` now, not ARTS XML).

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
  `adda_main_control.py` implements resolution selection. Consequence: backlog configs
  with frequencies < 192 GHz (the `=np.array([5,])*1e9` stubs) compute **nothing** —
  see report §1.7.
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
  pipeline implementation — HongPlate has true C6 (potential further ~6×), parked.
- **ADDA 1.2 → 1.4/1.5 compatibility** (checked against upstream changelog + `param.c` of
  the 1.5.0-alpha3 tree in `adda/`): `-eps <n>` semantics unchanged (residual exponent),
  `-orient`, `-m`, `-lambda`, `-eq_rad`, `-store_scat_grid`, `-scat_grid_inp`, `-scat_matr`,
  `-Csca`, `-alldir_inp` all still exist. **But `-orient avg` cannot be combined with
  `-scat_grid`, `-alldir` or `-Csca`** (1.4/1.5 reject it) → built-in averaging only yields
  cross sections, never Mueller grids; TRO Mueller data must come from own orientation
  sampling + spherical-harmonic expansion.
- ADDA output parsing (`ampl_scatgrid`, `mueller_scatgrid` in `acf.adda_call`) is
  **positional** (column indices hardcoded) → re-verify column layouts when switching
  ADDA binary versions.
- ADDA binary path is hardcoded in `adda_sub_control_spharmonics.py` (line ~107:
  `…/adda/adda_1.2/src/mpi/adda_mpi`) — to be replaced by a config-driven `adda_binary`.
- In `avg_params.dat` (TRO `-orient avg`): zyz Euler convention; exploit habit symmetries to
  shrink ranges (e.g. `beta max=90` for an xy-symmetry plane; axysymmetrical → `gamma max=0`).
- `force_new=True` in a config re-runs everything; default `False` skips completed cases —
  idempotency is tracked by looking for existing output (mat/NetCDF) files.
- Results/intermediate files go to `<prefix_data>/adda_results/` on the cluster — outside
  this repo. This repo only holds inputs, source and scripts.
- `SSD_scripts_deprecated/` and `old_configs/` are archives; never import from them in new code.
- No version control here (not a git repo). Consider `git init` before large changes.

## Key References

- Yurkin & Hoekstra 2011, *JQSRT* 112, 2234–2247 (ADDA paper). ADDA manual: `adda/doc/manual.pdf`, `adda_1.2/doc/manual.pdf`.
- Matzler 2006 dielectric function of ice (implemented in `acf.eps_ice_matzler06`).
- ARTS Microwave SSP Database: https://doi.org/10.5281/zenodo.1175572
- Liu / Hong-MoCOLAG / Icon habit definitions: particle IDs referenced in shape folder names (see `arts-sspdb-interface` documentation for habit naming conventions).
