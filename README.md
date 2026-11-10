# DDA — single-scattering properties of randomly oriented hydrometeors

Calculation of **single scattering properties** (extinction / absorption cross sections, Mueller
matrices, exact 4×4 backscatter matrices) for **azimuthally random oriented (ARO)** and **totally
random oriented (TRO)** particulate habits — hexagonal plates/columns, aggregates, Liu /
Hong-MoCOLAG and Icon snowflakes, liquid droplets — with the **Discrete Dipole Approximation
(ADDA)**, for the **ARTS Microwave Single Scattering Properties Database (SSDB)** and
radiative-transfer use (RTTOV).

- ARO methodology: Brath et al. 2020, *AMT* 13, 2309–2333, doi:10.5194/amt-13-2309-2020.
- Habit IDs, database lineage and the NetCDF format: Eriksson et al. 2018, *ESSD* 10, 1301–1326.
- Export writer emits SSDB **`beta_0.9.1`** NetCDF (Phase-6 additive backscatter variables).
- Published databases to cross-check against: [zenodo 1175572](https://doi.org/10.5281/zenodo.1175572)
  (TRO, Eriksson 2018) and [zenodo 3463003](https://doi.org/10.5281/zenodo.3463003)
  (ARO hex plates/aggregates, Brath 2020).

**Manual:** [`docs/DDA-manual.md`](docs/DDA-manual.md) (also `.html`/`.pdf`) — theory, architecture,
runbooks, gotcha catalogue. [`plans/DDA-modernization-report.md`](plans/DDA-modernization-report.md)
is the running findings log (§§0–14); [`agents.md`](agents.md) is the machine-readable project brief.

## Where the project stands (2026-11-10)

- **Modernization plan complete — Phases 0–9A.** numpy-2 / py3.12 / macOS port; config-driven ADDA
  binary + local no-queue mode; mirror-symmetry fill-in (XY/XZ/YZ per habit, from measured voxel
  closure); TRO strategy settled on ADDA's `-orient avg`; SSDB export verified end-to-end; exact
  backscatter Mueller matrices stored outside the SH route; internal Mueller artifact
  pickle → NetCDF4; size-parameter-adaptive scattering grids plus export-side SH truncation caps.
  Phase 9A (analytic diffraction-peak model) is an **analysis-only scan → conditional GO**
  (gated model+residual hybrid, deliberately not wired into the pipeline yet).
- **Production solver: ADDA 1.4.0** (Phase-3 decision; 1.2 and 1.5.0-alpha3 kept for cross-checks).
- **Done:** ARO production runs for `HexPlaAggCrystal_Id20` and `HongPlate_Id9`. The old HPC result
  store is inaccessible, so regression validation rests on the frozen `baselines/` (golden ADDA
  outputs) and analytic Mie / ellipsoid twins.
- **Ready, awaiting HPC submission:** TRO production (`adda_config_HongPlate_Id9_TRO`, 4608
  `-orient avg` jobs) and the ARO backlog (`HongColumn_Id7`, `IconSnow_Id28`,
  `LiuSectorSnowflake_Id3`) — ≈13.9k jobs, costed in `analysis/phase5_backlog_inventory.txt`
  (report §5.6).
- **Repo housekeeping (§§10–14):** 46 commits, 307 tracked files, `.git` 54 MB, **no submodules** —
  the pipeline is tracked here with its full history; geometry, results and third-party code are
  independent checkouts outside git.

## ⚠️ A fresh clone runs nothing — provision it first

| what | where it comes from | notes |
|---|---|---|
| **Particle geometry** (1.8 GB: 970 `.adda` + 970 `.meta.mat`) | not in git (purged, report §10) — copy it in from the authoritative store | manual §1.2 "Shape-data provisioning"; every habit config, gate and baseline globs this folder |
| **ADDA solver trees** | not in git (§11): `adda_1.4/` + `adda_1.2/` upstream tarballs, `adda/` = [`m-brath/adda`](https://github.com/m-brath/adda) (1.5.0-alpha3; upstream has no 1.5 tag), `adda-gui/` = [`m-brath/adda-gui`](https://github.com/m-brath/adda-gui) | manual §3.5: fetch, then build (`NO_FORTRAN` + FFTW from the env, relink with rpath on macOS) |
| *(optional)* ARTS SSPDB tooling | independent repo [`m-brath/arts-sspdb-interface`](https://github.com/m-brath/arts-sspdb-interface) (upstream `atmtools/arts-sspdb-interface`) | browsing/importing the *published* database; never invoked by the pipeline (§14) |

```sh
git clone https://github.com/m-brath/DDA.git
# then provision:  adda_shape_files/   adda_1.4/  adda_1.2/  adda/        (manual §1.2 / §3.5)
```

## Layout

Tracked here (project-owned):

| path | contents |
|---|---|
| `adda_python_scripts/` | **the pipeline** (87 files, tracked with its full history since 2026-11-10 — §13): `adda_config_*.py`, main/sub control drivers, `adda_control_functions.py` (ADDA CLI, Matzler 2006 ice dielectric, mirror transforms), `spherical_harmonics_for_adda.py` (shtns), `icosphere_grid.py`, SSDB export |
| `baselines/` | frozen golden ADDA outputs + `make_baseline.py` / `check_baselines.py` — the reference for every migration; do not regenerate casually |
| `analysis/` | phase analyses, gate/twin suites, benchmark results (`phase*_*.py`, `.json`, `.txt`, `.log`) |
| `plans/` | modernization plan + findings report — read before touching pipeline code |
| `docs/` | project manual (`.md` + generated `.html`/`.pdf`) + figures + build scripts |
| `adda_aux_input/` | ADDA `avg_params.dat` files (TRO `-orient avg` Euler-angle integration control) |
| `SSD_scripts_deprecated/` | deprecated SSDB-interface prototypes — archive, never imported |
| `agents.md`, `README.md`, `environment.yml` | project brief, this file, pinned conda env (py3.12, shtns 3.7.5, typhon 0.10.0) |

Deliberately **outside** git (independent checkouts, gitignored):

| path | what it is |
|---|---|
| `adda_1.4/` · `adda_1.2/` | ADDA 1.4.0 (**production**) / 1.2 upstream source + local builds |
| `adda/` · `adda-gui/` | ADDA 1.5.0-alpha3 fork (validated state `8f550a7`) · official ADDA GUI |
| `arts-sspdb-interface/` | ARTS SSPDB tooling fork (state at untracking `e7b71023`) |
| `adda_shape_files/` | 1.8 GB particle geometry — external **input** data |
| `adda_results/` | run output (also on the cluster under `<prefix_data>/adda_results/`) |

## Getting a run going

```sh
mamba env create -f environment.yml && mamba activate DDA     # py3.12 + shtns + typhon pinned
# build each ADDA tree once (manual §3.5):  make seq "OPTIONS=NO_FORTRAN" + FFTW rpath relink
python analysis/inventory.py            # habit × size × frequency × temperature status sheet
```

`adda_main_control.py` (ARO) / `adda_main_control_rndorient.py` (TRO) submit one job per
(T, frequency, size); on a local Mac use the `local` / no-queue config branch. Cross-version
experiments without editing configs: `ADDA_BINARY=../adda_1.4/src/seq/adda ADDA_VERSION=1.4.0 …`.

## Verify a checkout

```sh
python baselines/check_baselines.py        # golden ADDA outputs vs independent Mie theory
python analysis/phase2_equiv.py            # pipeline equivalence before/after the modernization
python analysis/phase3_equiv.py            # ADDA 1.2 / 1.4.0 / 1.5.0-alpha3 cross-version
python analysis/phase8_endtoend_twin.py    # adaptive-grid vs static full-pipeline twin
python analysis/mueller_nc_gate.py         # NetCDF4 Mueller artifact round-trip + SSDB export
```

All gates are **noise-based, never bit-level**: ADDA re-runs of an identical command are not
bit-reproducible at `-eps 3` (≈8e-4 at x = 5). Manual §4.2 lists what each suite proves.

## Git rules of this repo

- **Data and third-party code stay out of git** — `adda_shape_files/`, `adda_results/`, and all four
  ADDA trees plus the ARTS tooling (§10, §11, §14). Never `git add -f` them.
- **No submodules, no gitlinks, no `.gitmodules`** — that is the end state. Project code is tracked
  directly (`adda_python_scripts/`, `SSD_scripts_deprecated/`). If a gitlink is ever introduced
  anyway it must be mapped in `.gitmodules`, and its commits pushed to their own remote before
  pinning them here; otherwise `git submodule` / `git clone --recursive` abort and GitHub 404s it.
- **Never rewrite history on a non-bare checkout while the path being purged is still tracked** —
  `git filter-repo` ends with `git reset --hard`, which deletes the working-tree copy (gitignore
  does not protect index entries). Untrack first, or rewrite in a bare clone, and always take a
  `git bundle create … --all` backup. Report §10.4. Pre-restructure archives live beside the repo:
  `../DDA-prepurge-2026-11-10.bundle`, `../DDA-prepurge-addda-2026-11-10.bundle`,
  `../adda_python_scripts-history-2026-11-10.bundle`,
  `../SSD_scripts_deprecated-history-2026-11-10.bundle`.
