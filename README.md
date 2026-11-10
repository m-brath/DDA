# DDA — single-scattering properties of randomly oriented hydrometeors

Calculation of **single scattering properties** (extinction / absorption cross sections,
Mueller matrices, backscatter matrices) for **azimuthally random oriented (ARO)** and **totally
random oriented (TRO)** particulate habits — hexagonal plates/columns, aggregates, Liu / Hong-MoCOLAG
and Icon snowflakes, liquid droplets — with the **Discrete Dipole Approximation (ADDA)**, for the
**ARTS Microwave Single Scattering Properties Database (SSDB)** and radiative-transfer use (RTTOV).

Methodology: Brath et al. 2020, *AMT* 13, 2309–2333, doi:10.5194/amt-13-2309-2020.
Habit IDs and the database format: Eriksson et al. 2018, *ESSD* 10, 1301–1326.

**Read [`docs/DDA-manual.md`](docs/DDA-manual.md)** — the maintainer manual (theory, architecture,
runbooks, gotcha catalogue). The running findings log per phase is
[`plans/DDA-modernization-report.md`](plans/DDA-modernization-report.md); `agents.md` is the
agent/machine-readable project brief.

---

## ⚠️ A fresh clone runs nothing — provision it first

This repository deliberately holds **only** orchestration, configs, plans, docs and the frozen
regression baselines. Three things must be brought in by hand:

| what | where it comes from | notes |
|---|---|---|
| **The pipeline code** | submodule `adda_python_scripts/` → [`m-brath/adda_python_scripts`](https://github.com/m-brath/adda_python_scripts) | clone with `--recursive`, or `git submodule update --init` |
| **Particle geometry** | not in git (1.8 GB, 970 `.adda` + 970 `.meta.mat`) | manual §1.2 "Shape-data provisioning"; purged from history, report §10 |
| **ADDA solver trees** | not in git: `adda_1.4` (production) + `adda_1.2` upstream tarballs, `adda/` = `m-brath/adda` | manual §3.5: fetch, then build (`NO_FORTRAN` + FFTW rpath on macOS) |

```sh
git clone --recursive https://github.com/m-brath/DDA.git
# then: provision adda_shape_files/  and  adda_1.4/ + adda_1.2/ + adda/  (see manual §1.2 / §3.5)
```

## Layout

| path | contents |
|---|---|
| `adda_python_scripts/` | **submodule** — the pipeline: configs, main/sub control drivers, ADDA CLI wrapper, spherical-harmonic expansion, SSDB export |
| `arts-sspdb-interface/` | **submodule** — upstream ARTS SSPDB browsing/import tooling (own fork, read-only reference) |
| `adda_aux_input/` | ADDA `avg_params.dat` files (TRO `-orient avg` Euler-angle integration control) |
| `baselines/` | frozen golden ADDA outputs + regression tooling (the reference for all migrations) |
| `analysis/` | phase analyses, gate/twin suites, benchmark results |
| `plans/` | modernization plan + findings report (read before touching pipeline code) |
| `docs/` | project manual (`.md` + generated `.html`/`.pdf`) + figures + build scripts |
| `SSD_scripts_deprecated/` | deprecated SSDB-interface prototypes — archive, do not extend |

## Git rules of this repo

- **External data stays out of git** (`adda_shape_files/`, `adda_results/`) and **third-party ADDA
  trees stay out of git** (report §10, §11). Never `git add -f` them.
- **Submodules:** push the submodule's commits to its own remote *before* committing a new pointer
  here — otherwise GitHub renders a 404 for the folder. Every gitlink must be mapped in
  `.gitmodules`, or `git submodule`/`git clone --recursive` aborts.
- **Never rewrite history on a non-bare checkout while the path being purged is still tracked** —
  `git filter-repo` ends with `git reset --hard` and deletes the working-tree copy (gitignore does
  not protect index entries). Untrack first, or rewrite in a bare clone; always `git bundle create
  … --all` beforehand. Report §10.4.
