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

This repository holds the **pipeline, configs, plans, docs and the frozen regression
baselines**. Everything that is data or third-party code stays out of git and must be brought
in by hand:

| what | where it comes from | notes |
|---|---|---|
| **Particle geometry** | not in git (1.8 GB, 970 `.adda` + 970 `.meta.mat`) | manual §1.2 "Shape-data provisioning"; purged from history, report §10 |
| **ADDA solver trees** | not in git: `adda_1.4` (production) + `adda_1.2` upstream tarballs, `adda/` = `m-brath/adda` | manual §3.5: fetch, then build (`NO_FORTRAN` + FFTW rpath on macOS) |
| *(optional)* ARTS SSPDB reference tooling | independent repo [`m-brath/arts-sspdb-interface`](https://github.com/m-brath/arts-sspdb-interface) (upstream `atmtools/…`) | browsing/import tooling for the *published* database; not needed by the pipeline — clone it beside the repo (§14) |

```sh
git clone https://github.com/m-brath/DDA.git        # the pipeline itself is already inside
# then: provision adda_shape_files/  and  adda_1.4/ + adda_1.2/ + adda/  (manual §1.2 / §3.5)
git clone https://github.com/m-brath/arts-sspdb-interface.git    # optional, only for the published DB
```

## Layout

| path | contents |
|---|---|
| `adda_python_scripts/` | **the pipeline** (tracked in this repo, with its full history since 2026-11-10 — report §13): configs, main/sub control drivers, ADDA CLI wrapper, spherical-harmonic expansion, SSDB export |
| `arts-sspdb-interface/` | upstream ARTS SSPDB browsing/import tooling (own fork, read-only reference) — **not tracked**: independent repo, no submodule (report §14) |
| `adda_aux_input/` | ADDA `avg_params.dat` files (TRO `-orient avg` Euler-angle integration control) |
| `baselines/` | frozen golden ADDA outputs + regression tooling (the reference for all migrations) |
| `analysis/` | phase analyses, gate/twin suites, benchmark results |
| `plans/` | modernization plan + findings report (read before touching pipeline code) |
| `docs/` | project manual (`.md` + generated `.html`/`.pdf`) + figures + build scripts |
| `SSD_scripts_deprecated/` | deprecated SSDB-interface prototypes — archive, do not extend |

## Git rules of this repo

- **External data stays out of git** (`adda_shape_files/`, `adda_results/`) and **third-party ADDA
  trees stay out of git** (report §10, §11). Never `git add -f` them.
- **No submodules, no gitlinks** (report §13/§14): project code is tracked directly
  (`adda_python_scripts/` pipeline, `SSD_scripts_deprecated/` archive), everything third-party is an
  independent gitignored checkout. If a gitlink is ever introduced anyway, it **must** be mapped in
  `.gitmodules` — an unmapped one breaks `git submodule`/`git clone --recursive` and 404s on GitHub —
  and its commits must be pushed to their own remote before pinning them here.
- **Never rewrite history on a non-bare checkout while the path being purged is still tracked** —
  `git filter-repo` ends with `git reset --hard` and deletes the working-tree copy (gitignore does
  not protect index entries). Untrack first, or rewrite in a bare clone; always `git bundle create
  … --all` beforehand. Report §10.4.
