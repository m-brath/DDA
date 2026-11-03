# Baselines — frozen regression reference

Generated 2026-11-03 with **ADDA v1.2** (sequential, locally rebuilt on macOS
arm64) via `make_baseline.py`. These are the golden files for the modernization
plan (see `plans/DDA-modernization-plan.md`, Phase 0/3): every stack change
(numpy-2 migration, ADDA 1.4/1.5 switch, symmetry optimizations) must reproduce
these outputs within stated tolerances.

## How to (re)generate

```sh
mamba activate DDA   # numpy 2.x env
python baselines/make_baseline.py --adda <adda-seq-binary> --out baselines/<tag>
python baselines/check_baselines.py baselines/<tag>   # analytic sanity checks
```

Six cases, mirroring the exact CLI pattern of `acf.adda_call` /
`adda_sub_control_spharmonics.py` (`-orient 0 0 0`, `-prop 1 0 0` i.e.
incidence 90°, `-eps 3`, Matzler-2006 ice `m = sqrt(eps(f,T))`):

| Case | Shape | Notes |
|---|---|---|
| `sphere_ice_x0.31_f300GHz_T230` | `-shape sphere`, eq_rad 50 µm | independent Mie check (see below) |
| `ellipsoid_oblate_x{0.1,1,5}_f183GHz_T253` | `-shape ellipsoid 1.0 0.5` | analytic habit config, 3 size parameters |
| `hongplate_id9_dveq10_f{192,337}GHz_T230` | `-shape read` real habit, Dveq 10 µm | smallest Hong plate size grid entry |

The scattering-grid output uses the coarsened `scat_params_baseline.dat`
(10° steps) instead of the 1° production grid, to keep files small (2 MB total).
The production `alldir_params.dat` (Csca Romberg integration) is used unchanged,
so `CrossSec-X/Y` are directly comparable with production values.

## Verification results (Phase 0)

- **Cross-version:** `adda` v1.2 vs v1.5.0-alpha3 binaries produce identical
  `Qext` for the sphere case to 12 significant digits.
- **Cross-platform:** 1.5.0-alpha3 built on macOS arm64 (FFTW3 from the DDA
  conda env) reproduces the 1.2 numbers exactly.
- **vs. independent Mie theory** (`check_baselines.py`, scipy Riccati-Bessel;
  cross-validated against `miepython` 3.3.1 — agree to all digits):
  ADDA deviates 1.7–2.2% at the automatic grid of the tiny sphere
  (2176 dipoles, 160 dipoles/λ). This is confirmed DDA discretization error:
  forcing `-dpl 320/800` converges Qext 6.914e-3 → 6.852e-3 → 6.819e-3 toward
  the Mie value 6.793e-3.
- Convention note: this Mie helper uses Im(m)<0 for absorption (conjugate of
  ADDA's `m`); see docstring.

## Comparing a newer binary against these baselines

Diff every file in a case dir except `adda_log`/`stdout.log` (build/timing
noise). Expect exact/near-exact matches for identical ADDA semantics; documented
semantics changes need explicit tolerance decisions in Phase 3.
