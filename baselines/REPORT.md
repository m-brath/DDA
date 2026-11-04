# Baseline Results Report — ADDA v1.2 frozen regression set

*Generated 2026-11-04 · regenerable via `python baselines/plot_baseline_figs.py`
(figures → `baselines/report/`) · source data: `baselines/adda_1.2/`*

## 1. What these baselines certify

Six frozen ADDA v1.2 runs (sequential binary, rebuilt on macOS arm64) that mirror
the exact CLI of the production ARO pipeline (`acf.adda_call`: `-orient 0 0 0`,
`-prop 1 0 0`, `-eps 3`, Matzler-2006 ice). Every stack change planned in the
modernization (numpy-2 migration, ADDA 1.4/1.5 switch, symmetry optimizations)
must reproduce these files; only documented semantics changes may shift them.

## 2. Case summary and headline numbers

All cross sections from `CrossSec-X/Y`; ω = Csca/Cext with Csca from the
production `-alldir_inp` Romberg integration.

| Case | x | N dipoles | dpl (λ) | Qext (X / Y) | ω (X / Y) |
|---|---|---|---|---|---|
| sphere, 300 GHz, 230 K | 0.314 | 2 176 | 160.7 | 6.914e-3 / 6.914e-3 | 0.684 / 0.684 |
| oblate spheroid c/a=0.5, 183 GHz, 253 K | 0.1 | 1 104 | 402.8 | 3.824e-4 / 7.533e-4 | 0.0814 / 0.0820 |
| oblate spheroid c/a=0.5, 183 GHz, 253 K | 1 | 1 104 | 40.3 | 2.630e-1 / 6.612e-1 | 0.957 / **1.003** |
| oblate spheroid c/a=0.5, 183 GHz, 253 K | 5 | 12 312 | **18.0** | 1.574 / 2.217 | **1.569 / 2.091** |
| HongPlate_ID9, Dveq=10 µm, 192 GHz, 230 K | 0.020 | 1 015 | 1947 | 6.516e-5 / 1.015e-4 | 9.19e-4 / 9.35e-4 |
| HongPlate_ID9, Dveq=10 µm, 337 GHz, 230 K | 0.035 | 1 015 | 1109 | 2.052e-4 / 3.195e-4 | 2.77e-3 / 2.82e-3 |

Consistency checks that pass: sphere Qext X ≡ Y to all printed digits
(rotational symmetry of the sphere); all habit cases show the expected
X-vs-Y polarization splitting; the two Hong-plate runs differ only via m(f)
and the Romberg grid, with Qext scaling ≈ ×3.15 from 192→337 GHz
(Rayleigh: Cext ∝ Im(m)·f, ratio matches).

## 3. Validation results

**Sphere vs. independent Mie theory** (Riccati-Bessel in `check_baselines.py`,
cross-checked against `miepython` 3.3.1 to all digits):

| quantity | ADDA (auto grid) | Mie | deviation |
|---|---|---|---|
| Qext | 6.9144e-3 | 6.7929e-3 | +1.79 % |
| Qsca | 4.7310e-3 | 4.6539e-3 | +1.66 % |
| Qabs | 2.1865e-3 | 2.1390e-3 | +2.22 % |

All within the 3 % acceptance band, and the deviation is *confirmed DDA
discretization error*: forcing `-dpl 320/800` converges Qext
6.914e-3 → 6.852e-3 → 6.819e-3 toward the Mie value (fig. 2).

**Cross-version:** ADDA 1.2 vs 1.5.0-alpha3 — identical Qext for the sphere to
12 significant digits.
**Cross-platform:** 1.5.0-alpha3 rebuilt on macOS arm64 reproduces the Linux
cluster numbers exactly.

## 4. Findings from the new (2026-11-04) analysis

1. **x = 5 spheroid: ω > 1 is unphysical and was reproduced/attributed.**
   Csca (alldir Romberg) exceeds Cext (optical theorem) by 68 %.
   A dedicated convergence rerun of this case shows the *angular integration*
   is fully converged: production grid (θ Jmax=6) gives Csca-X = 1.31894e7 µm²,
   a fine grid (θ Jmax=9 → 513 θ-nodes, φ Jmax=8) gives 1.31758e7 µm²
   (−0.1 %), and ADDA reports full Romberg convergence for both.
   The inconsistency is therefore a **dipole-grid artifact**: the automatic
   grid at `-eps 3` is only 18.0 dipoles/λ at x = 5, far outside the LDR
   validity range (≲ 6–10 dpl). Consequence: this baseline case is a valid
   deterministic *regression fingerprint*, but its absolute Csca/Qsca/ω are
   not physics references. The same caution applies to production runs at the
   large-x end of the backlog (x up to 10) — a minimum `-dpl` (or CLDR/DDSCAT
   polarizability) should be decided in Phase 3/5.
2. **Minor:** at x = 1 (dpl = 40) ω reaches 1.0025 in Y — same effect,
   harmless (~0.2 %), and useful as a regression tripwire: any change that
   moves it should be explainable.
3. **Reproduction noise:** a plain rerun (without `-store_scat_grid`) of the
   x = 5 case agrees with the baseline to 0.03 % (solver path differences),
   confirming the frozen files are stable under identical semantics.

## 5. Figures

![Cross sections and single-scattering albedo, X vs Y incidence
polarization, all six baseline cases (log scale).](report/fig1_crosssections.png)

![Sphere baseline vs Mie theory: convergence of Qext with dipoles per
wavelength, and deviations at the automatic grid.](report/fig2_sphere_mie.png)

![S11 3D radiation surfaces (Mie-style: radial extent ∝ S11^0.4, color = S11 normalized
per case, beam axis marked; interpolated from the coarsened 10° output grid). The
near-Rayleigh particles (sphere, x=0.1 spheroid, both Hong plates) are essentially
isotropic shells; x=1 develops diffraction rings; x=5 shows the classic narrow
forward spike with a ringed side-lobe structure.](report/fig3_scattering_maps.png)

![Rayleigh → resonance → forward-peak evolution of the analytic oblate spheroid as
3D radiation surfaces (each normalized to its own max), plus polarization
discrimination |S12|/S11 in the φ=0 plane.](report/fig4_ellipsoid_phase.png)

## 6. Reproduce

```sh
mamba activate DDA
python baselines/plot_baseline_figs.py      # figures (PNG 250 dpi + vector PDF)
python baselines/check_baselines.py         # Mie consistency, exit code = pass/fail
python baselines/make_baseline.py --adda <binary> --out baselines/<tag>   # rerun cases
```
