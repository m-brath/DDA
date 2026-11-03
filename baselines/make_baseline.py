#!/usr/bin/env python3
"""
Generate frozen baseline outputs of ADDA runs for regression testing.

These baseline cases mirror the exact ADDA command-line pattern produced by
adda_control_functions.adda_call / adda_sub_control_spharmonics.py (ARO pipeline):
  adda <static_clo> -m <re> <im> -lambda <lamb_um> -eq_rad <radius_um>
       -orient <alpha> <beta> <gamma> -prop <px py pz> -dir <dir> -eps <eps>

Usage:
  python make_baseline.py --adda <path-to-adda-binary> --out <output-dir>

The same script is used to regenerate the cases with newer ADDA binaries
(Phase 3 equivalence checks of the modernization plan). Compare the outputs
numerically (all files except adda_log / stdout.log, which contain build and
timing info).
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C0 = 299792458.0  # m/s

SCAT_PARAMS = os.path.join(REPO, "baselines", "scat_params_baseline.dat")
ALLDIR_PARAMS = os.path.join(REPO, "adda_python_scripts", "alldir_params.dat")
SHAPE_MAIN = os.path.join(REPO, "adda_shape_files")

# orientation/propagation mirror the pipeline base call:
# alpha_0=0, beta_0=0, gamma_0=0, incidence angle theta=90 deg
ORIENT = (0.0, 0.0, 0.0)
PROP = (1.0, 0.0, 0.0)
EPS = 3


def eps_ice_matzler06(f, TK):
    """Permittivity of fresh water ice, Matzler 2006 (copy of acf.eps_ice_matzler06)."""
    f = f / 1e9
    deltabeta = np.exp(-9.963 + 0.0372 * (TK - 273.0))
    betam = (0.0207 / TK) * (np.exp(335.0 / TK) / ((np.exp(335.0 / TK) - 1.0) ** 2.0)) + 1.16e-11 * f**2.0
    beta = betam + deltabeta
    theta = 300.0 / TK - 1.0
    alfa = (0.00504 + 0.0062 * theta) * np.exp(-22.1 * theta)
    z = 3.1884 + 9.1e-4 * (TK - 273.0)
    return z + 1j * (alfa / f + beta * f)


def m_ice(f, TK):
    return np.sqrt(eps_ice_matzler06(f, TK))


def lamb_um(f):
    return C0 / f * 1e6


def build_cases():
    cases = []

    # 1) homogeneous sphere of ice - analytic (Mie) validation case
    f = 300e9
    T = 230.0
    cases.append(dict(
        name="sphere_ice_x0.31_f300GHz_T230",
        shape_str=["-shape", "sphere"],
        eq_rad=50.0, lamb=lamb_um(f), m=m_ice(f, T)))

    # 2) oblate spheroid (analytic shape, from adda_config_ellipsoid_oblate.py:
    #    -shape ellipsoid 1.0 0.5, f=183 GHz, T=253 K, x in {0.1, 1, 5})
    f = 183e9
    T = 253.0
    for x in (0.1, 1.0, 5.0):
        cases.append(dict(
            name=f"ellipsoid_oblate_x{x:g}_f183GHz_T253",
            shape_str=["-shape", "ellipsoid", "1.0", "0.5"],
            eq_rad=lamb_um(f) * x / (2.0 * math.pi),
            lamb=lamb_um(f), m=m_ice(f, T)))

    # 3/4) real habit: HongPlate_Id9, smallest size (Dveq=10um), two frequencies
    T = 230.0
    for f, shapefile in ((192e9, "Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0192.000GHz.adda"),
                         (337e9, "Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0337.000GHz.adda")):
        cases.append(dict(
            name=f"hongplate_id9_dveq10_f{f/1e9:g}GHz_T230",
            shape_str=["-shape", "read", os.path.join(SHAPE_MAIN, "HongPlate_Id9", shapefile)],
            eq_rad=5.0, lamb=lamb_um(f), m=m_ice(f, T)))

    return cases


def run_case(case, adda_bin, out_dir):
    command = ([adda_bin]
               + case["shape_str"]
               + ["-store_scat_grid", "-scat_grid_inp", SCAT_PARAMS,
                  "-scat_matr", "both", "-Csca", "-alldir_inp", ALLDIR_PARAMS]
               + ["-m", "%.17g" % case["m"].real, "%.17g" % case["m"].imag]
               + ["-lambda", "%.17g" % case["lamb"]]
               + ["-eq_rad", "%.17g" % case["eq_rad"]]
               + ["-orient", *["%.17g" % a for a in ORIENT]]
               + ["-prop", *["%.17g" % p for p in PROP]]
               + ["-eps", str(EPS)])

    print(">>", case["name"])
    with tempfile.TemporaryDirectory() as tmp:
        rundir = os.path.join(tmp, "out")
        os.makedirs(rundir, exist_ok=True)
        cmd = command + ["-dir", rundir]
        print("   " + " ".join(cmd))
        res = subprocess.run(cmd, capture_output=True, text=True)
        with open(os.path.join(rundir, "stdout.log"), "w") as f:
            f.write(res.stdout + "\n" + res.stderr)
        if res.returncode != 0:
            print("   FAILED (exit %d)" % res.returncode, file=sys.stderr)
            print(res.stdout[-2000:], res.stderr[-2000:], file=sys.stderr)
            return False

        case_out = os.path.join(out_dir, case["name"])
        os.makedirs(case_out, exist_ok=True)
        for fn in sorted(os.listdir(rundir)):
            shutil.copy(os.path.join(rundir, fn), os.path.join(case_out, fn))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adda", required=True, help="path to adda (seq) binary")
    ap.add_argument("--out", required=True, help="output directory for baselines")
    ap.add_argument("--cases", default=None, help="comma-separated subset of case names")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ok = True
    for case in build_cases():
        if args.cases and case["name"] not in args.cases.split(","):
            continue
        ok &= run_case(case, os.path.abspath(args.adda), args.out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
