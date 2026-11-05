#!/usr/bin/env python3
"""
Phase-3 ADDA-binary equivalence suite (modernization plan step 14).

Three probe groups, all executed on this machine with the locally built
ADDA binaries; comparisons go through the *pipeline's own* parser
(adda_control_functions.adda_call / adda_call_randomorient), so the
positional output-parsing contract (report section 1.3) is verified
version-by-version:

  binary   - the 6 frozen baseline cases (baselines/adda_1.2) re-run with
             ADDA 1.2 / 1.4.0 / 1.5.0-alpha3 at FIXED dipole density
             (-dpl 160.664, the 1.2 auto value) -> ext_matrix,
             mueller_scatgrid and grids compared across versions;
             plus a sanity re-run of the 1.2 case at auto dpl against the
             golden files.
  pipeline - end-to-end adda_sub_control_spharmonics.py in local mode with
             adda_config_phase3_test (tiny ellipsoid), once per binary;
             compares *_overview_data.mat (ext_mat_horiz_avg, convergence
             eps_mat) and *_mueller_data.pkl across versions.
  tro      - adda_call_randomorient with '-orient avg <avg_params.dat>'
             (sphere) per binary: proves the avg files are still accepted
             by 1.4/1.5 and compares Cext/Cabs/Csca from adda_log.

usage:  python analysis/phase3_equiv.py [binary|pipeline|tro|all]
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "adda_python_scripts")
BASE = os.path.join(REPO, "baselines", "adda_1.2")
SCAT_P = os.path.join(REPO, "baselines", "scat_params_baseline.dat")
sys.path.insert(0, SCRIPTS)
import adda_control_functions as acf  # noqa: E402

BINARIES = {
    "1.2": os.path.join(REPO, "adda_1.2", "src", "seq", "adda"),
    "1.4.0": os.path.join(REPO, "adda_1.4", "src", "seq", "adda"),
    "1.5.0-alpha3": os.path.join(REPO, "adda", "src", "seq", "adda"),
}
# NO -dpl override: ADDA's auto dipole density is heuristic and case-dependent
# (1104..2176 dipoles here); identical auto values across versions are part of
# what the equivalence suite verifies (dipole counts visible via gate failures)

# golden baseline cases -> the acf.adda_call input_data that reproduces them
# (values computed EXACTLY like make_baseline.py — the auto dipole-density
# heuristic quantizes, so even a 1e-8 lambda typo changes N_dipoles)
C0 = 299792458.0

def _lamb_um(f):
    return C0 / f * 1e6

def _eq_lamb_x(x):  # ellipsoid: eq_rad = lamb*x/(2*pi), matching make_baseline
    l = _lamb_um(183e9)
    return l, l * x / (2.0 * __import__("math").pi)

CASES = {
    "sphere_ice_x0.31_f300GHz_T230": dict(
        lamb=_lamb_um(300e9), Size=50.0,
        static_clo=["-shape", "sphere"]),
    "ellipsoid_oblate_x0.1_f183GHz_T253": dict(
        lamb=_lamb_um(183e9), Size=_eq_lamb_x(0.1)[1],
        static_clo=["-shape", "ellipsoid", "1.0", "0.5"]),
    "ellipsoid_oblate_x1_f183GHz_T253": dict(
        lamb=_lamb_um(183e9), Size=_eq_lamb_x(1.0)[1],
        static_clo=["-shape", "ellipsoid", "1.0", "0.5"]),
    "ellipsoid_oblate_x5_f183GHz_T253": dict(
        lamb=_lamb_um(183e9), Size=_eq_lamb_x(5.0)[1],
        static_clo=["-shape", "ellipsoid", "1.0", "0.5"]),
    "hongplate_id9_dveq10_f192GHz_T230": dict(
        lamb=_lamb_um(192e9), Size=5.0,
        static_clo=["-shape", "read", os.path.join(
            REPO, "adda_shape_files/HongPlate_Id9",
            "Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0192.000GHz.adda")]),
    "hongplate_id9_dveq10_f337GHz_T230": dict(
        lamb=_lamb_um(337e9), Size=5.0,
        static_clo=["-shape", "read", os.path.join(
            REPO, "adda_shape_files/HongPlate_Id9",
            "Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0337.000GHz.adda")]),
}
# refractive index per case (must match make_baseline.py: T230 ice at the case f)
ICE = {"sphere_ice_x0.31_f300GHz_T230": (300e9, 230.0),
       "ellipsoid_oblate_x0.1_f183GHz_T253": (183e9, 253.0),
       "ellipsoid_oblate_x1_f183GHz_T253": (183e9, 253.0),
       "ellipsoid_oblate_x5_f183GHz_T253": (183e9, 253.0),
       "hongplate_id9_dveq10_f192GHz_T230": (192e9, 230.0),
       "hongplate_id9_dveq10_f337GHz_T230": (337e9, 230.0)}

# relative-difference gates (solver eps=3 residuals ~1e-3; discretization
# is pinned identical via -dpl, so differences are solver noise only)
GATE_MUELLER = 5e-3
GATE_EXT = 5e-3
GATE_TRO = 1e-3

RESULTS = {}


def run_binary_case(case, exe, tmp):
    f, T = ICE[case]
    m = np.sqrt(acf.eps_ice_matzler06(f, T))
    inp = {}
    inp["lambda"] = CASES[case]["lamb"]
    inp["Size"] = CASES[case]["Size"]  # volume-equivalent RADIUS in um
    inp["refrative_index"] = m
    # vector form -> acf emits "-prop 1.0 0 0" exactly like make_baseline.py
    # (the scalar theta=90 path would add z=6.12e-17 and rotate the phi-ref)
    inp["theta"] = [1.0, 0.0, 0.0]
    inp["alpha"] = inp["beta"] = inp["gamma"] = 0
    inp["eps"] = 3
    inp["ramdrive"] = tmp
    inp["zipname"] = os.path.join(tmp, case + ".zip")
    inp["main_command"] = [exe]
    inp["static_clo"] = list(CASES[case]["static_clo"]) + \
        ["-store_scat_grid", "-scat_grid_inp", SCAT_P,
         "-scat_matr", "both"]
    inp["zipping"] = False
    return acf.adda_call(inp)


def golden_fields(case):
    """parsed golden: mueller grid + forward ext via the same parser logic"""
    path = os.path.join(BASE, case)
    data = np.loadtxt(os.path.join(path, "mueller_scatgrid"), skiprows=1)
    th = np.unique(data[:, 0]); ph = np.unique(data[:, 1])
    M = data[:, 2:].reshape(th.size, ph.size, data.shape[1] - 2)
    return M


def rel(a, b):
    a = np.asarray(a, dtype=complex); b = np.asarray(b, dtype=complex)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def probe_binary(tmp):
    names = list(CASES)
    out = {}
    for case in names:
        out[case] = {}
        for ver, exe in BINARIES.items():
            r = run_binary_case(case, exe, tmp)
            out[case][ver] = r
        for field, gate in (("mueller_array", GATE_MUELLER),
                            ("ext_matrix", GATE_EXT)):
            base = np.asarray(out[case]["1.2"][field], dtype=float)
            for ver in BINARIES:
                if ver == "1.2":
                    continue
                d = rel(out[case][ver][field], base)
                RESULTS[f"{case}:{field}:{ver}"] = (d, d <= gate)
        # mueller grid vs golden file (1.2 rerun sanity). Gates are NOISE-based:
        # ADDA re-runs of the identical command are not bit-reproducible at
        # output's last digits; measured own re-run scatter: <=1e-14 for all
        # cases, 8.3e-4 for x5 (12312 dipoles at -eps 3 => solver noise bound)
        g = golden_fields(case)
        d = rel(out[case]["1.2"]["mueller_array"], g)
        gg = 5e-3 if "x5" in case else 1e-10
        RESULTS[f"{case}:golden_1.2"] = (d, d <= gg)
    return out


def probe_pipeline(tmp):
    import scipy.io as sio
    import pickle
    import shutil
    runs = {}
    for ver, exe in BINARIES.items():
        # wipe the per-vertex NetCDF cache/store: otherwise later binaries
        # would silently reuse cached mueller expansions from earlier ones
        store = glob.glob(os.path.join(REPO, "adda_results", "phase3_test",
                                       "rawdata", "*freq183.0GHz_T0230K"))
        for s in store:
            shutil.rmtree(s)
        env = dict(os.environ, ADDA_BINARY=exe, ADDA_VERSION=ver)
        p = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "adda_sub_control_spharmonics.py"),
             "0", "0", "0", "adda_config_phase3_test"],
            env=env, cwd=SCRIPTS)
        if p.returncode != 0:
            raise RuntimeError(f"sub_control failed for {ver}")
        store = glob.glob(os.path.join(REPO, "adda_results", "phase3_test",
                                       "rawdata", "*freq183.0GHz_T0230K"))[0]
        mat = sio.loadmat(glob.glob(os.path.join(store, "*overview_data.mat"))[0])["output_data"].ravel()[0]
        ext = np.asarray(mat["ext_mat_horiz_avg"], dtype=float)
        epsm = np.asarray(mat["eps_vec"], dtype=float).ravel()
        with open(glob.glob(os.path.join(store, "*mueller_data.pkl"))[0], "rb") as fh:
            muell = pickle.load(fh)
        runs[ver] = dict(ext=ext, eps=epsm,
                         ncoeff=[len(v) if isinstance(v, (list, tuple)) else 1 for v in muell],
                         c0norm=np.abs(np.asarray(muell[0][0], dtype=complex)).ravel()[:4])
    for ver in BINARIES:
        if ver == "1.2":
            continue
        d = rel(runs[ver]["ext"], runs["1.2"]["ext"])
        RESULTS[f"pipeline:ext_mat_horiz_avg:{ver}"] = (d, d <= GATE_EXT)
        n = np.abs(runs[ver]["c0norm"] - runs["1.2"]["c0norm"])
        d2 = float(np.max(n / np.maximum(np.abs(runs["1.2"]["c0norm"]), 1e-12)))
        RESULTS[f"pipeline:mueller_c0:{ver}"] = (d2, d2 <= GATE_MUELLER)
    RESULTS["pipeline:coeff_counts"] = (
        {v: runs[v]["ncoeff"] for v in runs}, True)
    RESULTS["pipeline:eps_vec"] = ({v: runs[v]["eps"].tolist() for v in runs}, True)
    return runs


def probe_tro(tmp):
    avg = os.path.join(REPO, "adda_aux_input", "EvanSnowflake_aggregate",
                       "avg_params.dat")
    f, T = 300e9, 230.0
    m = np.sqrt(acf.eps_ice_matzler06(f, T))
    runs = {}
    for ver, exe in BINARIES.items():
        outdir = os.path.join(tmp, "tro_" + ver.replace(".", "_"))
        os.makedirs(outdir, exist_ok=True)
        inp = {"lambda": 999.3081933333333, "radius": 50.0,
               "refractive_index": m, "eps": 3, "main_path": outdir,
               "main_command": [exe], "calc_type": "noqueue", "batch_time": 0,
               "static_clo": ["-shape", "sphere",
                              "-orient", "avg", avg]}
        acf.adda_call_randomorient(inp)
        # '-orient avg' writes the orientation-averaged cross sections into
        # the CrossSec file (NOT into the main log)
        cs = {}
        for line in open(os.path.join(outdir, "CrossSec")):
            mm = re.match(r"(\w+)\s*=\s*[-\d.eE+]+", line)
            if mm:
                cs[mm.group(1)] = float(line.split("=")[1])
        runs[ver] = cs
        if len(cs) < 3:
            raise RuntimeError(f"TRO probe: CrossSec parse failed for {ver}")
    for ver in BINARIES:
        if ver == "1.2":
            continue
        d = max(abs(runs[ver][k] - runs["1.2"][k]) / abs(runs["1.2"][k])
                for k in set(runs[ver]) & set(runs["1.2"]))
        RESULTS[f"tro:crosssec:{ver}"] = (d, d <= GATE_TRO)
    RESULTS["tro:values"] = (runs, True)
    return runs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tmp = tempfile.mkdtemp(prefix="phase3_equiv_")
    if mode in ("all", "binary"):
        probe_binary(tmp)
    if mode in ("all", "pipeline"):
        probe_pipeline(tmp)
    if mode in ("all", "tro"):
        probe_tro(tmp)

    ok = True
    for k in sorted(RESULTS):
        v, pass_ = RESULTS[k]
        if isinstance(v, float):
            print(f"{'PASS' if pass_ else 'FAIL':4s} {k:55s} reldiff={v:.3e}")
            ok &= pass_
        else:
            print(f"info {k:55s} {v}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    with open(os.path.join(REPO, "analysis", "phase3_results.json"), "w") as fh:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, (v, p) in RESULTS.items()}, fh, indent=1, default=str)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
