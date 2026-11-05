#!/usr/bin/env python3
"""Phase-4b step 18: ADDA-internal cost/accuracy benchmark on the oblate
spheroid (the pipeline's analytic test shape: ellipsoid 1.0 0.5, ice
dielectric Matzler 2006 @ 230 K, 183 GHz), using the PRODUCTION binary
(ADDA 1.4.0, seq build).

Drives ADDA directly (not via acf.adda_call — that adds zip + grid-parsing
noise to the timings) and parses its stdout banner:
    Total number of occupied dipoles: N
    Cext = ...    Cabs = ...

For a grid of (size parameter x, -eps, -dpl) it records wall time, N_dipoles,
Cext, Cabs; errors are relative to the finest self-converged run of the same
(x, eps) family. Feeds the production cost model: how far to push dpl/eps as
x grows toward size_parameter_limit=10.

Run:  python analysis/phase4b_adda_benchmark.py
Output: analysis/phase4b_bench_results.json (+ table on stdout)
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
import adda_control_functions as acf  # noqa: E402

EXE = os.path.join(REPO, "adda_1.4", "src", "seq", "adda")
C0 = 299792458.0
F = 183e9
LAMB_UM = C0 / F * 1e6  # wavelength in µm (pipeline unit)
M = np.sqrt(np.complex128(acf.eps_ice_matzler06(F, 230.0)))
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "phase4b_bench")

RE_N = re.compile(r"Total number of occupied dipoles:\s+(\d+)")
RE_CEXT = re.compile(r"^Cext\s*=\s*([0-9.eE+-]+)", re.M)
RE_CABS = re.compile(r"^Cabs\s*=\s*([0-9.eE+-]+)", re.M)


def run(x, eps, dpl):
    """Oblate spheroid (axis ratio 0.5) at size parameter x of the
    volume-equivalent sphere (same x definition as the pipeline:
    x = k * a_eq); dpl=None -> ADDA automatic dpl heuristic."""
    aeq = x * LAMB_UM / (2 * np.pi)  # -eq_rad of the vol-equiv sphere
    outdir = os.path.join(TMP, "run_x%g_e%s_d%s" % (x, eps, dpl))
    shutil.rmtree(outdir, ignore_errors=True)  # ADDA refuses an existing -dir
    os.makedirs(TMP, exist_ok=True)
    cmd = [EXE, "-shape", "ellipsoid", "1.0", "0.5"]
    if dpl is not None:
        cmd += ["-dpl", str(dpl)]
    # NOTE: str/repr of numpy scalars must be unwrapped (np.float64(...) in
    # the string breaks ADDA's param parsing) — use format spec.
    cmd += ["-m", "%.12g" % M.real, "%.12g" % M.imag,
            "-lambda", "%.12g" % LAMB_UM, "-eq_rad", "%.12g" % aeq,
            "-orient", "0", "0", "0",
            "-prop", "0", "0", "1", "-eps", str(eps), "-dir", outdir]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError("ADDA failed (%d): %s" % (p.returncode,
                                                     p.stdout[-2000:]))
    out = p.stdout + p.stderr
    n = int(RE_N.search(out).group(1))
    cext = float(RE_CEXT.search(out).group(1))
    cabs = float(RE_CABS.search(out).group(1))
    dpl_eff = re.search(r"Dipoles/lambda:\s+([0-9.eE+-]+)", out)
    return {"x": x, "eps": eps, "dpl": dpl,
            "dpl_eff": float(dpl_eff.group(1)) if dpl_eff else None,
            "N_dipoles": n, "wall_s": round(dt, 2), "Cext": cext, "Cabs": cabs}


CASES = [
    # A: eps ladder at fixed geometry (x=5, auto dpl): solver accuracy vs
    #    runtime; reference = eps=2 (strictest)
    (5, 4, None), (5, 3, None), (5, 2, None),
    # B: dpl ladder at fixed x=5, eps 3: discretization cost vs accuracy;
    #    reference = dpl 7
    (5, 3, 4), (5, 3, 5), (5, 3, 6), (5, 3, 7),
    # C: x ladder at production setting (eps 3, auto dpl): cost growth with
    #    x; the x=8 manual-dpl twin checks the auto-heuristic grid at the
    #    production cap (x=5,auto is shared with family A)
    (1, 3, None), (3, 3, None), (8, 3, None), (8, 3, 6),
]


def main():
    os.makedirs(TMP, exist_ok=True)
    res = {}
    for x, eps, dpl in CASES:
        r = run(x, eps, dpl)
        print(json.dumps(r), flush=True)
        res[(x, eps, dpl)] = r
    auto = lambda d: d is None
    print("\n x  eps dpl  dpl_eff  N_dip  wall_s   Cerr vs ref          "
          "Cabserr")
    for (x, eps, dpl), r in res.items():
        if x == 5 and auto(dpl):
            ref, fam = res[(5, 2, None)], ("self" if eps == 2 else "eps->eps2")
        elif x == 5:
            ref, fam = res[(5, 3, 7)], ("self" if dpl == 7 else "dpl->dpl7")
        elif (x, eps, dpl) == (8, 3, None):
            ref, fam = res[(8, 3, 6)], "auto-vs-dpl6"
        elif (x, eps, dpl) == (8, 3, 6):
            ref, fam = None, "-"
        else:
            ref, fam = None, "-"
        if ref is None:
            derr = dabs = float("nan")
        else:
            derr = abs(r["Cext"] - ref["Cext"]) / abs(ref["Cext"])
            dabs = abs(r["Cabs"] - ref["Cabs"]) / abs(ref["Cabs"])
        print("%-3d %-3d %-4s %-8.3f %-6d %-8.2f   %.3e  %.3e  (%s)"
              % (x, eps, str(dpl or "auto"), r["dpl_eff"], r["N_dipoles"],
                 r["wall_s"], derr, dabs, fam))
    vals = sorted(res.values(), key=lambda r: r["N_dipoles"])
    print("\nwall-time vs N_dipoles (single core):")
    for r in vals:
        print("  N=%-6d  %8.2f s" % (r["N_dipoles"], r["wall_s"]))
    with open(os.path.join(REPO, "analysis", "phase4b_bench_results.json"),
              "w") as fh:
        json.dump([res[k] for k in CASES], fh, indent=1)
    print("\nsaved: analysis/phase4b_bench_results.json")


if __name__ == "__main__":
    main()
