#!/usr/bin/env python3
"""Phase-5 step 22: ARO backlog inventory (plan step 17-19 cost anchors +
Phase-4 symmetry factors) — per habit: how many (T,f,size) jobs the PRODUCTION
configs now define, and a CPU-hour estimate per job from the measured cost
model (analysis/phase4b_bench_results.json anchors).

Reads the config globals the same way adda_main_control.py does (exec of
config + the same f/fmin/size-parameter selection), counts jobs and dipoles.
"""
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
C0 = 299792458.0


def load_cfg(name):
    ns = {}
    path = os.path.join(REPO, "adda_python_scripts", name + ".py")
    code = open(path).read()
    # configs use __file__ (hostname/local branches) — provide it
    ns["__file__"] = path
    exec(compile(code, path, "exec"), ns)
    return ns


def count_jobs(cfg):
    sizes, fmins, shapelist = cfg["Sizes"], cfg["fmin"], cfg["shapelist"]
    jobs = []
    for T in cfg["temperatures"]:
        for f in cfg["frequencies"]:
            for D in np.unique(sizes):
                r = np.where(sizes == D)[0]
                df = fmins[r] - f
                df[df < 0] = np.inf
                if not np.isfinite(df).any():
                    continue
                if np.pi * D * 1e-6 * (f / C0) > cfg.get("size_parameter_limit", 10):
                    continue
                jobs.append((T, f, D, shapelist[r[np.argmin(df)]]))
    return jobs


def ndipoles(shapefile):
    n = 0
    with open(shapefile) as fh:
        for _ in fh:
            n += 1
    return max(n - 1, 1)  # header line


def solve_cost_s(N):
    """measured single-core anchors (1.4.0, x=5: N=12312 -> 0.42s;
    x=8: N=48597 -> 9.52s) + N log N extrapolation to production N"""
    if N <= 12312:
        return 0.42 * N / 12312
    return 9.52 * (N / 48597) * (np.log(N) / np.log(48597)) ** 2


SYM = {"sym_XYobj": 0, "sym_XZobj": 0, "sym_YZobj": 0}

for habit in ("adda_config_HongColumn_Id7", "adda_config_IconSnow_Id28",
              "adda_config_LiuSectorSnowflake_Id3", "adda_config_HongPlate_Id9"):
    cfg = load_cfg(habit)
    jobs = count_jobs(cfg)
    # cache dipoles per shape file
    cache = {}
    tot_s, tot_s_asym = 0.0, 0.0
    mirror_on = sum(1 for k in ("sym_XYobj", "sym_XZobj", "sym_YZobj")
                    if cfg.get(k, False))
    red = 1.0 / (2 ** mirror_on) if mirror_on else 1.0
    for (T, f, D, sf) in jobs:
        if sf not in cache:
            cache[sf] = ndipoles(sf)
        # ARO: one solve per icosphere vertex set: production first full
        # refinement level ~642 (levels 12+42+162 warmup cheaper; approximate
        # total ~ 642+162+42+12 = 858 at converged level-3; use 858 as anchor)
        c = solve_cost_s(cache[sf]) * 858
        tot_s_asym += c
        tot_s += c * red
    print(f"{habit:38s} jobs={len(jobs):5d}  mirrors={mirror_on} "
          f"({red*100:.0f}% solves)  est single-core ARO solve-time: "
          f"{tot_s/3600:10.1f} CPU-h (asym would be {tot_s_asym/3600:.1f})")
