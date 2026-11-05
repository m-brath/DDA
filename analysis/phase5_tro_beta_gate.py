#!/usr/bin/env python3
"""Phase-5 §5.3 TRO gate: orientation-averaging with XY-folded beta range
(0-90 deg, justified ONLY by the voxel-exact XY closure of the TOR size grid)
vs the full-range twin (avg_params_fullrange.dat). Compares, per probe job:
CrossSec (Cext/Cabs) and the 91-theta orientation-averaged Mueller/amplitude
file. Mismatch must sit at Romberg/solver-noise level (log_orient_avg shows
the achieved outer-beta d per run; gate = 1e-3 covers worst-case stalled
Romberg at Jmax=4 + solver noise).

Usage: python analysis/phase5_tro_beta_gate.py
(reads the committed probe stores under adda_results/HongPlate_Id9_TOR_sizegrid/)
"""
import io
import os
import re
import sys
import zipfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(REPO, "adda_results", "HongPlate_Id9_TOR_sizegrid")

cases = ["deq00005um_freq228.0GHz", "deq00005um_freq886.0GHz",
         "deq00010um_freq228.0GHz", "deq00010um_freq886.0GHz"]


def read_zip(fname, member):
    with zipfile.ZipFile(os.path.join(STORE, fname)) as zf:
        return zf.read(member).decode()


def crosssec(fname):
    txt = read_zip(fname, "CrossSec")
    return {k: float(v) for k, v in
            re.findall(r"(C\w+|Q\w+)\s*=\s*([0-9.eE+-]+)", txt)}


def mueller(fname):
    a = np.loadtxt(io.StringIO(read_zip(fname, "mueller")), skiprows=1)
    return a  # theta x (16 amplitude)


def beta_d(fname):
    txt = read_zip(fname, "log_orient_avg")
    m = re.search(r"reached d=([0-9.eE+-]+)", txt)
    return float(m.group(1)) if m else np.nan


worst = 0.0
for c in cases:
    fa = "HongPlate_Id9_TOR_%s_T0230K.zip" % c
    fb = "HongPlate_Id9_TOR_fullr_%s_T0230K.zip" % c
    Ca, Cb = crosssec(fa), crosssec(fb)
    dc = max(abs(Ca[k] - Cb[k]) / abs(Cb[k]) for k in ("Cext", "Cabs"))
    Ma, Mb = mueller(fa), mueller(fb)
    assert Ma.shape == Mb.shape
    dm = np.max(np.abs(Ma - Mb)) / np.max(np.abs(Mb))
    worst = max(worst, dc, dm)
    print("%-28s CrossSec reldiff %.2e   mueller reldiff %.2e   "
          "beta-d folded=%s full=%s"
          % (c, dc, dm, beta_d(fa), beta_d(fb)))

print("WORST folded-vs-fullrange reldiff:", "%.2e" % worst)
sys.exit(0 if worst < 1e-3 else 1)
