#!/usr/bin/env python3
"""Phase-4 gate part 1: compare the sub_control store artifacts
(overview_data.mat + mueller_data.pkl) of a symmetry-enabled run and its
non-symmetric twin. Reproduction of the report §4.9 sub_control gates:

    python analysis/phase4_subcontrol_compare.py phase4_hp_asym phase4_hp_sym '*T0190K'
"""
import glob
import os
import pickle
import sys

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(tag, tmodes):
    st = glob.glob(os.path.join(REPO, "adda_results", tag, tmodes))[0]
    mat = sio.loadmat(glob.glob(st + "/*overview_data.mat")[0])["output_data"].ravel()[0]
    ext = np.asarray(mat["ext_mat_horiz_avg"], dtype=float)
    eps = np.asarray(mat["eps_vec"], dtype=float).ravel()
    M = pickle.load(open(glob.glob(st + "/*mueller_data.pkl")[0], "rb"))
    return ext, eps, M


def rel(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    return np.max(np.abs(np.where(both_nan, 0, a - b))) / max(np.nanmax(np.abs(b)), 1e-300)


tag_a, tag_b, tmodes = (sys.argv[1], sys.argv[2],
                        sys.argv[3] if len(sys.argv) > 3 else "*T0*")
ea, epsa, Ma = load(tag_a, tmodes)
eb, epsb, Mb = load(tag_b, tmodes)

d_ext = rel(ea, eb)
d_eps = rel(epsa, epsb)
# robust comparison: per vertex, per component, over the coefficient overlap
# (eps_sphexp truncation can flip a few trailing coefficients at solver-noise
# level, and per-vertex coefficient counts vary — flattening by component
# would desync vertices). Globally normalized (max |diff| / max |all|):
# per-component normalization would punish near-zero-amplitude components,
# which is solver noise amplified, not a systematic difference.
worst = 0.0
den = 0.0
for va, vb in zip(Ma, Mb):
    for ca, cb in zip(va, vb):
        ca = np.asarray(ca, dtype=complex).ravel()
        cb = np.asarray(cb, dtype=complex).ravel()
        n = min(len(ca), len(cb))
        if n:
            worst = max(worst, float(np.max(np.abs(ca[:n] - cb[:n]))))
            den = max(den, float(np.max(np.abs(cb))))
n_a = sum(np.asarray(c, dtype=complex).size for v in Ma for c in v)
n_b = sum(np.asarray(c, dtype=complex).size for v in Mb for c in v)
d_mue = worst / max(den, 1e-300)

print(f"ext_mat_horiz_avg reldiff: {d_ext:.3e}")
print(f"eps_vec          reldiff: {d_eps:.3e}   (mueller coeffs {n_a} vs {n_b})")
print(f"mueller coeffs   reldiff: {d_mue:.3e}")
# noise gates: solver noise level (identical-command ADDA re-run scatter, cf.
# Phase-3 report) — ext 1e-10, expansion coefficients 1e-9
sys.exit(0 if (d_ext < 1e-10 and d_eps < 1e-10 and d_mue < 1e-9) else 1)
