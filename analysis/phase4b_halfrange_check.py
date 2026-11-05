#!/usr/bin/env python3
"""Phase-4b step 17: can the beta/incidence-angle grids be quadrature-optimized
(Gauss-Legendre nodes, half-range under mirror symmetry)?

Finding (documented, NOT implemented): beta and theta_inc are NOT quadrature
axes of the pipeline. The ADDA solves happen on the icosphere VERTEX set
(alpha=beta=gamma=0, incidence = vertex direction); beta/incidence grids are
SSDB OUTPUT coordinates on which the vertex data are interpolated + averaged
post-hoc (aoa.rawdata2average, ~6.5 s per (beta,theta) pair). Therefore:
  (a) Gauss-Legendre nodes on beta/theta change zero ADDA solves — they only
      change the output sampling (and would need spharmonic re-expansion at
      the new nodes: same post-processing cost either way);
  (b) half-range reduction (0-90 deg instead of 0-180) is NOT raw identity:
      this probe re-averages one committed symmetry-enabled store on PAIRED
      grids b vs 180-b / t vs 180-t and measures the mismatch. If a mirror
      symmetry mapped the output tensor onto itself, pairs would match to
      solver noise; they do not (the alpha-averaging at each output node is
      a different vertex subset, and vpol/phi bookkeeping does not transform
      trivially). A correct reduction would need the proper group action on
      the averaged products — not implemented; the numbers below show why
      naive half-range reuse is wrong.

Usage: python analysis/phase4b_halfrange_check.py <store_tag> [<Tglob>]
Reproduction: run aoa.rawdata2average twice-equally on the store with the
paired grids below (absolute store path required — internal os.chdir).
"""
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
import adda_config_phase4_hp_sym as cfg  # noqa: E402
import adda_orientational_avg_spharmonics_functions as aoa  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "phase4_hp_sym"
tglob = sys.argv[2] if len(sys.argv) > 2 else "*T0190K"
store = glob.glob(os.path.join(REPO, "adda_results", tag, tglob))[0]
assert os.path.isabs(store)
out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "phase4b_halfrange")
os.makedirs(out, exist_ok=True)

# paired grids: every node's 180-degree partner is present
betas = np.array([0.0, 36.0, 72.0, 108.0, 144.0, 180.0])
incs = np.array([0.0, 18.0, 72.0, 90.0, 108.0, 162.0, 180.0])

od = aoa.rawdata2average(store + "/", incs, betas, out, 0, 0, 0,
                         parallel=False)
E = np.asarray(od["ext_mat_horiz_avg"])
M = np.asarray(od["mue_mat_horiz_avg_sph"])


def rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


cb = [list(betas).index(180 - b) for b in betas]
ct = [list(incs).index(180 - t) for t in incs]
print("beta  pair (b -> 180-b): ext %.3e   mueller %.3e"
      % (rel(E[cb], E), rel(M[cb], M)))
print("theta pair (t -> 180-t): ext %.3e   mueller %.3e"
      % (rel(E[:, ct, :], E), rel(M[:, ct, :], M)))
print("(solver noise reference is ~1e-11; mismatch >> noise => naive "
      "half-range reuse would be a real error)")
