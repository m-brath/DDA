#!/usr/bin/env python3
"""Phase-4 gate part 2: exercise aoa.rawdata2average (the SSDB-export
post-processing path) on a symmetry-enabled store and its non-symmetric twin,
and compare every averaged product. Reproduction of the report §4.9 aoa gate:

    python adda_sub_control_spharmonics.py <T> <f> <D> <config_asym>
    python adda_sub_control_spharmonics.py <T> <f> <D> <config_sym>
    python analysis/phase4_aoa_compare.py phase4_hp_asym phase4_hp_sym T0190K
"""
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
import adda_config_phase4_hp_sym as cfg  # noqa: E402
import adda_orientational_avg_spharmonics_functions as aoa  # noqa: E402

OUT = os.path.join(os.environ.get("TMPDIR", "/tmp"), "phase4_aoa_avg")

tag_a, tag_b, tmodes = (sys.argv[1], sys.argv[2], sys.argv[3]
                        if len(sys.argv) > 3 else "*T0190K")

res = {}
for tag in (tag_a, tag_b):
    store = glob.glob(os.path.join(REPO, "adda_results", tag, tmodes))[0] + "/"
    outp = os.path.join(OUT, tag + "_avg/")
    os.makedirs(outp, exist_ok=True)
    res[tag] = aoa.rawdata2average(store, cfg.incidence_angles,
                                   np.linspace(0.0, 180.0, num=cfg.number_of_betas),
                                   outp, 0, 0, 0, parallel=False)

a, b = res[tag_a], res[tag_b]
worst = 0.0
SKIP = ('sym_XYobj', 'sym_XZobj', 'sym_YZobj')  # store metadata: differ by design
for k in sorted(a):
    if k in SKIP:
        continue
    xa, xb = a[k], b[k]
    if isinstance(xa, str) or isinstance(xb, str):
        print(f"{k:35s} (str) equal={xa == xb}")
        if xa != xb:
            sys.exit(1)
        continue
    xa, xb = np.asarray(xa, dtype=complex), np.asarray(xb, dtype=complex)
    if xa.shape != xb.shape:
        print(f"{k:35s} SHAPE MISMATCH {xa.shape} {xb.shape}")
        sys.exit(1)
        continue
    both_nan = np.isnan(xa) & np.isnan(xb)  # eps_vec has NaN for unused passes
    d = np.max(np.abs(np.where(both_nan, 0, xa - xb))) / max(np.nanmax(np.abs(xb)), 1e-300)
    worst = max(worst, float(d))
    print(f"{k:35s} shape={xa.shape} maxreldiff={d:.3e}")
print("WORST reldiff over all averaged products:", f"{worst:.3e}")
# noise-level gates (solver noise, cf. Phase-3 report): cross sections 1e-10,
# Mueller sph coefficients 1e-9
sys.exit(0 if worst < 1e-9 else 1)
