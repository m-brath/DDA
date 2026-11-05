#!/usr/bin/env python3
"""Phase-4 core physics gate: derive + VERIFY the mueller_scatgrid transform
between ADDA runs with mirrored PROPAGATION directions for a mirror-symmetric
particle.

This is what the pipeline's skipped-vertex reconstruction must implement
(adda_sub_control_spharmonics.py:326 skips negative-coordinate vertices;
orientational_avg:1320 refuses to average the missing Mueller data — the
'Symmetry is still under construction' raise).

Method (empirical, no convention guessing): run ADDA twice, propagation p and
S p (S = particle mirror symmetry), then FIT the mapping between the two
mueller_scatgrids: theta-map x phi-map x signed permutation of the 16 Mueller
components. A residual ~ solver noise proves the transform.

Cases: oblate spheroid (-shape ellipsoid 1.0 0.5, rotationally symmetric, XY
and XZ mirror planes) and, as control, cases where the transform must be
trivial. ADDA 1.2, coarse baseline scat grid, auto dpl.
"""
import itertools
import os
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
import adda_control_functions as acf  # noqa: E402

EXE = os.path.join(REPO, "adda_1.2", "src", "seq", "adda")
SCAT_P = os.path.join(REPO, "baselines", "scat_params_baseline.dat")
C0 = 299792458.0


def run(exe, prop, shape_clo, lamb, eq_rad, m, tmp):
    inp = {"lambda": lamb, "Size": eq_rad,  # acf passes -eq_rad Size directly
           "refrative_index": m, "theta": list(prop),
           "alpha": 0, "beta": 0, "gamma": 0, "eps": 3,
           "ramdrive": tmp, "zipname": os.path.join(tmp, "z.zip"),
           "main_command": [exe],
           "static_clo": list(shape_clo) + ["-store_scat_grid", "-scat_grid_inp",
                                            SCAT_P, "-scat_matr", "both"],
           "zipping": False}
    return acf.adda_call(inp)


def grids(out):
    """mueller (Nth,Nph,16), thetas, phis as parsed by the pipeline"""
    M = np.asarray(out["mueller_array"], dtype=float)
    th = np.asarray(out["thetas_grid"], dtype=float)
    ph = np.asarray(out["phis_grid"], dtype=float)
    return M, th, ph


def colmaps(th, ph):
    """candidate index maps: theta identity/reverse; phi identity / (360-phi)
    / (180-phi) — implemented as column (not row) index maps"""
    nth, nph = len(th), len(ph)
    thmaps = {"id": np.arange(nth), "rev": nth - 1 - np.arange(nth)}
    def phimap(expr):
        vals = []
        for p in ph:
            v = expr(p) % 360.0
            j = np.argmin(np.abs(ph - v))
            if abs(ph[j] - v) > 1e-6:
                return None
            vals.append(j)
        return np.array(vals)
    phmaps = {"id": np.arange(nph),
              "neg": phimap(lambda p: -p),
              "180-neg": phimap(lambda p: 180.0 - p),
              "180+id": phimap(lambda p: 180.0 + p)}
    phmaps = {k: v for k, v in phmaps.items() if v is not None}
    return thmaps, phmaps


def fit_signed_perm(B, A):
    """B,M1 = A_permuted @ P with P signed permutation; returns (perm, signs,
    max relative residual) or None if no bijection found."""
    n = B.shape[-1]
    A2 = A.reshape(-1, n)
    B2 = B.reshape(-1, n)
    perm, signs = {}, {}
    for c in range(n):
        best = None
        for d in range(n):
            den = np.max(np.abs(A2[:, d]))
            if den == 0:
                continue
            for s in (1, -1):
                r = np.max(np.abs(B2[:, c] - s * A2[:, d])) / den
                if best is None or r < best[0]:
                    best = (r, d, s)
        if best is None or best[0] > 1e-8:
            return None
        perm[c] = best[1]
        signs[c] = best[2]
    if len(set(perm.values())) != n:
        return None
    # full residual check
    P = np.zeros((n, n))
    for c in range(n):
        P[perm[c], c] = signs[c]
    res = np.max(np.abs(A2 @ P - B2)) / np.max(np.abs(B2))
    return perm, signs, res


def main():
    tmp = tempfile.mkdtemp(prefix="p4sym_")
    f = 183e9
    lamb = C0 / f * 1e6
    m = np.sqrt(acf.eps_ice_matzler06(f, 253.0))
    shape = ["-shape", "ellipsoid", "1.0", "0.5"]

    tests = [
        ("XY mirror (z->-z), generic p",
         [0.30, 0.50, 0.81], [0.30, 0.50, -0.81]),
        ("XZ mirror (y->-y), generic p",
         [0.30, 0.50, 0.81], [0.30, -0.50, 0.81]),
        ("YZ mirror (x->-x), generic p",
         [0.30, 0.50, 0.81], [-0.30, 0.50, 0.81]),
    ]
    for name, p1, p2 in tests:
        p1 = np.asarray(p1, dtype=float) / np.linalg.norm(p1)
        p2 = np.asarray(p2, dtype=float) / np.linalg.norm(p2)
        A, th, ph = grids(run(EXE, p1, shape, lamb, 260.7292436741498, m, tmp))
        B, th2, ph2 = grids(run(EXE, p2, shape, lamb, 260.7292436741498, m, tmp))
        print(f"\n=== {name} ===")
        found = False
        for tname, ti in colmaps(th, ph)[0].items():
            for pname, pi in colmaps(th, ph)[1].items():
                At = A[ti[:, None], pi[None, :], :].copy()
                # theta reversal also reverses which mueller rows/cols get
                # relabelled; the signed-perm fit absorbs that
                fit = fit_signed_perm(B, At)
                if fit:
                    perm, signs, res = fit
                    print(f"  MATCH th={tname:8s} ph={pname:8s} resid={res:.2e}")
                    print(f"    comp perm: {[perm[c] for c in range(16)]}")
                    print(f"    comp sign: {[signs[c] for c in range(16)]}")
                    found = True
        if not found:
            print("  no transform found in candidate space")


if __name__ == "__main__":
    main()
