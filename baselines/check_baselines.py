#!/usr/bin/env python3
"""
Analytic sanity checks for baseline cases.

Checks the homogeneous-sphere baseline case against independent Mie theory
(Bohren & Huffman / Mishchenko convention, direct Riccati-Bessel evaluation).
Used to certify baselines/adda_1.2 and to re-validate newer ADDA binaries.
"""

import os
import re
import sys

import numpy as np
from scipy.special import spherical_jn, spherical_yn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mie_qext_qsca(x, m, nmax=None):
    if nmax is None:
        nmax = int(x + 4 * x ** (1 / 3.0) + 10)
    n = np.arange(1, nmax + 1)
    z = m * x

    def psi_n(arg):  # Riccati-Bessel psi_n = arg*j_n(arg)
        return arg * spherical_jn(n, arg)

    def dpsi_n(arg):
        return spherical_jn(n, arg) + arg * spherical_jn(n, arg, derivative=True)

    psi_x = psi_n(x)
    dpsi_x = dpsi_n(x)

    # xi_n(x) = x*h_n^(1)(x) = psi_n(x) + i*chi_n(x), chi_n = -x*y_n(x)
    chi_x = -x * spherical_yn(n, x)
    dchi_x = -spherical_yn(n, x) - x * spherical_yn(n, x, derivative=True)
    xi_x = psi_x + 1j * chi_x
    dxi_x = dpsi_x + 1j * dchi_x

    psi_z, dpsi_z = psi_n(z), dpsi_n(z)

    an = (m * psi_z * dpsi_x - psi_x * dpsi_z) / (m * psi_z * dxi_x - xi_x * dpsi_z)
    bn = (psi_z * dpsi_x - m * psi_x * dpsi_z) / (psi_z * dxi_x - m * xi_x * dpsi_z)

    Qext = 2.0 / x**2 * np.sum((2 * n + 1) * (an.real + bn.real))
    Qsca = 2.0 / x**2 * np.sum((2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2))
    return Qext, Qsca


def read_crosssec(path):
    """Parse ADDA CrossSec-X/Y file (key = value lines) into dict."""
    out = {}
    with open(path) as f:
        for line in f:
            mm = re.match(r"(\w+)\s*=\s*([-+\d.eE]+)", line)
            if mm:
                out[mm.group(1)] = float(mm.group(2))
    return out


def check_sphere(case_dir, eq_rad, lamb, m):
    """Compare sphere-case cross sections against independent Mie theory.

    Note on conventions: this Mie helper uses the e^{+ikr} convention in which
    absorption requires Im(m) < 0, while ADDA (and eps_ice_matzler06 -> sqrt)
    use Im(m) > 0 for absorption. We pass the conjugated m here.
    Tolerance is loose (3%) on purpose: at the small auto-selected dipole
    density of this tiny case, DDA discretization error vs Mie is ~2%
    (verified to converge: Qext 6.914e-3 @160 dpl, 6.852e-3 @320,
    6.819e-3 @800 -> Mie 6.793e-3).
    """
    x = 2 * np.pi * eq_rad / lamb
    Qext, Qsca = mie_qext_qsca(x, np.conj(m))
    ok = True
    for pol in ("X", "Y"):
        cs = read_crosssec(os.path.join(case_dir, f"CrossSec-{pol}"))
        for key, mie_val in (("Qext", Qext), ("Qsca", Qsca), ("Qabs", Qext - Qsca)):
            if key in cs:
                rel = abs(cs[key] / mie_val - 1.0)
                status = "OK" if rel < 3e-2 else "MISMATCH"
                if rel >= 3e-2:
                    ok = False
                print(f"  {pol}: {key:4s} adda={cs[key]:.9e} mie={mie_val:.9e} "
                      f"reldev={rel:.2e} [{status}]")
    return ok


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "baselines", "adda_1.2")
    case = os.path.join(base, "sphere_ice_x0.31_f300GHz_T230")
    print("sphere case:", case)
    ok = check_sphere(case, eq_rad=50.0, lamb=999.30819333333329,
                      m=1.7746225684057333 + 0.0039064427503776146j)
    sys.exit(0 if ok else 1)
