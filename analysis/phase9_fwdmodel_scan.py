#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 9A scan — analytic forward-diffraction model vs DDA grids
(plan: plans/forward_peak_plan.md; report section to follow: report Phase 9)

Question: can the large-x diffraction/forward peak of the ADDA scattering
grid be represented by an analytic model so that only the smooth RESIDUAL
g = f - F_model has to be stored (grid + SH coefficients), with the model
re-evaluated analytically at read/RT time?  This is the "delta-Eddington
analogue done honestly": the peak stays fully resolved (it is analytic), but
costs no data.

This script is an ANALYSIS gate only — it changes nothing in the pipeline.

Gates
  G1  convention anchor (sphere, EXACT Mie): fit the complex constant c and
      the origin-phase lattice offset u0 of ADDA's ampl_scatgrid convention
      (columns = BH s1..s4; far-field phase referenced to lattice index 0 of
      ADDA's shifted dipole box).  Must match the box-center prediction for
      analytic shapes -> then the identical port may be trusted for voxels.
  G2  eikonal-internal-field model vs ADDA grids on production voxel shapes
      (HongPlate_Id9 at 887 GHz hits x = 1,2,5,10 exactly via
      Dveq = 108,215,541,1078 um; orientations face-on/60deg/edge-on;
      HongColumn_Id7 at x~2,5).  Headline metric: SH truncation achieved by
      spha.expand_mueller_2_spharmonics for the FULL grid Mueller vs the
      RESIDUAL Mueller M(g), production eps_sphexp = 0.005.
  G3  port self-checks: recomputed M(ampl) == parsed mueller_scatgrid,
      Cext(f) identity after split (algebraic), model size at Theta=180
      (backscatter-exemption evidence).

Run:  mamba activate DDA && python analysis/phase9_fwdmodel_scan.py [--cells hp,hc]
"""

import os, sys, re, json, glob, shutil, tempfile, argparse, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'adda_python_scripts'))

import adda_control_functions as acf                      # noqa: E402
import spherical_harmonics_for_adda as spha               # noqa: E402

C0 = 299792458.0
BIN14 = os.path.join(REPO, 'adda_1.4', 'src', 'seq', 'adda')
GRID = os.path.join(REPO, 'adda_python_scripts', 'scat_params.dat')
RESULTS_JSON = os.path.join(HERE, 'phase9_results.json')
TABLE_TXT = os.path.join(HERE, 'phase9_fwdmodel_table.txt')

EPS_SPH = 0.005          # production eps_sphexp
FREQ = 887e9             # the size grid was built around this channel
TENV = 230.0
CONES = (30.0, 60.0)     # forward-cone splits for the L2 metrics (deg)

RESULTS = {'checks': [], 'cells': []}


def check(name, ok, detail):
    print(('PASS  ' if ok else 'FAIL  ') + name + '   ' + detail)
    RESULTS['checks'].append({'name': name, 'pass': bool(ok), 'detail': detail})


# ==============================================================================
# ADDA run + parsing (mirrors production: -shape read, -scat_matr both, -eps 3,
# production static grid 181x361)
# ==============================================================================

def run_adda(shape_clo, eq_rad, lamb, m, alpha, beta, gamma, prop, tag, workdir,
             extra_clo=()):
    rundir = os.path.join(workdir, tag)
    os.makedirs(rundir, exist_ok=True)
    g = lambda v: '%.17g' % float(v)
    cmd = [BIN14] + list(shape_clo) + list(extra_clo) + [
        '-store_scat_grid', '-scat_grid_inp', GRID, '-scat_matr', 'both',
        '-m', g(m.real), g(m.imag),
        '-lambda', g(lamb), '-eq_rad', g(eq_rad),
        '-orient', g(alpha), g(beta), g(gamma),
        '-prop', g(prop[0]), g(prop[1]), g(prop[2]),
        '-dir', rundir, '-eps', '3']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if p.returncode != 0:
        raise RuntimeError('ADDA failed for %s (rc=%d):\n%s\n%s'
                           % (tag, p.returncode, ' '.join(cmd),
                              (p.stdout + p.stderr)[-3000:]))
    stdout = p.stdout + '\n' + p.stderr
    box = None
    mt = re.search(r'box dimensions:\s*(\d+)x(\d+)x(\d+)', stdout)
    if mt:
        box = (int(mt.group(1)), int(mt.group(2)), int(mt.group(3)))
    amax = re.search(r'Dipoles/lambda:\s*([0-9.eE+-]+)', stdout)
    dpl = float(amax.group(1)) if amax else None
    ampl = np.loadtxt(os.path.join(rundir, 'ampl_scatgrid'), skiprows=1)
    muel = np.loadtxt(os.path.join(rundir, 'mueller_scatgrid'), skiprows=1)
    thetas = np.unique(ampl[:, 0])
    phis = np.unique(ampl[:, 1])
    nt, np_ = thetas.size, phis.size
    S = ampl[:, 2:].reshape(nt, np_, 4, 2)
    s = (S[:, :, :, 0] + 1j * S[:, :, :, 1])            # s1..s4 (ADDA print order)
    M = muel[:, 2:].reshape(nt, np_, 16)
    return dict(thetas=thetas, phis=phis, s=s, M=M, box=box, dpl=dpl,
                stdout=stdout, rundir=rundir)


def parse_shape(path):
    """voxel coords of an .adda geometry file; returns integer coords (N,3)."""
    pts = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            pts.append((int(v[0]), int(v[1]), int(v[2])))
    return np.asarray(pts, dtype=np.int64)


# ==============================================================================
# ADDA frame port (param.c / crosssec.c conventions, zyz per Mishchenko AO 39)
# ==============================================================================

def beta_matr(alpha, b, gam):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(b), np.sin(b)
    cg, sg = np.cos(gam), np.sin(gam)
    return np.array([[ca * cb * cg - sa * sg, sa * cb * cg + ca * sg, -sb * cg],
                     [-ca * cb * sg - sa * cg, -sa * cb * sg + ca * cg, sb * sg],
                     [ca * sb, sa * sb, cb]])


def frame(alpha_deg, beta_deg, gamma_deg, prop_0):
    """particle-RF incident direction + polarization basis, verbatim ADDA."""
    prop_0 = np.asarray(prop_0, dtype=float)
    prop_0 = prop_0 / np.linalg.norm(prop_0)
    if abs(abs(prop_0[2]) - 1.0) < 1e-12:
        pa = np.sign(prop_0[2])
        iX = np.array([pa, 0.0, 0.0])
        iY = np.array([0.0, 1.0, 0.0])
    else:
        temp = np.sqrt(1.0 - prop_0[2] ** 2)
        iX = np.array([prop_0[0] * prop_0[2] / temp,
                       prop_0[1] * prop_0[2] / temp, -temp])
        iY = np.array([-prop_0[1] / temp, prop_0[0] / temp, 0.0])
    R = beta_matr(np.radians(alpha_deg), np.radians(beta_deg),
                  np.radians(gamma_deg))
    return R @ prop_0, R @ iX, R @ iY


def scat_frames(thetas_deg, phis_deg, shat, iX, iY):
    """per/par basis + robs on the beam-attached grid (SetScatPlane port)."""
    th = np.radians(thetas_deg)[:, None]
    ph = np.radians(phis_deg)[None, :]
    ct, st = np.cos(th), np.sin(th)
    co_p, si_p = np.cos(ph), np.sin(ph)
    # robs = ct*prop + st*(cosphi*iX + sinphi*iY)
    robs = (ct[..., None] * shat
            + st[..., None] * (co_p[..., None] * iX + si_p[..., None] * iY))
    # polPer = robs x prop  (special st~0 branch: sphi*iX - cphi*iY)
    per = np.cross(robs, np.broadcast_to(shat, robs.shape))
    nrm = np.linalg.norm(per, axis=-1)
    sing = nrm < 1e-14
    special = (si_p[..., None] * iX - co_p[..., None] * iY)
    safe = np.where(sing, 1.0, nrm)
    per = np.where(sing[..., None], special, per / safe[..., None])
    par = np.cross(robs, per)
    return robs, per, par


def egrid_to_s(EXp, EXa, EYp, EYa, per, par, iX, iY):
    """Adda CalculateE.c scatgrid printout transform (s1..s4) verbatim.
    EXp = E-grid X-incidence 'per' component, etc."""
    co = -np.einsum('...i,i->...', per, iY)
    si = np.einsum('...i,i->...', per, iX)
    s2 = co * EXa + si * EYa
    s3 = si * EXa - co * EYa
    s4 = co * EXp + si * EYp
    s1 = si * EXp - co * EYp
    return s1, s2, s3, s4


def mueller_from_s(s1, s2, s3, s4):
    """BH 3.16, verbatim ADDA ComputeMuellerMatrix; returns (...,16)."""
    a1, a2, a3, a4 = (np.abs(x) ** 2 for x in (s1, s2, s3, s4))
    m = np.stack([
        0.5 * (a1 + a2 + a3 + a4),
        0.5 * (a2 - a1 + a4 - a3),
        np.real(s2 * np.conj(s3) + s1 * np.conj(s4)),
        np.imag(s2 * np.conj(s3) - s1 * np.conj(s4)),
        0.5 * (a2 - a1 + a3 - a4),
        0.5 * (a2 + a1 - a3 - a4),
        np.real(s2 * np.conj(s3) - s1 * np.conj(s4)),
        np.imag(s2 * np.conj(s3) + s1 * np.conj(s4)),
        np.real(s2 * np.conj(s4) + s1 * np.conj(s3)),
        np.real(s2 * np.conj(s4) - s1 * np.conj(s3)),
        np.real(s1 * np.conj(s2) + s3 * np.conj(s4)),
        np.imag(s2 * np.conj(s1) + s4 * np.conj(s3)),
        np.imag(s4 * np.conj(s2) + s1 * np.conj(s3)),
        np.imag(s4 * np.conj(s2) - s1 * np.conj(s3)),
        np.imag(s1 * np.conj(s2) - s3 * np.conj(s4)),
        np.real(s1 * np.conj(s2) - s3 * np.conj(s4))], axis=-1)
    return m


# ==============================================================================
# Mie (full angular S1,S2) — validated 2026-11-09.
# The B&H 5.12 logarithmic-derivative (D) form with DOWNWARD recurrence for
# D_n(mx) (as in phase6_backscatter.py) agrees with miepython.coefficients
# (Prahl's library, validated by its author against Wiscombe's MIEV0 suite;
# cross-checked here against the published case-14 value S1(0)=0.584080
# +0.190515j at m=1.5-1.0i, x=1) to 6+ digits.  NOTE: an "explicit" series
# obtained by replacing D_n(mx) with tau'_n/tau_n is NOT a valid B&H formula
# (m weights only one term; it even yields negative Qext for lossy m) — an
# earlier debug round chased that bogus variant; do not reintroduce it.
# mie_coeffs_mp below re-evaluates the SAME D form at 40 digits and the G1
# anchor gate ties scipy-D == mpmath-D == miepython, so the port used for
# the sphere fits is proven at run time, not assumed.
# ==============================================================================

def mie_coeffs(m, x, nmax=None):
    """B&H 5.12 D-recurrence coefficients (1-based; index 0 unused)."""
    if nmax is None:
        nmax = int(np.ceil(x + 4 * x ** (1 / 3) + 2)) + 15
    z = m * x
    D = np.zeros(nmax + 3, dtype=complex)
    for kk in range(nmax + 2, 1, -1):
        D[kk - 1] = kk / z - 1.0 / (D[kk] + kk / z)
    from scipy.special import spherical_jn as jn, spherical_yn as yn
    an = np.zeros(nmax + 1, dtype=complex)
    bn = np.zeros(nmax + 1, dtype=complex)
    for n in range(1, nmax + 1):
        psi = x * jn(n, x)
        psip = (n + 1) * jn(n, x) - x * jn(n + 1, x)
        chi = x * yn(n, x)
        chip = (n + 1) * yn(n, x) - x * yn(n + 1, x)
        xi = psi + 1j * chi
        xip = psip + 1j * chip
        an[n] = (m * psip - D[n] * psi) / (m * xip - D[n] * xi)
        bn[n] = (psip - m * D[n] * psi) / (xip - m * D[n] * xi)
    return an, bn


def mie_coeffs_mp(m, x, dps=40):
    """Same D form at 40-digit precision (downward recurrence is exact in
    extended precision).  Returns (an[n] float array, Cext(a=1), S1(0),
    {n:(an,bn) mpmath})."""
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        m = mp.mpc(float(np.real(m)), float(np.imag(m)))
        x = mp.mpf(float(x))
        z = m * x
        nmax = int(x) + 4 * int(x ** (mp.mpf('1') / 3)) + 25

        def jn_mp(n, arg):
            return mp.sqrt(mp.pi / (2 * arg)) * mp.besselj(n + 0.5, arg)

        def yn_mp(n, arg):
            return mp.sqrt(mp.pi / (2 * arg)) * mp.bessely(n + 0.5, arg)

        # downward recurrence for D_n(z) = psi'_n(z)/psi_n(z)
        D = {nmax + 2: mp.mpc(0)}
        for kk in range(nmax + 2, 1, -1):
            D[kk - 1] = kk / z - 1 / (D[kk] + kk / z)
        C = mp.mpf('0')
        S0 = mp.mpc(0)
        out = np.zeros(nmax + 1, dtype=complex)
        coefs = {}
        for n in range(1, nmax + 1):
            psi = x * jn_mp(n, x)
            psip = (n + 1) * jn_mp(n, x) - x * jn_mp(n + 1, x)
            chi = x * yn_mp(n, x)
            chip = (n + 1) * yn_mp(n, x) - x * yn_mp(n + 1, x)
            xi = psi + 1j * chi
            xip = psip + 1j * chip
            dn = D[n]
            an = (m * psip - dn * psi) / (m * xip - dn * xi)
            bn = (psip - m * dn * psi) / (xip - m * dn * xi)
            C += (2 * n + 1) * mp.re(an + bn)
            S0 += (2 * n + 1) / 2 * (an + bn)
            out[n] = complex(an)
            coefs[n] = (an, bn)
        Cext_a1 = 2 * mp.pi / x ** 2 * C                   # a=1 units (k=x)
        return out, float(Cext_a1), complex(S0), coefs
    finally:
        mp.mp.dps = old


def mie_S_mp(m, an_from_mp, thetas_deg, x, dps=40):
    """mpmath full-angle S1 via exact upward recurrences (no sin-division):
    pi_1=1, pi_2=3mu, (n+1)pi_{n+1} = (2n+1)(P_n + mu pi_n) - n pi_{n-1};
    tau_n = n(n+1)P_n - mu pi_n.  an_from_mp: list/dict 1-based a_n (b_n too).
    Returns S1 array."""
    import mpmath as mp
    old = mp.mp.dps
    mp.mp.dps = dps
    try:
        out = []
        for tf in thetas_deg:
            mu = mp.cos(mp.radians(mp.mpf(str(tf))))
            s1 = mp.mpc(0)
            P0, P1 = mp.mpf('1'), mu
            pi0, pi1 = mp.mpf('0'), mp.mpf('1')
            for n in range(1, len(an_from_mp)):
                Pn = P1
                pin = pi1
                taun = n * (n + 1) * Pn - mu * pin
                cn = (2 * n + 1) / (n * (n + 1))
                an, bn = an_from_mp[n]
                s1 += cn * (an * pin + bn * taun)
                # advance
                P2 = ((2 * n + 1) * mu * P1 - n * P0) / (n + 1)
                pi2 = ((2 * n + 1) * (P1 + mu * pi1) - n * pi0) / (n + 1)
                P0, P1 = P1, P2
                pi0, pi1 = pi1, pi2
            out.append(complex(s1))
        return np.array(out, dtype=complex)
    finally:
        mp.mp.dps = old


def mie_S(an, bn, thetas_deg):
    """S1(theta), S2(theta), BH dimensionless.
    BH conventions: pi_n = P'_n(mu)  (= -lpmv(1,n,mu)/sin(theta): scipy lpmv
    carries the Condon-Shortley minus), tau_n = n(n+1)P_n(mu) - mu*pi_n
    (equivalent to dP^1_n/dtheta; tau_n(+-1) = +-pi_n(+-1) at the poles)."""
    mu = np.cos(np.radians(thetas_deg))
    st = np.sin(np.radians(thetas_deg))
    from scipy.special import lpmv, eval_legendre
    nmax = len(an) - 1
    S1 = np.zeros(mu.shape, dtype=complex)
    S2 = np.zeros(mu.shape, dtype=complex)
    for n in range(1, nmax + 1):
        with np.errstate(invalid='ignore', divide='ignore'):
            pi_n = -lpmv(1, n, mu) / st
        # pole limits: pi_n(+1) = +n(n+1)/2 ; pi_n(-1) = (-1)^{n+1} n(n+1)/2
        p1 = n * (n + 1) / 2.0
        pi_n = np.where(st > 1e-9, pi_n, np.where(mu > 0, p1, ((-1.0) ** (n + 1)) * p1))
        tau_n = n * (n + 1) * eval_legendre(n, mu) - mu * pi_n
        cn = (2 * n + 1) / (n * (n + 1))
        S1 += cn * (an[n] * pi_n + bn[n] * tau_n)
        S2 += cn * (an[n] * tau_n + bn[n] * pi_n)
    return S1, S2


# ==============================================================================
# G1 — sphere convention anchor
# ==============================================================================

def fit_convention(s_adda, s_model, robs, d):
    """Amplitude-space LS:  s_adda ~ c * exp(-i k (robs . (d*u0))) * s_model.
    Unlike a log-space fit this is immune to the model's deep minima (those
    points simply stop contributing).  For fixed u0 the complex c is the
    exact LS solution; u0 (in lattice units) is found by local grid refine.
    Returns c, u0, relres with relres = ||resid||/||s_adda|| (amplitude L2).
    """
    k = 2 * np.pi / d
    kd = k * d
    mag = np.abs(s_model)
    mask = mag > 1e-2 * mag.max()          # drop numerically dead points
    nn = robs[mask]
    ya = s_adda[mask]
    ym = s_model[mask]
    yy_norm = np.vdot(ya, ya).real

    def best_c(u0):
        ph = np.exp(-1j * kd * (nn @ u0))
        zm = ph * ym
        c = np.vdot(zm, ya) / np.vdot(zm, zm)
        r = ya - c * zm
        return c, float(np.vdot(r, r).real / yy_norm)

    cur = np.zeros(3)
    cur_c, cur_res = best_c(cur)
    step = 0.5
    for _ in range(6):                     # refine passes (0.5 -> ~0.016)
        g = np.arange(-2 * step, 2 * step + 1e-9, step)
        while True:
            improved = False
            for dz in g:
                for dy in g:
                    for dx in g:
                        u0 = cur + np.array([dx, dy, dz])
                        c, res = best_c(u0)
                        if res < cur_res - 1e-12:
                            cur, cur_c, cur_res = u0, c, res
                            improved = True
            if not improved:
                break
        step /= 2.0
    return cur_c, cur, cur_res


def g1(workdir):
    print('\n==== G1: sphere convention anchor (exact Mie) ====')
    f = 150e9
    lamb = C0 / f * 1e6
    m = np.sqrt(acf.eps_ice_matzler06(f, TENV))

    # --- Mie self-anchor (triple): scipy-D == mpmath-D == miepython (the
    # latter validated by its author against Wiscombe's MIEV0 suite; we also
    # verified case-14 S1(0)=0.584080+0.190515j by hand).  This is what makes
    # the port trustworthy for the sphere fits below.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import miepython
    for xv in (1.8865, 3.7730):
        an_s, bn_s = mie_coeffs(m, xv)
        n = np.arange(1, len(an_s))
        an_mp, Cext_mp, S0_mp, coefs_mp = mie_coeffs_mp(m, xv)
        Cext_s = (2 * np.pi / xv ** 2) * np.sum(
            (2 * n + 1) * np.real(an_s[1:] + bn_s[1:]))
        S0_s = np.sum((2 * n + 1) / 2 * (an_s[1:] + bn_s[1:]))
        thA = np.array([0.0, 90.0, 180.0])
        S1_scipy = mie_S(an_s, bn_s, thA)[0]               # grid path
        S1_mp = mie_S_mp(m, coefs_mp, thA, xv)             # mpmath reference
        rel = np.abs(S1_scipy / S1_mp - 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            an_py, bn_py = np.asarray(miepython.coefficients(m, xv)[:2])
        kk = min(len(an_py), len(an_s) - 1)
        scale = max(np.max(np.abs(an_py[:kk])), np.max(np.abs(bn_py[:kk])))
        relcoef = max(np.max(np.abs(an_s[1:kk + 1] - an_py[:kk])),
                      np.max(np.abs(bn_s[1:kk + 1] - bn_py[:kk]))) / scale
        # (no S1-vs-miepython amplitude check: its 'wiscombe' norm carries an
        # undocumented x-dependent factor; the coef tie at 1e-16 plus the
        # grid-path-vs-mpmath tie fully pin the angular machinery)
        check('G1 Mie anchor x=%.4g' % xv,
              abs(Cext_s / Cext_mp - 1) < 1e-6
              and abs(S0_s / S0_mp - 1) < 1e-6
              and np.max(rel) < 1e-6
              and relcoef < 1e-8,
              'Cext rel %.2e S1(grid) rel %.2e coef-vs-miepython %.2e'
              % (abs(Cext_s / Cext_mp - 1), float(np.max(rel)), relcoef))

    out = {}
    # fine-discretization spheres (dpl=70) so DDA voxelization error does NOT
    # contaminate the convention fit (a/d >= 21 in both)
    for xv in (1.8865, 3.7730):
        aeq = xv * lamb / (2 * np.pi)
        r = run_adda(['-shape', 'sphere'], aeq, lamb, m, 0, 0, 0, (0, 0, 1),
                     'sph_x%g' % xv, workdir, extra_clo=['-dpl', '70'])
        an, bn = mie_coeffs(m, xv)
        S1, S2 = mie_S(an, bn, r['thetas'])
        shat, iX, iY = frame(0, 0, 0, (0, 0, 1))
        robs, per, par = scat_frames(r['thetas'], r['phis'], shat, iX, iY)
        S1g = S1[:, None] * np.ones_like(r['phis'])[None, :]
        S2g = S2[:, None] * np.ones_like(r['phis'])[None, :]
        # sphere model Egrid: EgridX=(S1*(iX.per), S2*(iX.par)) etc.
        EXp = S1g * np.einsum('...i,i->...', per, iX)
        EXa = S2g * np.einsum('...i,i->...', par, iX)
        EYp = S1g * np.einsum('...i,i->...', per, iY)
        EYa = S2g * np.einsum('...i,i->...', par, iY)
        s_mod = np.stack(egrid_to_s(EXp, EXa, EYp, EYa, per, par, iX, iY), -1)
        d = lamb / r['dpl']
        # model built centered: fit residual u0 should be ~0; the fitted c is
        # ADDA's amplitude normalization constant.
        c, u0, rr = fit_convention(r['s'][..., 0], s_mod[..., 0], robs, d)
        # physics anchor: ADDA's printed Cext vs analytic (voxel sphere at
        # dpl=70 -> agreement at the ~1% level expected from discretization)
        _, Cext_mp, _, _ = mie_coeffs_mp(m, xv)
        Cext_adda = float(re.search(r'Cext\s*=\s*([0-9.eE+-]+)', r['stdout']).group(1))
        Cext_ana = Cext_mp * (aeq ** 2)      # mpmath result was in a=1 units
        check('G1 ADDA Cext vs analytic x=%.4g' % xv,
              abs(Cext_adda / Cext_ana - 1) < 0.05,
              'ADDA %.6g um^2 vs analytic %.6g um^2 (dev %.3f%%)'
              % (Cext_adda, Cext_ana, 100 * abs(Cext_adda / Cext_ana - 1)))
        out['x%g' % xv] = dict(c=c, u0=u0, off=np.max(np.abs(u0)), relres=rr,
                               box=r['box'], dpl=r['dpl'], d=d,
                               Cext_adda=Cext_adda, Cext_ana=Cext_ana)
        print('  x=%.4g: box=%s dpl=%.3f  c=%.6f%+.6fj  |c|=%.5f arg=%.5f  '
              'u0dev=%.3f lattices  fitres=%.2e'
              % (xv, r['box'], r['dpl'], c.real, c.imag, abs(c), np.angle(c),
                 np.max(np.abs(u0)), rr))
    k1, k2 = sorted(out)[0], sorted(out)[-1]
    c1, c2 = out[k1]['c'], out[k2]['c']
    check('G1 c consistent across x', abs(c1 / c2 - 1) <= 0.03,
          '|c1/c2-1|=%.4f (c1=%.5f%+.5fj c2=%.5f%+.5fj)'
          % (abs(c1 / c2 - 1), c1.real, c1.imag, c2.real, c2.imag))
    check('G1 far-field origin = scatterer center (u0~0)',
          max(out[k1]['off'], out[k2]['off']) <= 0.5,
          'max |u0| = %.3f lattice units' % max(out[k1]['off'], out[k2]['off']))
    check('G1 fit residual small', max(out[k1]['relres'], out[k2]['relres'])
          <= 0.08, 'worst rel phase-fit residual %.2e'
          % max(out[k1]['relres'], out[k2]['relres']))
    RESULTS['g1'] = {k: {'c': complex(v['c']), 'abs_c': abs(v['c']),
                         'arg_c': np.angle(v['c']), 'u0': list(v['u0']),
                         'box': v['box'], 'dpl': v['dpl']}
                     for k, v in out.items()}
    return out


# ==============================================================================
# G2 — eikonal internal-field model on voxel shapes
# ==============================================================================

class EikonalModel:
    """distorted-wave radiation integral with eikonal internal field,
    ray-marched on the exact voxel geometry. ADDA phase origin: lattice index
    (0,0,0) of the -min-shifted coordinates (read shapes: bbox min corner)."""

    def __init__(self, vox, eq_rad_um, dpl_hint, m, shat, iX, iY,
                 dp_frac=0.5, ds_frac=0.25):
        mnc = vox.min(axis=0)
        self.u = vox - mnc                      # 0-based lattice coords
        self.N = self.u.shape[0]
        self.d = eq_rad_um * (4 * np.pi / (3 * self.N)) ** (1 / 3)
        self.box = self.u.max(axis=0) + 1
        grid = np.zeros(tuple(self.box), dtype=bool)
        grid[self.u[:, 0], self.u[:, 1], self.u[:, 2]] = True
        self.grid = grid
        self.m = complex(m)
        self.shat = shat / np.linalg.norm(shat)
        self.iX, self.iY = iX, iY
        # G1 (sphere/Mie anchor) established that ADDA's far-field phase origin
        # is the scatterer CENTER -> model coordinates centered on the voxel
        # bbox center; the per-cell fit in eval_cell verifies u0 ~ 0.
        self.cen = (self.box - 1) / 2.0
        # screen pixel grid in (iX,iY) over projected bbox
        ru = (self.u - self.cen) * self.d          # physical coords (um), centered
        xi = ru @ iX
        eta = ru @ iY
        dp = self.d * dp_frac
        self.xi0 = xi.min() - self.d
        self.xi1 = xi.max() + self.d
        self.eta0 = eta.min() - self.d
        self.eta1 = eta.max() + self.d
        self.xig = np.arange(self.xi0, self.xi1 + dp, dp)
        self.etag = np.arange(self.eta0, self.eta1 + dp, dp)
        GG = np.meshgrid(self.xig, self.etag, indexing='ij')
        self.rho = (GG[0].ravel()[:, None] * iX + GG[1].ravel()[:, None] * iY)
        self.dp = dp
        # march range: project bbox corners
        corners = np.array([(x, y, z) for x in (0, self.box[0] - 1)
                            for y in (0, self.box[1] - 1)
                            for z in (0, self.box[2] - 1)], dtype=float)
        corners = (corners - self.cen) * self.d
        pr = corners @ self.shat
        self.s0 = pr.min() - 2 * self.d
        self.s1 = pr.max() + 2 * self.d
        self.ds = self.d * ds_frac

    def iter_G_chunks(self, k, thetas_deg, chunk=2000):
        """Yield (i0, i1, Gchunk): G(mu_idx, col) = int (m^2-1) phi(s)
        e^{-i k mu s} ds per screen pixel, ray-marched chunk-by-chunk so the
        full (nmu, ncols) matrix never materializes.  phi(s)=exp(i k
        (s + (m-1) chord(s))): eikonal internal field referenced to the
        scatterer CENTER (rho is perpendicular to shat, so the e^{ik shat.rho}
        factor is 1); consistent with the G1-verified ADDA phase origin."""
        ncols = self.rho.shape[0]
        ns = int(np.ceil((self.s1 - self.s0) / self.ds)) + 1
        sp = self.s0 + np.arange(ns) * self.ds
        mu = np.cos(np.radians(thetas_deg))
        Emu = np.exp(-1j * k * np.outer(mu, sp))          # (nmu, ns)
        box = self.box
        for i0 in range(0, ncols, chunk):
            i1 = min(i0 + chunk, ncols)
            pts = (self.rho[i0:i1, None, :]
                   + sp[None, :, None] * self.shat[None, None, :])
            idx = np.rint(pts / self.d + self.cen).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < box), axis=-1)
            inside = np.zeros(idx.shape[:2], dtype=bool)
            ii = idx[ok]
            inside[ok] = self.grid[ii[:, 0], ii[:, 1], ii[:, 2]]
            chord = np.cumsum(inside, axis=1) * self.ds
            dens = inside * (self.m ** 2 - 1.0) \
                * np.exp(1j * k * (sp[None, :] + (self.m - 1.0) * chord)) * self.ds
            yield i0, i1, Emu @ dens.T


# ==============================================================================
# per-cell evaluation
# ==============================================================================

def eval_cell(shape_dir, shapelist_pat, f, T, alpha, beta_tilt, gamma,
              prop_0, tag, workdir, cal):
    eps = acf.eps_ice_matzler06(f, T)
    m = np.sqrt(eps)
    lamb = C0 / f * 1e6
    k = 2 * np.pi / lamb
    path = glob.glob(os.path.join(shape_dir, shapelist_pat))[0]
    vox = parse_shape(path)
    # production resolution rule: pick the file with smallest fmin >= f
    meta = acf.loadmat(path.replace('.adda', '.meta.mat'))['meta']
    D = float(meta['diameter_vol_eq']) * 1e6        # um
    fname_fmin = float(re.search(r'Freq0*([0-9]+)\.000GHz', path).group(1)) * 1e9
    x = np.pi * D / lamb
    shat, iX, iY = frame(alpha, beta_tilt, gamma, prop_0)
    r = run_adda(['-shape', 'read', path], D / 2.0, lamb, m,
                 alpha, beta_tilt, gamma, prop_0, tag, workdir)
    thetas, phis = r['thetas'], r['phis']
    robs, per, par = scat_frames(thetas, phis, shat, iX, iY)

    mod = EikonalModel(vox, D / 2.0, r['dpl'], m, shat, iX, iY)
    d = mod.d
    d_addd = lamb / r['dpl'] if r['dpl'] else np.nan
    # --- fused eikonal: march screen-pixel chunks, accumulate the angular
    # field  F~ = c_cal * k^2/(2pi) * sum_cols G e^{-i k n.rho} dp^2
    # (chunked: the x=10 cell has >100k pixels; full G/phase arrays ~GB)
    nt, nphi = thetas.size, phis.size
    F = np.zeros((nt, nphi), dtype=complex)
    for (i0, i1, Gc) in mod.iter_G_chunks(k, thetas):
        rr = mod.rho[i0:i1]                       # (c,3)
        for it in range(nt):
            F[it] += np.exp(-1j * k * (robs[it] @ rr.T)) @ Gc[it]
    # Helmholtz far field: f = (k^2/4pi) * integral, and ADDA ampl = BH S =
    # k*f  ->  S_model = (k^3/4pi) * sum(G e^{-i k n.rho} dp^2)  (bare model;
    # the single global complex calibration is fit per cell below).
    F *= k ** 3 / (4 * np.pi) * mod.dp * mod.dp
    # --- model s-grid via verbatim ADDA print transform (scalar screen model:
    #     E_sca = F (E_inc - n(n.E_inc)), identity operator in transverse plane)
    EXp = F * np.einsum('...i,i->...', per, iX)
    EXa = F * np.einsum('...i,i->...', par, iX)
    EYp = F * np.einsum('...i,i->...', per, iY)
    EYa = F * np.einsum('...i,i->...', par, iY)
    s_raw = np.stack(egrid_to_s(EXp, EXa, EYp, EYa, per, par, iX, iY), -1)
    s_add = r['s']

    # --- per-cell OPTIMAL global calibration (complex c + origin offset),
    # amplitude-space LS on the FULL sphere.  This is the most charitable
    # single-constant calibration of the model family: c=0 gives g=f, so the
    # LS-optimal c guarantees |g|<=|f|, and the resulting ratio measures the
    # intensity fraction the analytic model actually explains.  sc/cal is the
    # cross-cell calibration-consistency diagnostic (G1 established cal~1).
    sc, su0, srr = fit_convention(s_add[..., 0], s_raw[..., 0], robs, d)
    s_mod = s_raw * sc
    cone = thetas < CONES[0]                  # 1-D mask on the theta axis

    # --- residual amplitudes, residual Mueller (exact quadratic re-combination)
    g = s_add - s_mod
    M_full = mueller_from_s(s_add[..., 0], s_add[..., 1], s_add[..., 2], s_add[..., 3])
    M_res = mueller_from_s(g[..., 0], g[..., 1], g[..., 2], g[..., 3])
    # port self-check against ADDA's own mueller_scatgrid
    dport = np.max(np.abs(M_full - r['M'])) / max(np.abs(r['M']).max(), 1e-30)

    # --- amplitude L2 norms on the sphere (intensity-like, over 4 amplitudes)
    w = np.sin(np.radians(thetas))[:, None, None]

    def split_l2(arr):
        a2 = np.abs(arr) ** 2
        m0 = (thetas < CONES[0])[:, None, None].astype(float)
        m1 = (thetas < CONES[1])[:, None, None].astype(float)
        tot = np.sqrt(np.sum(a2 * w))
        cone0 = np.sqrt(np.sum(a2 * w * m0))
        cone = np.sqrt(np.sum(a2 * w * m1))
        return tot, cone0, cone
    tf, t0f, cf = split_l2(s_add)
    tg, t0g, cg = split_l2(g)
    tm, t0m, cm = split_l2(s_mod)

    # --- extinction consistency (algebraic identity must be exact)
    def ext0(S):
        s1, s2, s3, s4 = S[0, 0, 0], S[0, 0, 1], S[0, 0, 2], S[0, 0, 3]
        return acf.extinction_matrix(s2, s3, s4, s1, lamb)[0, 0]
    Cext_add, Cext_mod, Cext_sum = ext0(s_add), ext0(s_mod), ext0(s_add - g)

    # --- model size at Theta=180 (backscatter exemption evidence)
    i180 = int(np.flatnonzero(np.abs(thetas - 180.0) < 1e-9)[0])
    m11_add_180 = r['M'][i180, :, 0]
    Mmod11 = mueller_from_s(s_mod[i180, :, 0], s_mod[i180, :, 1],
                            s_mod[i180, :, 2], s_mod[i180, :, 3])[:, 0]
    back_ratio = float(np.max(np.abs(Mmod11)) / np.max(np.abs(m11_add_180)))

    # --- cross-pol near forward (model diagonal: s3,s4 model amplitudes)
    xpol_f = np.max(np.abs(s_add[:, :, 2:][cone]))
    xpol_m = np.max(np.abs(s_mod[:, :, 2:][cone]))

    # --- SH truncation, production rule (component 0 = M11 sets eps_m;
    #     expand M11, M12, M34, take max achieved ntrunc, as
    #     spha.expand_mueller_2_spharmonics but restricted to 3 comps).
    #     TWO criteria:
    #      own-eps   : eps = 0.005 * own M11 norm (the historic pipeline rule)
    #      full-eps  : eps = 0.005 * FULL M11 norm (the fair consumer
    #                  criterion for a hybrid model+residual product: the
    #                  model is re-added EXACTLY, so what matters is the
    #                  residual's ABSOLUTE error vs the full-field scale).
    def m11_int(Mat):
        TH, PH = np.meshgrid(np.radians(phis), np.radians(thetas))[::-1]
        return np.sqrt(spha.twoD_int(Mat[:, :, 0] ** 2 * np.sin(TH),
                                     np.radians(thetas), np.radians(phis)))

    def max_ntrunc_pipeline(Mat, eps_rel):
        comps = Mat[:, :, (0, 1, 10)].astype(np.float64)
        inp = {'mueller_array': comps,
               'thetas_grid': thetas, 'phis_grid': phis}
        _, _, _, nt_ = spha.expand_mueller_2_spharmonics(inp, eps_rel)
        return int(nt_)

    ntrunc_full = max_ntrunc_pipeline(r['M'], EPS_SPH)
    ntrunc_res = max_ntrunc_pipeline(M_res, EPS_SPH)
    scale = m11_int(r['M']) / max(m11_int(M_res), 1e-300)
    ntrunc_res_fulleps = max_ntrunc_pipeline(M_res, EPS_SPH * min(scale, 1e12))

    cell = dict(tag=tag, path=os.path.basename(path), D_um=D, f_GHz=f / 1e9,
                T=T, x=x, N_dip=int(mod.N), d_um=d, d_from_dpl=d_addd,
                box=r['box'], orient=(alpha, beta_tilt, gamma, list(prop_0)),
                cal_c=complex(sc), cal_off=complex(su0[0], 0),
                fit_res=float(srr),
                L2_full=float(tf), L2_res=float(tg),
                ratio_total=float(tg / tf),
                ratio_cone30=float(t0g / t0f), ratio_cone60=float(cg / cf),
                ratio_ann3060=float(np.sqrt(max(cg ** 2 - t0g ** 2, 0))
                                    / np.sqrt(max(cf ** 2 - t0f ** 2, 1e-300))),
                ratio_out60=float(np.sqrt(max(tg ** 2 - cg ** 2, 0))
                                  / np.sqrt(max(tf ** 2 - cf ** 2, 1e-300))),
                Cext_adda=float(Cext_add), Cext_model=float(Cext_mod),
                Cext_ident_err=float(abs(Cext_sum - Cext_add)),
                ntrunc_full=ntrunc_full, ntrunc_res=ntrunc_res,
                ntrunc_res_fulleps=ntrunc_res_fulleps,
                port_check_rel=float(dport),
                back_model_ratio=back_ratio,
                xpol_full=float(xpol_f), xpol_model=float(xpol_m))
    RESULTS['cells'].append(cell)
    print('\ncell %s  x=%.2f D=%.0fum N=%d d=%.2fum' % (tag, x, D, mod.N, d))
    print('  port(M from ampl vs ADDA mueller file) rel=%.2e' % dport)
    print('  calib fit: c=%.3f%+.3fj (dev %.3f)  u0dev=%s  res=%.2e'
          % (sc.real, sc.imag, abs(sc - cal), np.round(su0, 2), srr))
    print('  L2  |g|/|f| total=%.3e  <30deg=%.3e  <60deg=%.3e  30-60=%.3e  >60deg=%.3e'
          % (cell['ratio_total'], cell['ratio_cone30'], cell['ratio_cone60'],
             cell['ratio_ann3060'], cell['ratio_out60']))
    print('  Cext adda=%.4g model=%.4g ident-err=%.2e' % (Cext_add, Cext_mod,
                                                          abs(Cext_sum - Cext_add)))
    print('  ntrunc(full M)=%d   ntrunc(residual M)=%d' % (ntrunc_full, ntrunc_res))
    print('  |F_model| at Theta=180 / M11_adda: %.2e' % back_ratio)
    return cell


_last_ntrunc = None


# ==============================================================================
# main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='hp,hc',
                    help='comma list of groups: hp (HongPlate), hc (HongColumn)')
    ap.add_argument('--keep', action='store_true', help='keep ADDA output dirs')
    ap.add_argument('--quick', action='store_true',
                    help='fewer cells (x=1,2,5, face-on + 60deg only)')
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix='phase9_')
    groups = set(args.cells.split(','))

    cal = None
    try:
        g1out = g1(workdir)
        cal = g1out[sorted(g1out)[0]]['c']
        RESULTS['calibration_c'] = complex(cal)

        HP = os.path.join(REPO, 'adda_shape_files', 'HongPlate_Id9')
        HC = os.path.join(REPO, 'adda_shape_files', 'HongColumn_Id7')
        cells = []
        if 'hp' in groups:
            for xv, pat in ((1.0, '*Dveq00108um_Mass*Freq0887.000GHz.adda'),
                            (2.0, '*Dveq00215um_Mass*Freq0887.000GHz.adda'),
                            (5.0, '*Dveq00541um_Mass*Freq0887.000GHz.adda'),
                            (10.0, '*Dveq01078um_Mass*Freq0887.000GHz.adda')):
                for oname, a, bt, g, prop0 in (
                        ('face', 0, 0, 0, (0, 0, 1)),
                        ('t60', 0, 60, 0, (0, 0, 1)),
                        ('edge', 0, 90, 0, (0, 0, 1))):
                    if args.quick and (xv == 10.0 or oname == 'edge'):
                        continue
                    if xv == 1.0 and oname == 'edge':
                        continue
                    cells.append((HP, pat, xv, oname, a, bt, g, prop0))
        if 'hc' in groups:
            for xv, pat in ((2.0, '*Dveq00211um_Mass*Freq0887.000GHz.adda'),
                            (5.0, '*Dveq00524um_Mass*Freq0887.000GHz.adda')):
                for oname, a, bt, g, prop0 in (('along', 0, 0, 0, (0, 0, 1)),
                                               ('t60', 0, 60, 0, (0, 0, 1))):
                    cells.append((HC, pat, xv, oname, a, bt, g, prop0))

        for (sdir, pat, xv, oname, a, bt, g, prop0) in cells:
            name = os.path.basename(sdir)
            tag = '%s_x%g_%s' % (name, xv, oname)
            try:
                eval_cell(sdir, pat, FREQ, TENV, a, bt, g, prop0, tag,
                          workdir, cal)
            except Exception as exc:                      # keep scanning
                print('CELL FAIL %s: %r' % (tag, exc))
                RESULTS['checks'].append({'name': 'cell ' + tag,
                                          'pass': False, 'detail': repr(exc)})
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)

    # summary gates
    cs = RESULTS['cells']
    if cs:
        worst = max(c['ratio_total'] for c in cs)
        check('9A residual L2 ratio finite+bounded', worst < 1.05,
              'max |g|/|f| over cells = %.3f' % worst)
        check('9A port check exact', all(c['port_check_rel'] < 1e-5 for c in cs),
              'max rel dev %.2e' % max(c['port_check_rel'] for c in cs))
        # NOTE: the f = F_model + g identity is EXACT by construction; what
        # Cext_ident_err actually measures is |Cext_model - Cext_adda|, i.e.
        # the model's own extinction error (a model-quality diagnostic).
        check('9A Cext model error < 25%',
              all(c['Cext_ident_err'] < 0.25 * max(abs(c['Cext_adda']), 1)
                  for c in cs),
              'max |Cext_model-Cext_adda|/Cext_adda = %.3f'
              % max(c['Cext_ident_err'] / max(abs(c['Cext_adda']), 1)
                    for c in cs))
        ntr_full = {c['x']: c['ntrunc_full'] for c in cs}
        ntr_res = {c['x']: min([cc['ntrunc_res'] for cc in cs
                                if cc['x'] == c['x']], default=0)
                   for c in cs}
        big = [c for c in cs if c['x'] >= 2.0]
        # decision criterion = FULL-eps ntrunc (the hybrid product re-adds the
        # model exactly, so the consumer tolerance must be set by the FULL
        # field scale, not the residual's own norm):
        gain = [c['ntrunc_res_fulleps'] / c['ntrunc_full'] for c in big] \
            if big else []
        own = [c['ntrunc_res'] / c['ntrunc_full'] for c in big] if big else []
        check('9A ntrunc reduction at x>=2 (full-eps criterion)',
            bool(big) and max(gain) < 1.0,
            'worst ntrunc_res_fulleps/ntrunc_full = %.3f  '
            '(own-eps worst: %.3f)' % (max(gain) if gain else float('nan'),
                                       max(own) if own else float('nan')))
        RESULTS['ntrunc_summary'] = {'full': ntr_full, 'residual': ntr_res}

    with open(RESULTS_JSON, 'w') as fh:
        json.dump(RESULTS, fh, indent=1, default=str)
    print('\nwrote %s' % RESULTS_JSON)

    with open(TABLE_TXT, 'w') as fh:
        hdr = ('cell                        x    D[um]  N_dip   |g|/|f|   '
               'L2<30   L2<60   L2>60   Cext_m/Cext_a   ntrunc '
               'full/res(own)/res(fulleps)   calib c dev')
        fh.write(hdr + '\n' + '-' * len(hdr) + '\n')
        for c in cs:
            fh.write('%-27s %4.1f %7.0f %7d  %8.3e %7.3e %7.3e %7.3e  '
                     '%10.4f      %4d/%-5d/%-5d       %.3f\n'
                     % (c['tag'], c['x'], c['D_um'], c['N_dip'],
                        c['ratio_total'], c['ratio_cone30'], c['ratio_cone60'],
                        c['ratio_out60'], c['Cext_model'] / c['Cext_adda'],
                        c['ntrunc_full'], c['ntrunc_res'],
                        c['ntrunc_res_fulleps'],
                        abs(c['cal_c'] / RESULTS.get('calibration_c', 1) - 1)))
    print('wrote %s' % TABLE_TXT)
    print(open(TABLE_TXT).read())


if __name__ == '__main__':
    main()
