#!/usr/bin/env python3
"""
Publication-quality figures for the baseline report (baselines/REPORT.md).

Reads the frozen ADDA v1.2 baseline outputs in baselines/adda_1.2/ and writes
four figures to baselines/report/ (PNG + PDF):

  fig1_crosssections.png     cross sections & single-scattering albedo, X vs Y pol
  fig2_sphere_mie.png        sphere case vs independent Mie theory + dpl convergence
  fig3_scattering_maps.png   S11 3D radiation surfaces (all 6 cases)
  fig4_ellipsoid_phase.png   spheroid 3D radiation surfaces + pol. discrimination

Regenerate:
  mamba activate DDA
  python baselines/plot_baseline_figs.py
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from scipy.interpolate import RectBivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "adda_1.2")
OUT = os.path.join(HERE, "report")

# ---------------------------------------------------------------- style ----
C = {
    "polx": "#2b6cb0",   # X incidence polarization (solid blue)
    "poly": "#c05621",   # Y incidence polarization (hatched orange)
    "x01": "#2c7a7b",
    "x1":  "#c05621",
    "x5":  "#804000",
    "mie": "#c53030",
    "gray": "#4a5568",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#4a5568",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#cbd5e0",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

MU = "\u00b5"       # µ
DEG = "\u00b0"      # °
EMD = "\u2014"     # —
OMG = "\u03c9"     # ω
ARR = "\u2190"     # ←
SUB = "\u2091\u1d63"  # ₑᵣ

# ------------------------------------------------------------ data io ----
def read_crosssec(path):
    out = {}
    with open(path) as f:
        for line in f:
            mm = re.match(r"(\w+)\s*=\s*([-+\d.eE]+)", line)
            if mm:
                out[mm.group(1)] = float(mm.group(2))
    return out


def read_mueller(case):
    """Return S-matrix grid: theta[Nt], phi[Np], dict of Sij arrays [Nt,Np]."""
    arr = np.loadtxt(os.path.join(BASE, case, "mueller_scatgrid"), skiprows=1)
    th = np.unique(arr[:, 0])
    ph = np.unique(arr[:, 1])
    cols = arr[:, 2:].reshape(len(th), len(ph), 16)
    names = [f"s{r}{c}" for r in range(1, 5) for c in range(1, 5)]
    return th, ph, dict(zip(names, cols.transpose(2, 0, 1)))


CASES = [
    # (dir name, group label, bar label, size parameter)
    ("sphere_ice_x0.31_f300GHz_T230",      "ice sphere\n300 GHz",       "x=0.31",  0.31),
    ("ellipsoid_oblate_x0.1_f183GHz_T253", "oblate spheroid\n183 GHz",  "x=0.1",   0.1),
    ("ellipsoid_oblate_x1_f183GHz_T253",   "oblate spheroid\n183 GHz",  "x=1",     1.0),
    ("ellipsoid_oblate_x5_f183GHz_T253",   "oblate spheroid\n183 GHz",  "x=5",     5.0),
    ("hongplate_id9_dveq10_f192GHz_T230",  "Hong plate ID9\nD" + SUB + "q=10 " + MU + "m", "192 GHz", 0.020),
    ("hongplate_id9_dveq10_f337GHz_T230",  "Hong plate ID9\nD" + SUB + "q=10 " + MU + "m", "337 GHz", 0.035),
]

CS = {case: {pol: read_crosssec(os.path.join(BASE, case, f"CrossSec-{pol}"))
             for pol in "XY"} for case, *_ in CASES}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=250)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    print("wrote", name)


# ---------------------------------------------------- fig 1: crosssecs ----
def fig_crosssections():
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7),
                             gridspec_kw={"wspace": 0.40})
    n = len(CASES)
    idx = np.arange(n)
    w = 0.38
    panels = [("Cext", r"Extinction $C_{\mathrm{ext}}$ [" + MU + "m$^2$]", 2e-3),
              ("Cabs", r"Absorption $C_{\mathrm{abs}}$ [" + MU + "m$^2$]", 2e-3),
              (r"omega", r"Single-scattering albedo $\omega\,=\,C_{\mathrm{sca}}/C_{\mathrm{ext}}$", 3e-4)]
    for ax, (key, ylab, ymin) in zip(axes, panels):
        for pol, off, col, yoff in (("X", -w / 2, C["polx"], 2),
                                    ("Y", +w / 2, C["poly"], 12)):
            if key == "omega":
                vals = [CS[c][pol]["Csca"] / CS[c][pol]["Cext"] for c, *_ in CASES]
            else:
                vals = [CS[c][pol][key] for c, *_ in CASES]
            bars = ax.bar(idx + off, vals, w, color=col,
                          hatch="//" if pol == "Y" else None,
                          edgecolor=col, linewidth=0.5)
            if key == "omega":
                for b, v in zip(bars, vals):
                    ax.annotate(f"{v:.1e}", (b.get_x() + b.get_width() / 2, v),
                                xytext=(0, yoff), textcoords="offset points",
                                ha="center", fontsize=6, color=col)
        ax.set_yscale("log")
        ax.set_ylim(bottom=ymin)
        ax.set_xticks(idx, ["x=0.31", "x=0.1", "x=1", "x=5", "192", "337"], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1e}"))
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    # case group headers over the first panel
    for x0, x1, label in ((-0.5, 0.5, "ice sphere\n300 GHz"),
                          (0.5, 3.5, CASES[1][1]),
                          (3.5, 5.5, "Hong plate ID9, 10 " + MU + "m\nf [GHz]")):
        axes[0].annotate(label, ((x0 + x1) / 2, 1.02), xycoords=("data", "axes fraction"),
                         xytext=(0, 12), textcoords="offset points",
                         ha="center", fontsize=7.8, color=C["gray"])
        axes[0].plot([x0 + .07, x1 - .07], [1.02, 1.02],
                     transform=axes[0].get_xaxis_transform(), lw=1.6, color="#a0aec0")
    h = [plt.Rectangle((0, 0), 1, 1, fc=C["polx"], ec=C["polx"]),
         plt.Rectangle((0, 0), 1, 1, fc=C["poly"], ec=C["poly"], hatch="//")]
    fig.legend(h, ["incidence polarization X (perpendicular)",
                   "incidence polarization Y (parallel, hatched)"],
               loc="lower center", ncol=2, handlelength=1.4,
               bbox_to_anchor=(0.5, -0.06))
    axes[2].annotate(
        "x=5: " + OMG + ">1 (unphysical)\n" + ARR + " coarse-grid artifact, see report",
        xy=(3.2, 1.57 * 1.12), xycoords="data", xytext=(2.15, 9),
        fontsize=7, color=C["mie"],
        arrowprops=dict(arrowstyle="->", color=C["mie"], lw=0.9))
    fig.suptitle("Frozen regression baselines " + EMD + " ADDA v1.2 (macOS arm64), "
                 "Matzler-2006 ice, orient 0 0 0, incidence 90" + DEG,
                 fontsize=11.5, fontweight="bold", y=1.16)
    save(fig, "fig1_crosssections")


# ------------------------------------------------------ fig 2: sphere-Mie ----
def fig_sphere_mie():
    # data: baselines/adda_1.2 sphere case (auto grid, 160.7 dpl) + documented
    # Phase-0 convergence reruns (-dpl 320/800) + Mie value (check_baselines.py)
    dpl = [160.7, 320, 800]
    qext = [6.914445e-3, 6.852e-3, 6.819e-3]
    mie = 6.792946e-3

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5),
                                 gridspec_kw={"wspace": 0.30})
    a1.axhline(mie, color=C["mie"], lw=1.4, ls="--",
               label=rf"Mie theory   $Q_{{\mathrm{{ext}}}}$ = {mie:.4g}")
    a1.plot(dpl, qext, "o-", color=C["polx"], lw=1.6, ms=7,
            label="ADDA 1.2 (auto / forced dpl, LDR)")
    for x, q in zip(dpl, qext):
        a1.annotate(f"{q:.4g} ({(q/mie-1)*100:+.2f}%)", (x, q),
                    xytext=(2, 9), textcoords="offset points",
                    fontsize=7.5, color=C["polx"])
    a1.set_xscale("log")
    a1.set_xticks(dpl)
    a1.set_xticklabels(["160.7\n(auto grid)", "320", "800"])
    a1.xaxis.set_minor_formatter(mticker.NullFormatter())
    a1.set_xlim(125, 1500)
    a1.set_ylim(6.75e-3, 7.01e-3)
    a1.set_xlabel("dipoles per wavelength")
    a1.set_ylabel(r"$Q_{\mathrm{ext}}$")
    a1.set_title("Ice sphere, x=0.31, 300 GHz, 230 K\nconvergence of DDA toward Mie theory")
    a1.legend(loc="lower right")
    a1.spines[["top", "right"]].set_visible(False)

    keys = ["Qext", "Qsca", "Qabs"]
    csx = CS["sphere_ice_x0.31_f300GHz_T230"]["X"]
    adda = [csx["Qext"], csx["Qsca"], csx["Qext"] - csx["Qsca"]]
    mievals = [6.792946e-3, 4.653942e-3, 2.139004e-3]
    dev = [abs(a / m - 1) * 100 for a, m in zip(adda, mievals)]
    bars = a2.bar(np.arange(3), dev, 0.55,
                  color=[C["polx"], C["x01"], C["poly"]])
    for b, v in zip(bars, dev):
        a2.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5)
    a2.axhline(3.0, color=C["mie"], ls="--", lw=1.3)
    a2.annotate("check_baselines tolerance 3 %", (2.45, 3.0), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=7.5, color=C["mie"])
    a2.set_xticks(np.arange(3), keys)
    a2.set_xlim(-0.6, 2.6)
    a2.set_ylim(0, 3.6)
    a2.set_ylabel("deviation from Mie [%]")
    a2.set_title("Deviation vs independent Mie theory\n(Riccati-Bessel, miepython-verified)")
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Analytic validation of the sphere baseline",
                 fontsize=11.5, fontweight="bold", y=1.07)
    save(fig, "fig2_sphere_mie")


# ------------------------------------- 3D radiation-surface helpers ----
LOGR = 4.0  # decades of S11 dynamic range mapped to the radial extent

def s11_fine(case, nth=61, nph=121):
    """Bicubic-spline interpolate the 10 deg S11 grid -> fine normalized grid."""
    th, ph, S = read_mueller(case)
    spl = RectBivariateSpline(th, ph, S["s11"])
    thf = np.linspace(0.0, 180.0, nth)
    phf = np.linspace(0.0, 360.0, nph, endpoint=False)
    s = spl(thf, phf)
    return thf, phf, s / s.max()

def radiation_surface(ax, thf, phf, ratio, norm, cmap):
    """Mie-style 3D radiation surface: radius = intensity^(1/2.5) (amplitude-like,
    classic radiation-diagram scaling), color = intensity (log), Lambert shading;
    beam axis = polar (z) axis, forward = +z."""
    r = np.clip(ratio ** 0.4, 0.03, 1.0)
    th, ph = np.deg2rad(thf), np.deg2rad(phf)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    X = r * np.sin(TH) * np.cos(PH)
    Y = r * np.sin(TH) * np.sin(PH)
    Z = r * np.cos(TH)
    ax.plot_surface(X, Y, Z, facecolors=cmap(norm(ratio)), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=True)
    # equatorial reference ring (unit sphere = forward-scattering level)
    a = np.linspace(0, 2 * np.pi, 140)
    ax.plot(np.cos(a), np.sin(a), np.zeros_like(a), color="#a0aec0", lw=0.9, ls=":")
    # incident-beam arrow entering from below toward +z (forward = +z)
    ax.quiver(0, 0, -1.40, 0, 0, 1, color=C["mie"], lw=1.6, arrow_length_ratio=0.16)
    ax.text(0, 0, -1.50, "incident beam", color=C["mie"], fontsize=7,
            ha="center", va="top")
    ax.text(0, 0, 1.10, "forward", color=C["gray"], fontsize=7,
            ha="center", va="bottom", style="italic")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1.55, 1.18)
    ax.set_box_aspect((1, 1, 1.4))
    ax.set_axis_off()
    ax.view_init(elev=11, azim=-56)

# ------------------------------------------------ fig 3: 3D radiation surfaces ----
def fig_scatter_maps():
    norm = LogNorm(vmin=10 ** -LOGR, vmax=1)
    cmap = matplotlib.colormaps["cividis"]
    fig = plt.figure(figsize=(10.8, 8.2))
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.01, top=0.94)
    gs = fig.add_gridspec(2, 3, wspace=0.02, hspace=0.14,
                          width_ratios=[1, 1, 1])
    for i, (case, grp, lbl, x) in enumerate(CASES):
        ax = fig.add_subplot(gs[i // 3, i % 3], projection="3d")
        thf, phf, ratio = s11_fine(case)
        radiation_surface(ax, thf, phf, ratio, norm, cmap)
        ax.set_title(f"{grp.splitlines()[0]}   {lbl}", fontsize=9.5, pad=12)
    cax = fig.add_axes([0.915, 0.16, 0.017, 0.66])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cax, label=r"$S_{11}\,/S_{11}^{\max}$")
    fig.suptitle("3D radiation surfaces of the scattering-grid $S_{11}$ " + EMD +
                 " radial extent $\\propto S_{11}^{0.4}$ (radiation-diagram scaling), "
                 "color = " + r"$S_{11}/S_{11}^{\max}$ " + EMD + " interpolated from the 10" + DEG +
                 " output grid",
                 fontsize=11, fontweight="bold", y=0.98)
    save(fig, "fig3_scattering_maps")


# --------------------------------- fig 4: 3D ellipsoid + pol. discrimination ----
def fig_ellipsoid_phase():
    norm = LogNorm(vmin=10 ** -LOGR, vmax=1)
    cmap = matplotlib.colormaps["cividis"]
    ell = [c for c, *_ in CASES if "ellipsoid" in c]
    cols = [C["x01"], C["x1"], C["x5"]]
    fig = plt.figure(figsize=(10.8, 8.4))
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.92)
    gs = fig.add_gridspec(2, 2, wspace=0.10, hspace=0.24)
    for k, case in enumerate(ell):
        ax = fig.add_subplot(gs[k // 2, k % 2], projection="3d")
        thf, phf, ratio = s11_fine(case)
        radiation_surface(ax, thf, phf, ratio, norm, cmap)
        xlab = re.search(r"x([\d.]+)_", case).group(1)
        ax.set_title("oblate spheroid,  x = " + xlab, fontsize=9.5, pad=12)
    a2 = fig.add_subplot(gs[1, 1])
    for case, col in zip(ell, cols):
        theta, phi, S = read_mueller(case)
        dop = np.abs(S["s12"])[:, 0] / S["s11"][:, 0] * 100
        a2.plot(theta, dop, lw=1.7, color=col)
    a2.legend(["x = 0.1", "x = 1", "x = 5"], loc="upper left")
    a2.set_xlabel("scattering angle [" + DEG + "]")
    a2.set_ylabel("$|S_{12}|/S_{11}$  [%]")
    a2.set_title("Polarization discrimination\nincident X, plane \u03c6=0" + DEG,
                 fontsize=9.5)
    a2.set_xticks(np.arange(0, 181, 30))
    a2.spines[["top", "right"]].set_visible(False)
    cax = fig.add_axes([0.915, 0.16, 0.017, 0.66])
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cax=cax, label=r"$S_{11}\,/S_{11}^{\max}$")
    fig.suptitle("Rayleigh " + EMD + " resonance " + EMD + " forward peak: 3D radiation surfaces\n"
                 "analytic oblate spheroid c/a=0.5, 183 GHz, 253 K, X incidence "
                 + EMD + " each surface normalized to its own max, identical view",
                 fontsize=11, fontweight="bold", y=1.00)
    save(fig, "fig4_ellipsoid_phase")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_crosssections()
    fig_sphere_mie()
    fig_scatter_maps()
    fig_ellipsoid_phase()
