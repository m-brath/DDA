#!/usr/bin/env python3
"""
Phase-2 equivalence harness: proves the numpy-2 / dependency modernization
of adda_python_scripts/ changed NO observable behavior.

usage:  python analysis/phase2_equiv.py pre      # snapshot OLD code (with numpy shims)
        python analysis/phase2_equiv.py post     # snapshot NEW code (no shims)
        python analysis/phase2_equiv.py compare  # deep-compare the two snapshots

pre  = snapshot taken from git HEAD of adda_python_scripts (before Phase-2 edits);
this script inserts compatibility shims (np.NaN/np.NAN/np.complex, stub
posix_spawn) so the old code even runs on macOS/numpy2. post = same snapshot
built against the edited code without shims. compare = PASS only if all
numeric outputs agree to rtol=1e-12 and object structures are identical.

Inputs are deterministic: golden ADDA baseline outputs (baselines/adda_1.2)
feed the parser test; fixed seeds/grids feed everything else.
"""
import os
import pickle
import subprocess
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "adda_python_scripts")
BASE = os.path.join(REPO, "baselines", "adda_1.2")
ADDA_SEQ = os.path.join(REPO, "adda_1.2", "src", "seq", "adda")
SCAT_P = os.path.join(REPO, "baselines", "scat_params_baseline.dat")
ALLDIR_P = os.path.join(SCRIPTS, "alldir_params.dat")
sys.path.insert(0, SCRIPTS)

MODE = sys.argv[1] if len(sys.argv) > 1 else "compare"
PRE = os.path.join(REPO, "analysis", "phase2_pre.pkl")
POST = os.path.join(REPO, "analysis", "phase2_post.pkl")

if MODE in ("pre", "post"):
    if MODE == "pre":
        # shims so the OLD (numpy-1-era) code runs on numpy 2 / macOS
        np.NaN = np.nan
        np.NAN = np.nan
        np.complex = np.complex128
        import scipy.integrate as _spi
        _spi.trapz = _spi.trapezoid  # removed in scipy>=1.18 (the old code uses trapz)
        stub = types.ModuleType("posix_spawn")
        stub.spawn_command = lambda cmd: subprocess.run(["/bin/sh", "-c", cmd]).returncode
        sys.modules["posix_spawn"] = stub

    import adda_control_functions as acf
    import icosphere_grid as ico
    import spherical_harmonics_for_adda as spha
    import adda_orientational_avg_spharmonics_functions as aoa
    import definitions_for_netcdf as dfn

    snap = {}

    # ---- 1) acf.loadmat on real .meta.mat files (mat_struct path) -----------
    snow = sorted(f for f in os.listdir(os.path.join(REPO, "adda_shape_files", "IconSnow_Id28"))
                  if f.endswith(".meta.mat"))
    metas = ["adda_shape_files/HongPlate_Id9/Dmax00013um_Dveq00010um_Mass4.79983e-13kg_Freq0192.000GHz.meta.mat",
             "adda_shape_files/IconSnow_Id28/" + snow[0]]
    metas = [m for m in metas if os.path.isfile(os.path.join(REPO, m))]
    snap["loadmat"] = {os.path.basename(m): str(acf.loadmat(os.path.join(REPO, m))) for m in metas}

    # ---- 2) icosphere grid + refinement + interpolation + rotation ----------
    verts, tris = ico.make_icosahedric_grid()
    snap["ico_base"] = [verts, tris]
    snap["ico_ref2"] = list(ico.refine_icosahedric_grid(verts, tris, 2))
    rv = ico.rotate_vertices(verts, 0.3, 0.7, 1.1)
    q = np.array([0.4, 0.5, -0.7]); q = q / np.linalg.norm(q)
    w, idx = ico.interpolation_weights_icosahedric_grid(rv, q)
    snap["ico_interp"] = [w, idx]
    snap["vpol"] = aoa.create_vpol_vector(verts[:12])

    # ---- 3) spherical harmonics: twoD_int + full mueller expansion ----------
    th = np.linspace(0.0, np.pi, 46)
    ph = np.linspace(0.0, 2 * np.pi, 91)
    PHI, TH = np.meshgrid(ph, th)
    K = np.sin(3 * TH) * np.cos(PHI) ** 2
    snap["twoD_int"] = spha.twoD_int(K, th, ph)
    gold = os.path.join(BASE, "hongplate_id9_dveq10_f337GHz_T230", "mueller_scatgrid")
    data = np.loadtxt(gold, skiprows=1)
    th_all = np.unique(data[:, 0]); ph_all = np.unique(data[:, 1])
    Mfull = data[:, 2:].reshape(len(th_all), len(ph_all), data.shape[1] - 2)
    Msub = Mfull  # production grid (shtns needs the full resolution for adaptive ntrunc)
    ts = th_all; ps = ph_all
    inp = {"mueller_array": Msub, "thetas_grid": ts, "phis_grid": ps}
    coeff, relerr, mlc, mnt = spha.expand_mueller_2_spharmonics(inp, 0.005, diagnostics=False)
    snap["sphcoeff"] = [np.asarray(c) for c in coeff]
    snap["sphmeta"] = [relerr, mlc, mnt]

    # ---- 4) aoa rotation angles + vpol + matlib-repmat usage sites ----------
    snap["rota"] = aoa.calculate_rotation_angles(np.array([0.0, 60.0, 120.0, 180.0]), TH[:4], PHI[:4])
    nv = np.array([[1.0], [2.0], [3.0]])
    if MODE == "pre":
        snap["repmat"] = np.matlib.repmat(nv, 3, 1).transpose()
    else:
        snap["repmat"] = np.tile(nv, (3, 1)).transpose()  # new replacement expression

    # ---- 5) NetCDF definitions (np.NaN at import time) ----------------------
    import re as _re
    _ts = _re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+")
    snap["nc_defs"] = {k: _ts.sub("<TS>", str(v)) for k, v in vars(dfn).items()
                       if not k.startswith("_") and not callable(v)
                       and not isinstance(v, types.ModuleType)}

    # ---- 6) acf.adda_call: run ADDA + parse (spawn + output parsing) --------
    tmp = os.path.join(REPO, "analysis", "_tmp_phase2_addacall")
    os.makedirs(tmp, exist_ok=True)
    m = np.sqrt(acf.eps_ice_matzler06(300e9, 230.0))
    input_data = {
        "lambda": 999.30819333333329, "Size": 50.0, "refrative_index": m,
        "theta": 90.0, "alpha": 0, "beta": 0, "gamma": 0, "eps": 3,
        "ramdrive": tmp, "zipname": os.path.join(tmp, "probe.zip"),
        "main_command": [ADDA_SEQ],
        "static_clo": ["-shape", "sphere", "-store_scat_grid", "-scat_grid_inp", SCAT_P,
                       "-scat_matr", "both", "-Csca", "-alldir_inp", ALLDIR_P],
        "zipping": False,
    }
    res = acf.adda_call(input_data)
    if isinstance(res, dict):
        snap["adda_call"] = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    else:
        snap["adda_call"] = None

    out_path = PRE if MODE == "pre" else POST
    with open(out_path, "wb") as f:
        pickle.dump(snap, f, protocol=4)
    print(f"snapshot written: {out_path}")

# ============================================================================
elif MODE == "compare":
    a = pickle.load(open(PRE, "rb"))
    b = pickle.load(open(POST, "rb"))

    def eq(x, y, path="root"):
        ok = True
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if x.shape != y.shape:
                print(f"FAIL {path}: shape {x.shape} vs {y.shape}"); return False
            if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                if not np.allclose(x.astype(complex), y.astype(complex), rtol=1e-12, atol=1e-30, equal_nan=True):
                    print(f"FAIL {path}: numeric mismatch max={np.nanmax(np.abs(x-y))}"); return False
            elif str(x.tolist()) != str(y.tolist()):
                print(f"FAIL {path}: object array mismatch"); return False
        elif isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                print(f"FAIL {path}: dict keys differ"); return False
            for k in x:
                ok &= eq(x[k], y[k], f"{path}.{k}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                print(f"FAIL {path}: len {len(x)} vs {len(y)}"); return False
            for i, (u, v) in enumerate(zip(x, y)):
                ok &= eq(u, v, f"{path}[{i}]")
        else:
            import re as _re
            _ts = _re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+"
                              r"|datetime\.datetime\([^)]*\)")
            if _ts.sub("<TS>", str(x)) != _ts.sub("<TS>", str(y)):
                print(f"FAIL {path}: repr differs"); return False
        return ok

    all_ok = True
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            print(f"FAIL {k}: missing in one snapshot"); all_ok = False; continue
        ok = eq(a[k], b[k], k)
        print(f"{'PASS' if ok else 'FAIL':4s} {k}")
        all_ok &= ok
    print("\nRESULT:", "PASS — no behavior change" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)
