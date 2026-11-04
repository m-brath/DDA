#!/usr/bin/env python3
"""
Phase-1 inventory: habit x size x frequency x temperature status matrix.

Reads each active adda_config_*.py (textually — configs cannot be imported
locally), scans the corresponding adda_shape_files/ folder for .adda shapes
with matching .meta.mat metadata, and evaluates which (size, frequency)
combinations are active under the config's size_parameter_limit and the
shape's frequency_min.

Output: markdown table printed to stdout (embedded into the modernization
report). Run with the DDA env python.
"""
import glob
import os
import re

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "adda_python_scripts")
SHAPES = os.path.join(REPO, "adda_shape_files")
C0 = 299792458.0

DONE = {  # per agents.md Current Status
    "HexPlaAggCrystal_Id20": "ARO done",
    "HongPlate_Id9": "ARO done",
    "HongPlate_Id9_TOR_sizegrid": "TRO pending",
    "HongColumn_Id7": "queued",
    "IconSnow_Id28": "queued",
    "LiuSectorSnowflake_Id3": "queued",
}


def parse_config(path):
    txt = open(path, errors="replace").read()

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, txt, re.M)
        if not m:
            return default
        vals = [cast(v) for v in re.findall(r"[-+\d.eE]+", m.group(1))]
        scale = re.match(r"\)?\s*\*\s*([\d.eE+]+)", txt[m.end():m.end() + 40])
        if scale:
            vals = [v * float(scale.group(1)) for v in vals]
        return vals

    freqs = grab(r"^frequencies=np\.array\(\[([^\]]+)\]", default=[])
    temps = grab(r"^temperatures=np\.array\(\[([^\]]+)\]", default=[])
    m = re.search(r"^size_parameter_limit\s*=\s*([\d.]+)", txt, re.M)
    xlim = float(m.group(1)) if m else 10.0
    m = re.search(r"^shape_mainpath.*", txt)
    # shape folder: search folder= style strings
    folder = None
    m = re.search(r"^shape_name\s*=\s*['\"]([^'\"]+)['\"]", txt, re.M)
    if m and os.path.isdir(os.path.join(SHAPES, m.group(1))):
        folder = m.group(1)
    else:
        for mm in re.finditer(r"['\"]([^'\"]*/)?([A-Za-z0-9_]+)/['\"]", txt):
            cand = mm.group(2)
            if os.path.isdir(os.path.join(SHAPES, cand)):
                folder = cand
                break
    return dict(freqs=freqs, temps=temps, xlim=xlim, folder=folder)


def main():
    print("| Config | Habit folder | status | shapes | T | GHz freqs | size x combos active (of total) | skipped by fmin | skipped by x>limit | fmin classes (GHz) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    tot_all = 0
    folder_fmins = {}
    for cfg in sorted(glob.glob(os.path.join(SCRIPTS, "adda_config_*.py"))):
        name = os.path.basename(cfg)[len("adda_config_"):-len(".py")]
        c = parse_config(cfg)
        if not c["folder"]:
            print(f"| {name} | (analytic/no folder) | - | 0 | {len(c['temps'])} | {len(c['freqs'])} | - | - | - |")
            continue
        folder = os.path.join(SHAPES, c["folder"])
        shapes = sorted(glob.glob(os.path.join(folder, "*.adda")))
        n_active = n_tot = n_fmin = n_xlim = 0
        ndip_range = [None, None]
        for s in shapes:
            meta_path = s[:-len(".adda")] + ".meta.mat"
            if not os.path.isfile(meta_path):
                continue
            try:
                rec = sio.loadmat(meta_path)["meta"].ravel()[0]
                dveq = float(np.asarray(rec["diameter_vol_eq"]).ravel()[0])  # m
                fmin = float(np.asarray(rec["frequency_min"]).ravel()[0])    # Hz
                ndip = float(np.asarray(rec["N_dipoles"]).ravel()[0])
            except Exception as e:
                print(f"| !! {os.path.basename(s)} meta parse error: {e} |")
                continue
            ndip_range[0] = ndip if ndip_range[0] is None else min(ndip_range[0], ndip)
            ndip_range[1] = ndip if ndip_range[1] is None else max(ndip_range[1], ndip)
            folder_fmins.setdefault(c["folder"], set()).add(fmin)
            for f in c["freqs"]:
                n_tot += 1
                x = np.pi * dveq * f / C0  # size parameter (dveq in m)
                if f < fmin - 1e-9:
                    n_fmin += 1
                elif x > c["xlim"]:
                    n_xlim += 1
                else:
                    n_active += 1
        tot_all += n_active * len(c["temps"])
        nrange = f"{int(ndip_range[0])}..{int(ndip_range[1])}" if ndip_range[0] is not None else "-"
        fmins = sorted({int(x) for x in folder_fmins.get(c["folder"], [])})
        fminstr = ",".join(f"{x//1e9:g}" for x in fmins) if fmins else "-"
        print(f"| {name} | {c['folder']} | {DONE.get(c['folder'],'pending')} | {len(shapes)} "
              f"(N_dip {nrange}) | {len(c['temps'])} | {len(c['freqs'])} "
              f"| {n_active} (of {n_tot}) | {n_fmin} | {n_xlim} | {fminstr} |")
    print(f"\nTotal active (size,f,T) ADDA-job count if all pending habits processed: {tot_all}")


if __name__ == "__main__":
    main()
