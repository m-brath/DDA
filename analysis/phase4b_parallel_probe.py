#!/usr/bin/env python3
"""Phase-4b step 19: parallelism probe for the orientation solves of one
(T,f,size) job. Each icosphere-vertex solve is an INDEPENDENT ADDA process
(particle at alpha=beta=gamma=0, incidence = vertex unit vector, production
CLO: -shape read + -store_scat_grid/-scat_grid_inp/-scat_matr both, eps 3) —
perfectly parallel at process level, but today sub_control runs them
sequentially inside one job.

Probe: take the HongPlate_Id9 10 µm probe shape (as committed phase4_hp_*
gate), icosphere level-3 vertex set (642 = production first refinement step
after 12/42/162 warmup), and time a representative subset (a) sequentially,
(b) in a process pool of N. Pure ADDA subprocess time (no zip/parse noise —
the same noise applies to both layouts).

Implication measured here feeds the HPC job-structure recommendation:
one Slurm job per (T,f,size) running 642+ sequential solves wastes the core
allocation; the solves are embarrassingly parallel (job arrays / pools).
"""
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "adda_python_scripts"))
import adda_control_functions as acf  # noqa: E402
import icosphere_grid as ico  # noqa: E402

SHAPE = glob.glob(os.path.join(
    REPO, "adda_shape_files/HongPlate_Id9/"
    "Dmax00013um_Dveq00010um_*Freq0192.000GHz.adda"))[0]
META = sio.loadmat(SHAPE.replace(".adda", ".meta.mat"))["meta"].ravel()[0]
DEQ_UM = float(np.ravel(META["diameter_vol_eq"])[0]) * 1e6  # -> µm
F = 228e9            # phase4_hp probe frequency
T = 190.0            # phase4_hp probe temperature
LAMB = 299792458.0 / F * 1e6
M = np.sqrt(np.complex128(acf.eps_ice_matzler06(F, T)))
EXE = os.path.join(REPO, "adda_1.4", "src", "seq", "adda")
SCATP = os.path.join(REPO, "adda_python_scripts", "scat_params.dat")
EPS = 3
TMP = os.path.join(os.environ.get("TMPDIR", "/tmp"), "phase4b_par")


def solve(v):
    """one production-equivalent ADDA solve at vertex direction v"""
    outdir = os.path.join(TMP, "v_%03d_%03d_%03d"
                          % tuple(np.round((v + 1) * 500).astype(int)))
    subprocess.run(["/bin/rm", "-rf", outdir])
    cmd = [EXE, "-shape", "read", SHAPE,
           "-store_scat_grid", "-scat_grid_inp", SCATP,
           "-scat_matr", "both",
           "-m", "%.12g" % M.real, "%.12g" % M.imag,
           "-lambda", "%.12g" % LAMB, "-eq_rad", "%.12g" % (DEQ_UM / 2.0),
           "-orient", "0", "0", "0",
           "-prop", "%.12g" % v[0], "%.12g" % v[1], "%.12g" % v[2],
           "-eps", str(EPS), "-dir", outdir]
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1")  # seq build: pin 1 core/solve
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError("ADDA failed (%d): %s" % (p.returncode,
                                                     p.stdout[-1500:]))
    subprocess.run(["/bin/rm", "-rf", outdir])
    return dt


def main(nsubset=60, nproc=None):
    os.makedirs(TMP, exist_ok=True)
    vertices, tri = ico.make_icosahedric_grid()
    for _ in range(3):  # 12 -> 42 -> 162 -> 642 (probe's production level)
        vertices, tri = ico.refine_icosahedric_grid(vertices, tri, 1)
    print("icosphere level-3 vertices:", len(vertices))
    # deterministic representative subset: Fibonacci-like index spread over
    # the (unsorted) vertex list
    idx = np.linspace(0, len(vertices) - 1, nsubset).astype(int)
    tasks = [vertices[i] for i in idx]

    t0 = time.time()
    seq = [solve(v) for v in tasks]
    t_seq = time.time() - t0

    nproc = nproc or min(8, os.cpu_count(), nsubset)
    t0 = time.time()
    with mp.Pool(nproc) as pool:
        par = pool.map(solve, tasks)
    t_par = time.time() - t0

    print(f"shape: {os.path.basename(SHAPE)}  f={F/1e9:.0f}GHz T={T:.0f}K "
          f"x={np.pi * DEQ_UM / LAMB:.2f}  eps={EPS}")
    print(f"n={nsubset} solves, host cores={os.cpu_count()}")
    print(f"sequential: {t_seq:8.1f}s  (mean {np.mean(seq):.2f}s sd "
          f"{np.std(seq):.2f})")
    print(f"pool[{nproc}]:   {t_par:8.1f}s  (mean {np.mean(par):.2f}s sd "
          f"{np.std(par):.2f})")
    print(f"wall speedup: {t_seq / t_par:.2f}x   CPU efficiency: "
          f"{t_seq / t_par / nproc:.2f}")


if __name__ == "__main__":
    main()
