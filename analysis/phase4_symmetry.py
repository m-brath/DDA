#!/usr/bin/env python3
"""Phase-4 step 16: determine the TRUE symmetries of each habit's dipole set
numerically instead of hand-guessing (all config flags are currently False).

For every habit folder, the lowest-resolution shape file (fmin=192 GHz class;
optionally more) is parsed and tested:

  mirror planes  sym_XYobj (z->-z), sym_XZobj (y->-y), sym_YZobj (x->-x)
      tested EXACTLY on the voxel index set after bbox-center flip + best
      integer translation (voxel-grid exact -> flag is safe);
  rotation Cn about the particle z-axis (sym_Zobj, ADDA -orient-reduction)
      tested geometrically: rotate point cloud about the centroid, score the
      fraction of points whose nearest neighbour is < 0.5 voxel (a discretized
      Cn habit scores ~1, a broken one much less).

Reports per habit; exact mirror (fraction 1.000) -> pipeline flag can be set
True; strict Cn (score>=0.999 AND max deviation <= covering radius) ->
sym_Zobj=n plausible (ADDA pipeline support: Phase 4 design). Coarse shape
files (<~20 voxels across) are not resolvable for Cn — trust the fine ones.
"""
import glob
import itertools
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPE_MAIN = os.path.join(REPO, "adda_shape_files")
HABITS = ["HongPlate_Id9", "HongColumn_Id7", "IconSnow_Id28",
          "LiuSectorSnowflake_Id3", "HexPlaAggCrystal_Id20",
          "HongPlate_Id9_TOR_sizegrid"]


def parse_adda(path):
    """Return (idx[N,3] ints, divisions[3], step[3]) from an ADDA geometry file."""
    divisions = step = None
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.search(r"divisions are \(([^)]*)\)", line)
            if m:
                divisions = np.array([float(x) for x in m.group(1).split(",")])
            m = re.search(r"Step sizes are \(([^)]*)\)", line)
            if m:
                step = np.array([float(x) for x in m.group(1).split(",")])
    idx = np.loadtxt(path, comments="#", dtype=int)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"unexpected data format in {path}")
    return idx, divisions, step


def _rows(a):
    """view of unique rows as 1-D scalar array (intersect1d would otherwise
    flatten the 2-D index set into scalar values). NOTE the itemsize must be
    the FULL ROW width, not the scalar element."""
    a = np.ascontiguousarray(a)
    return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1]))))


def mirror_exact(idx, axis):
    """exact voxel-grid mirror test about bbox centre (allow integer shift);
    array-set based so it scales to tens of millions of voxels"""
    n = idx.max(axis=0) + 1
    s = idx.copy().astype(int)
    s[:, axis] = n[axis] - 1 - s[:, axis]
    # best integer translation = centroid difference (tight bbox => usually 0)
    t = np.round(s.mean(axis=0) - idx.mean(axis=0)).astype(int)
    A = _rows(idx)
    B = _rows(s + t)
    inter = np.intersect1d(A, B, assume_unique=True).shape[0]
    return inter / min(len(A), len(B))  # 1.0 -> exact


def rot_exact_z(idx, n):
    """exact voxel-grid closure test for lattice-preserving rotations about z
    (n=2: 180 deg, n=4: 90 deg). CRITICAL (voxel shapes are cubes on a CUBIC
    lattice): rotations by other angles (n=3, 6, ...) do NOT map the lattice
    onto itself, so a discretized hexagonal habit can never be exactly C6 as
    a SHAPE MODEL — its voxelization under 60 deg rotation is a different
    voxel set differing by an O(1-voxel) boundary layer. Only C2/C4 about
    z (and compositions of exact mirrors, which are implied exactly) can ever
    be exact; use rot_score() to detect ideal-shape Cn at sub-voxel level."""
    if n == 2:
        s = idx.copy().astype(int)
        s[:, 0] = idx.max(axis=0)[0] - s[:, 0]
        s[:, 1] = idx.max(axis=0)[1] - s[:, 1]
    elif n == 4:
        s = np.stack([idx[:, 1], idx.max(axis=0)[0] - idx[:, 0], idx[:, 2]],
                     axis=1).astype(int)
    else:
        raise ValueError('only n=2,4 preserve the cubic lattice about z')
    t = np.round(s.mean(axis=0) - idx.mean(axis=0)).astype(int)
    A = _rows(idx)
    B = _rows(s + t)
    inter = np.intersect1d(A, B, assume_unique=True).shape[0]
    return inter / min(len(A), len(B))  # 1.0 -> exact


def rot_score(pos, n, maxpts=3_000_000):
    """IDEAL-SHAPE Cn test (geometric, NOT model-level): (score, max_dev) of
    rotated point set vs itself, tolerance = cubic lattice covering radius
    sqrt(3)/2. score=1 means the rotation of the shape differs from the
    voxel model only sub-voxel-wide (boundary-layer discretization error),
    NOT that the voxel set is invariant — see rot_exact_z. False Cn: score<1,
    max_dev up to shape-size. NOTE: only meaningful for well-resolved shapes
    (>>10 voxels across); large sets are randomly subsampled, which dilutes
    the score (sampled tree = inflated NN distances) — keep the cap above the
    production-habit sizes so strict gates stay valid."""
    if len(pos) > maxpts:
        rng = np.random.default_rng(0)
        pos = pos[rng.choice(len(pos), size=maxpts, replace=False)]
    ang = 2 * np.pi / n
    R = np.array([[np.cos(ang), -np.sin(ang), 0],
                  [np.sin(ang), np.cos(ang), 0],
                  [0, 0, 1]])
    pos = pos - pos.mean(axis=0)
    tree = cKDTree(pos)
    d, _ = tree.query(pos @ R.T, distance_upper_bound=1e9)
    return float(np.mean(d <= np.sqrt(3) / 2)), float(np.nanmax(d))


def main(habits=HABITS):
    print(f"{'habit':28s} {'shape file':24s} {'XY':>6s} {'XZ':>6s} {'YZ':>6s}  {'C2z':>5s} {'C4z':>5s}   ideal-shape Cn scores (geometric, sub-voxel — model-level impossible for n≠2,4 about z)")
    for habit in habits:
        files = sorted(glob.glob(os.path.join(SHAPE_MAIN, habit, "*.adda")))
        if not files:
            print(f"{habit:28s}  -- no .adda files")
            continue
        # test the 192-GHz (coarsest) file and the finest, if different
        probe = [files[0]]
        if len(files) > 1:
            probe.append(files[-1])
        for path in probe:
            idx, div, step = parse_adda(path)
            # mirror test on the FULL voxel set (a random subsample of a
            # mirror-closed set is NOT closed); rot_score samples internally
            fr = [mirror_exact(idx, ax) for ax in (2, 1, 0)]  # XY, XZ, YZ
            e2, e4 = rot_exact_z(idx, 2), rot_exact_z(idx, 4)
            pos = idx.astype(float)  # voxel units (isotropic step in meta)
            scores = {n: rot_score(pos, n) for n in (2, 3, 4, 6, 12)}
            symZ = next((n for n in (12, 6, 4, 3, 2) if scores[n][0] >= 0.999), None)
            name = os.path.basename(path).split(".")[0][:24]
            print(f"{habit:28s} {name:24s} N={len(idx):7d} "
                  f"{fr[0]:6.3f} {fr[1]:6.3f} {fr[2]:6.3f}  "
                  f"{e2:5.3f} {e4:5.3f}   "
                  + " ".join(f"n{n}={scores[n][0]:.3f}/{scores[n][1]:5.1f}" for n in (2, 3, 4, 6, 12))
                  + f"  ideal-Cn={symZ}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else HABITS)
