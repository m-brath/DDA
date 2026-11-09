# -*- coding: utf-8 -*-
"""
Dump the ANGULAR grids (full field, eikonal model, residual) of a few Phase-9A
scan cells so docs/make_figures.py (fig15) can draw the delta-scaling story
from REAL scan data instead of a toy.  Re-solves ADDA (noise-based, like the
scan itself).  Grids land in analysis/phase9_grids_<tag>.npz (committed).

Run:  ~/miniforge3/envs/DDA/bin/python analysis/phase9_dump_grids.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import phase9_fwdmodel_scan as p9  # noqa: E402

# G1-established global calibration (only used for the dev printout; the
# per-cell LS calibration is fitted inside eval_cell anyway)
CAL = complex(1.0053, -0.0061)

HP = os.path.join(REPO, 'adda_shape_files', 'HongPlate_Id9')

CELLS = [
    # (shape glob, orientation, tag)
    ('*Dveq00215um_Mass*Freq0887.000GHz.adda', (0, 0, 0, (0, 0, 1)),
     'HP_x2_face'),      # clean model win (10 -> 3 full-eps ntrunc)
    ('*Dveq00541um_Mass*Freq0887.000GHz.adda', (0, 0, 0, (0, 0, 1)),
     'HP_x5_face'),      # poster child (25 -> 6)
    ('*Dveq00541um_Mass*Freq0887.000GHz.adda', (0, 90, 0, (0, 0, 1)),
     'HP_x5_edge'),      # actively harmful (|g|/|f| = 1.73, Cext_model < 0)
]


def main():
    workdir = os.path.join(HERE, 'phase9_dump_work')
    os.makedirs(workdir, exist_ok=True)
    for pat, (a, b, gm, prop), tag in CELLS:
        out = os.path.join(HERE, 'phase9_grids_%s.npz' % tag)
        if os.path.exists(out):
            print('skip %s (cached)' % tag)
            continue
        p9.eval_cell(HP, pat, p9.FREQ, p9.TENV, a, b, gm, prop, tag,
                     workdir, CAL, dump_npz=out)
    print('done')


if __name__ == '__main__':
    main()
