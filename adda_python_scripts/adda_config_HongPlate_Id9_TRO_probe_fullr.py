# -*- coding: utf-8 -*-
"""Phase-5 TRO GATE twin: same jobs as adda_config_HongPlate_Id9_TRO_probe but
with the FULL beta range (no XY folding). Compare CrossSec + averaged mueller
against the folded run: agreement ~Romberg eps + solver noise proves the
XY-mirror-reduced orientation average (report Phase-5 §5.3)."""

from adda_config_HongPlate_Id9_TRO_probe import *

avg_param_path=adda_prefix+\
    'adda_aux_input/HongPlate_Id9_TOR_sizegrid/avg_params_fullrange.dat'

static_clo=shape_str+[scat_mat,scat_mat_type]+orient_input+[avg_param_path]

shape_name='HongPlate_Id9_TOR_fullr'   # distinct storage folders
