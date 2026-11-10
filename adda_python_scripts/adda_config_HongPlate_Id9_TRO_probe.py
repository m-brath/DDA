# -*- coding: utf-8 -*-
"""Phase-5 TRO PROBE config: production HongPlate_Id9_TRO restricted to the
smallest size classes and two frequencies — end-to-end validation of the
modernized TRO stack (9-row liste with temperature, orient-avg CLO, ADDA
1.4.0) before the production backlog (report Phase-5 §5.2).
"""

from adda_config_HongPlate_Id9_TRO import *

frequencies=np.array([228., 886.])*1e9
temperatures=np.array([230.])

_size_keep=Sizes<=30.   # Dveq 10 and 20 um entries only

liste=build_liste(frequencies,temperatures,_size_keep)

num_chunks=1
