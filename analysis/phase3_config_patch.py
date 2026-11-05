#!/usr/bin/env python3
"""Phase 3: uniform config modernization (kept as a record of the patch).

For each active adda_config_*.py:
 1. import os/sys; 2. darwin 'local' branch in the hostname block;
 3. prefix+'/adda/... -> adda_prefix+... path indirection;
 4. provenance + configurable ADDA binary block (adda_prefix, cluster_name,
    adda_version/adda_binary/adda_launcher with env overrides);
 5. ramdrive=None for local mode; 6. drop never-parsed -Csca/-alldir_inp
    from scatoutput_str.
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(os.path.dirname(HERE), "adda_python_scripts")

DARWIN = '''elif sys.platform.startswith('darwin'):

    # local (no-queue) mode (Phase 3): paths relative to this repo
    _here=os.path.dirname(os.path.abspath(__file__))+'/'
    _repo=os.path.dirname(_here[:-1])+'/'
    prefix=_here
    prefix_data=_repo
    calc_type='local'
    Partition='n/a'
    #number of cores
    ncores=4

'''

BINBLOCK = '''#==============================================================================
# ADDA checkout, provenance and binary (Phase 3 - configurable binary)
#==============================================================================

#path of the adda checkout (cluster: <prefix>/adda; local: this repo root)
if calc_type=='local':
    adda_prefix=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/'
else:
    adda_prefix=prefix+'adda/'

#provenance strings for the SSDB CalculationData group
if calc_type=='mistral':
    cluster_name='DKRZ: mistral-compute2'
elif calc_type=='local':
    cluster_name='local: '+hereiam
else:
    cluster_name='TCUB: thunder'

#ADDA executable and launcher (consumed by the adda_sub_control_*.py scripts;
#they fall back to the historic 1.2 defaults if a config defines neither).
#Env overrides enable cross-version runs without editing configs:
#  ADDA_BINARY=<exe> ADDA_VERSION=<ver> python adda_main_control.py ...
adda_version='1.2'

if calc_type=='mistral':
    adda_binary=adda_prefix+'adda_1.2/src/mpi/adda_mpi'
    adda_launcher=['srun','-n',str(ncores)]
elif calc_type=='local':
    adda_binary=adda_prefix+'adda_1.2/src/seq/adda'
    adda_launcher=[]
else:
    adda_binary='adda_mpi'
    adda_launcher=['mpiexec','-n',str(ncores)]

adda_version=os.environ.get('ADDA_VERSION',adda_version)
adda_binary=os.environ.get('ADDA_BINARY',adda_binary)

'''

RAM_NEW = '''if calc_type=='mistral':
    ramdrive="/dev/shm/"
elif calc_type=='local':
    ramdrive=None      # system temp dir
else:
    ramdrive="/dev/shm/u242031/"'''

SCAT_NEW = '''# -Csca / -alldir_inp dropped (Phase 3): CrossSec*/Csca/mueller_alldir are
# never parsed by this pipeline (ARO extinction comes from the forward
# ampl_scatgrid column); skipping them saves the all-direction integration
# and avoids the 1.4/1.5 flag-combination validation traps (report \u00a71.3)
scatoutput_str=[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type]'''

files = sorted(glob.glob(os.path.join(CFG, "adda_config_*.py")))
for path in files:
    name = os.path.basename(path)
    src = open(path).read()
    orig = src
    n = {}

    if "import os" not in src.split("hereiam")[0]:
        src, n["imports"] = re.subn(r"(?m)^import socket$", "import socket\nimport os\nimport sys", src, count=1)

    src, n["darwin"] = re.subn(
        r"(?m)^else:\n(    prefix='/scratch/uni/u237/users/mbrath/')",
        DARWIN + r"else:\n\1", src, count=1)

    src, n["path1"] = re.subn(r"prefix\+'/adda/", "adda_prefix+'", src)
    src, n["path2"] = re.subn(r"prefix\+'adda/", "adda_prefix+'", src)

    src, n["binblock"] = re.subn(r"(?m)^#ramdrive", BINBLOCK + "#ramdrive", src, count=1)

    src, n["ramdrive"] = re.subn(
        r'''(?m)if calc_type=='mistral':\n    ramdrive="/dev/shm/"\nelse:\n    ramdrive="/dev/shm/u242031/"\s*''',
        RAM_NEW, src, count=1)

    src, n["scatoutput"] = re.subn(
        r"(?m)^scatoutput_str=\[scat_output,scat_input,scat_param_path,scat_mat,scat_mat_type,\n\s*Csca_output,alldir_input,alldir_param_path\][^\n]*",
        SCAT_NEW, src, count=1)

    assert all(v == 1 for k, v in n.items() if k not in ("path1", "path2")), (name, n)
    assert n["path1"] + n["path2"] > 0, (name, n)
    if src != orig:
        open(path, "w").write(src)
        print(f"patched {name}: {n}")
    else:
        print(f"NO CHANGE {name}: {n}")
