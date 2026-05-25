"""
Shared configuration for the KS_KG seismicity-catalog pipeline.

One source of truth for paths, detection/association parameters, and the
model/year-aware path resolvers used by core.py and the CLIs.

The picker model ("stead", "original", ...) and velocity model ("kim1983",
"kim2011") are independent dimensions — see models/README.md.
"""
import os
import sys
import calendar

# allow `import config` whether run as a script or imported from a notebook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------- paths
ROOT = "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG"
MODELS = os.path.join(ROOT, "models")
CONTINUOUS = os.path.join(ROOT, "continuous")
STATION_UPDATE = os.path.join(ROOT, "station_table", "station_update.dat")
STATIONS_CSV = os.path.join(ROOT, "station_table", "stations.csv")
VELOCITY_CSV = os.path.join(ROOT, "velocity_model", "kim1983.csv")

# --------------------------------------------- detection (PhaseNet) defaults
# Verified uniform across 2010-2024.
SAMPLING_RATE = 100.0
PAD_TIME = 10.0
BANDPASS = dict(freqmin=1.0, freqmax=40.0, corners=4, zerophase=False)
MERGE = dict(method=1, fill_value=0)
P_THRESHOLD = 0.2
S_THRESHOLD = 0.2

# ---------------------------------------------- association (PyOcto) defaults
REGION = dict(
    lat=(34.5, 37.0),
    lon=(128.5, 130.0),
    zlim=(0, 40),
    time_before=300,
    n_picks=4,
    n_p_picks=2,
    n_s_picks=2,
    n_p_and_s_picks=1,
)
# label embedded in pyocto_<label>_<year>.csv (the PyOcto layered velocity model)
PYOCTO_VELMODEL = "kim1983"

# -------------------------------------------------- HYPOINVERSE crustal model
DEFAULT_VELMODEL = "kim2011"          # which crustal model UF<year>.sh used
PHASE_CHANNELS = {"P": "HHZ", "S": "HHN"}


def days_in_year(year):
    return 366 if calendar.isleap(year) else 365


# ------------------------------------------------------------- path resolvers
def model_root(model):              return os.path.join(MODELS, model)
def detection_year_dir(model, y):   return os.path.join(MODELS, model, "detection_location", str(y))
def picks_dir(model, y):            return os.path.join(detection_year_dir(model, y), "picks")
def pyocto_dir(model):              return os.path.join(MODELS, model, "pyocto")
def pyocto_events(model, y):        return os.path.join(pyocto_dir(model), f"pyocto_{PYOCTO_VELMODEL}_{y}.csv")
def pyocto_assign(model, y):        return os.path.join(pyocto_dir(model), f"pyocto_assignment_{PYOCTO_VELMODEL}_{y}.csv")
def station_table_dir(model):       return os.path.join(MODELS, model, "station_table")
def stations_year_csv(model, y):    return os.path.join(station_table_dir(model), f"stations_{y}.csv")
def hyp_dir(model):                 return os.path.join(MODELS, model, "HypoInv")
def phs_dir(model):                 return os.path.join(hyp_dir(model), "PHS")
def phs_file(model, y):             return os.path.join(phs_dir(model), f"UF{y}.phs")
def velmodel_dir(model, vm):        return os.path.join(hyp_dir(model), vm)


# --------------------------------------------------------------- safety guard
class SteadWriteError(RuntimeError):
    pass


def assert_writable(model, force=False):
    """`models/stead/*` are symlinks to the existing reference run — refuse to
    write there unless the caller explicitly forces it."""
    if model == "stead" and not force:
        raise SteadWriteError(
            "Refusing to write into model='stead' (it symlinks the existing reference run). "
            "Pass --force only if you really intend to overwrite the reference outputs."
        )
