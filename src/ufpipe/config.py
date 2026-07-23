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
# Anti-alias lowpass applied BEFORE any down-sample to SAMPLING_RATE (e.g. 200 Hz KG stations).
# 45 Hz = 0.9 * new Nyquist (50 Hz), leaving a guard band; zero-phase keeps onset times put.
ANTIALIAS = dict(freq=45.0, corners=8, zerophase=True)
BANDPASS = dict(freqmin=1.0, freqmax=40.0, corners=4, zerophase=False)
MERGE = dict(method=1, fill_value=0)
P_THRESHOLD = 0.2
S_THRESHOLD = 0.2
# SHARED 64-core server — keep the footprint polite. Detection sizes its preprocessing
# pool and torch threads from the process CPU affinity (os.sched_getaffinity), so launching
# under `taskset -c <cores>` automatically scopes everything to that core budget. MAX_CORES
# is a safety ceiling applied even if taskset is forgotten. The forkserver pool is created
# ONCE per year (lean workers, never forks the model/CUDA parent); inference runs on the GPU.
# See docs/performance-notes.md ("Polite CPU use").
MAX_CORES = 24
# Fragmentation handling. Some station-days are stored as tens of thousands of short
# contiguous miniSEED records (e.g. YSB 2010.082 = 55,572 records). The DATA is real and
# continuous — obspy merge() stitches it correctly but is ~O(n^2) in record count (~100 s
# for 55k). We process such days LOSSLESSLY (the merge is just slow) and only log them.
# A hard cap guards against a truly corrupt file that would stall for many minutes.
MAX_SEGMENTS = 2000          # above this: log "fragmented, slow merge" but still process
HARD_MAX_SEGMENTS = 300000   # above this: skip (abnormal — avoid an unbounded stall)

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
# Stage-2 tightened defaults — opt-in via `association.py --strict` (default off so
# the loose baseline is reproducible). PyOcto's default `pick_match_tolerance=1.5 s`
# admits hypocenters within ±5–10 km of the true location at typical 5–8 km/s
# wavespeeds, which produces the 2017-11-15 06:09-style chimera associations
# (HypoInverse max|residual| up to 4.86 s). The tightened set rejects them.
REGION_STRICT = dict(
    lat=(34.5, 37.0),
    lon=(128.5, 130.0),
    zlim=(0, 40),
    time_before=300,
    n_picks=6,                  # was 4 — avoid trivial events
    n_p_picks=3,                # was 2 — stronger origin-time constraint
    n_s_picks=3,                # was 2 — symmetric with P; demands genuine S coverage
    n_p_and_s_picks=2,          # was 1 — two stations with both P+S for depth
    pick_match_tolerance=1.0,   # default 1.5 — THE primary residual cap (middle ground)
    min_node_size=2.0,          # default 10.0 km — finer initial octree; reduces wrong-basin
                                # convergences on one-sided station coverage (the 2013-03-22
                                # 13:40:04 case where PyOcto's coarse search landed at a phantom
                                # 36.66°N hypocenter 110 km north of the true location).
    min_node_size_location=0.5, # default 1.5 km — finer hypocenter refinement
    refinement_iterations=8,    # default 3 — more localise+pick-rematch cycles to escape
                                # local minima. NOTE: tweak alone is insufficient — PyOcto's
                                # streaming associator truncates at threshold so FARTHER
                                # stations get dropped. Paired with the post-locate
                                # pick-augmentation stage (uses PyOcto's now-correct hypocenter
                                # to scan for picks PyOcto missed at threshold).
    min_interevent_time=2.0,    # default 3.0 s — allows real doublets
    n_threads=16,               # default = all available cores. Be polite on a shared 64-core box:
                                # 16 threads is enough to make progress without crowding out
                                # other users. Bump up only if the box is dedicated.
)
# label embedded in pyocto_<label>_<year>.csv (the PyOcto layered velocity model)
PYOCTO_VELMODEL = "kim1983"

# -------------------------------------------------- HYPOINVERSE crustal model
DEFAULT_VELMODEL = "kim2011"          # which crustal model UF<year>.sh used
PHASE_CHANNELS = {"P": "HHZ", "S": "HHN"}

# ------------------------------------------------ picker backends (by model)
# SeisBench PhaseNet weights run via the default backend; EQNet PhaseNet+ runs
# via the in-process EQNet backend (see core._run_detection_year_eqnet).
SEISBENCH_MODELS = {"stead", "original", "instance", "ethz", "scedc", "geofon", "neic"}
EQNET_MODELS = {"phasenet_plus"}

# EQNet (AI4EPS) — external clone required for PhaseNet+ (not vendored in this repo)
EQNET_DIR = "/home/msseo/works/14.EQNet/EQNet"
EQNET_WEIGHTS = os.path.join(EQNET_DIR, "docs", "model_phasenet_plus", "model_99.pth")

# PhaseNet+ params. NOTE: PhaseNet-style pickers expect RAW (demeaned) data + their
# own internal normalization — do NOT bandpass. A gentle highpass is optional.
PNPLUS_MIN_PROB = 0.3          # EQNet default pick threshold
PNPLUS_HIGHPASS = 0.0          # Hz; 0.0 = no filter (raw). Set ~1.0 for a gentle highpass.
PNPLUS_NT = 1024 * 36          # time-samples per inference patch (cut_patch)


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
def phasenet_plus_raw_dir(model, y): return os.path.join(MODELS, model, "phasenet_plus_raw", str(y))


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
