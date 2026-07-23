"""Gyeongju-catalog detection pipeline — ALL parameters in ONE disclosed place (single source of truth).

Every runtime script imports from here; nothing is hard-coded or tuned per-month/per-dataset. If a value is
not in this file, it is either (a) a user-selected identity (picker=PhaseNet+, associator=PyOcto) or (b) pure
engineering with NO effect on results (GPU device, worker count, day-sharding, pre-decimation path).

Provenance tags:  [USER] your explicit choice   [BUAN] from the validated Buan/Ulsan config, unchanged
                  [DSP] standard signal processing   [EVAL] scoring only, never enters the catalog
"""

# ---- station set (build_stations.py) ---------------------------------------------------------------
REGION_CENTER = (35.856, 129.224)     # [USER] Gyeongju reference point (same convention as stage-1)
RMAX_KM       = 100.0                  # [USER] region radius
BANDS         = ("HH", "EL", "HG")     # [USER] channel priority, one band per station
MIN_COVERAGE  = 0.0                    # 0 => NO coverage filter: use every station with any local data

# ---- detection (run_pnplus_month.py / run_seisbench_picker.py) --------------------------------------
PICK_PROB        = 0.2                 # [USER/benchmark] PhaseNet+ pick probability threshold (same every month)
TARGET_FS        = 100.0              # [DSP] decimate anything faster to 100 Hz (the models' trained rate)
ANTIALIAS_FRAC   = 0.4                 # [DSP] anti-alias low-pass corner = FRAC * target Nyquist (= 40 Hz)
ANTIALIAS_CORNERS = 4                  # [DSP] low-pass poles, zero-phase

# ---- association (associate_daily.py — PyOcto, kim2011 1-D) -----------------------------------------
KIM2011 = {"depth": [0.00, 7.29, 20.70, 31.30],     # [USER] kim2011 1-D velocity (km, km/s)
           "vp":    [5.63, 6.17, 6.58, 7.77],
           "vs":    [3.40, 3.60, 3.70, 4.45]}
VEL_TOLERANCE        = 1.0             # [BUAN] PyOcto VelocityModel1D tolerance
GATE                 = {"n_picks": 4, "n_p": 2, "n_s": 2, "n_ps": 1}   # [BUAN] fixed permissive gate, ALL epochs
PICK_MATCH_TOLERANCE = 1.5             # [BUAN] PyOcto pick-match tolerance (s)
ZLIM                 = (0.0, 30.0)     # depth search range (km) — crustal
TIME_BEFORE          = 300.0           # PyOcto origin-search window (s)
ASSOC_LAT_PAD        = 1.0             # association area = REGION_CENTER +/- these (deg)
ASSOC_LON_PAD        = 1.2
ASSOC_OVERLAP_S      = 150             # daily-chunk overlap (s) > any local S-arrival + TIME_BEFORE margin

# ---- evaluation (notebooks) — SCORING ONLY, never enters the catalog --------------------------------
MATCH_DT_S  = 5.0                      # [EVAL] truth-match origin-time window
MATCH_DX_KM = 30.0                     # [EVAL] truth-match epicentre distance
