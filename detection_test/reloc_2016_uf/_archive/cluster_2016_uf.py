"""DRAFT cluster config — 2016 Ulsan-Fault subregion relocation (FOR REVIEW, not yet run).

Relocates the 2016 UF-subregion events detected by our PhaseNet+ -> PyOcto pipeline, through the
korea-cluster-relocation stages: waveforms(=our event-SAC) -> [picks reused] -> HYPOINVERSE -> ph2dt -> dt.ct
-> GPU dt.cc. Template = east_gyeongju (the 2016 KS+KG relocation), with the waveform backend switched to our
per-event SAC store (event_sac_export output) and the choices below adapted for OUR catalog.
"""
import os
from pipeline.config import ClusterConfig
from pipeline.clusters._base import STD_HYP, STD_PH2DT, _kim_models, _dtct_inp, _dtcc_variants

ROOT = "/home/msseo/works/02.Ulsan_Fault_detection/detection_test/reloc_2016_uf"
HYP  = "/home/msseo/works/02.Ulsan_Fault_detection/East_gyeongju_cluster/1.HypoInv"  # reuse kim1983/kim2011 model dirs

CONFIG = ClusterConfig(
    name="gyeongju_uf_2016",
    region="Gyeongju_UF_2016",
    src_root=ROOT,

    # --- catalog: our 2016 UF-box events. Written with year/month/day..sec = the UTC ORIGIN (see decision #1) ---
    event_catalog_csv=os.path.join(ROOT, "event_catalog", "event_catalog.csv"),
    kst_offset_hours=0,                       # #1 our origins are already UTC -> offset 0 (NOT 9)

    # --- stations: KS + KG + NS + GJ coords, in the Network,Code,Latitude,Longitude,Elevation master format ---
    station_master_csvs=(os.path.join(ROOT, "station_table", "KS_station.csv"),
                         os.path.join(ROOT, "station_table", "KG_station.csv"),
                         os.path.join(ROOT, "station_table", "NS_station.csv"),
                         os.path.join(ROOT, "station_table", "GJ_station.csv")),

    # --- region: the UF box (129.25-129.55, 35.60-35.90) ---
    epicenter=(35.75, 129.40),                # UF-box centre (lat, lon)
    radius_km=100.0,
    region_bounds=(35.60, 35.90, 129.25, 129.55),   # (latmin, latmax, lonmin, lonmax) = the UF box

    # --- waveforms: OUR per-event SAC store (event_sac_export), FLAT per-event layout ---
    wf_source="stp_sac",                      # #2 reuse our extracted SAC (not KMA/STP download)
    stp_sac_root=os.path.join(ROOT, "event_sac"),
    stp_sac_glob={"HH": "*.HH{comp}.sac", "EL": "*.EL{comp}.sac", "HG": "*.HG{comp}.sac"},  # flat: <ev>.<NET>.<STA>.<CHAN>.sac
    sensor_priority=("HH", "EL", "HG"),       # matches our detection band priority (gj_config.BANDS)
    target_sampling_hz=100.0,

    # --- picks: REUSE our PhaseNet+ picks (event_sac_export wrote them to SAC a/t0 + picks CSV) ---
    #      => run the pipeline from stage "hypoinverse" (skip re-picking). See decision #3.
    picker_weights="stead", p_threshold=0.2, s_threshold=0.2, sp_max_gap_s=15.0,
    phs_weight_scheme="probability",          # HypoInverse weight from PhaseNet+ pick probability (as east_gyeongju)

    # --- location: HYPOINVERSE, standard control block, kim2011 + kim1983 ---
    hyp_control=STD_HYP,                       # CON50/MIN4/ZTR10F/DIS 4 50 1 3/RMS 4 .12 2 4 (byte-identical to gyeongju)
    velocity_models=_kim_models(HYP),         # (kim1983, kim2011); kim2011 = primary reloc model (decision #4)

    # --- relocation: HypoDD ph2dt -> dt.ct -> dt.cc (GPU) ---
    ph2dt=STD_PH2DT,
    hypodd_dtct=_dtct_inp(isolv=1),           # #5 isolv=1 (SVD); switch to 2 (LSQR) if the QC'd set is large
    hypodd_dtcc_variants=_dtcc_variants(isolv=1),
    xcorr_backend="cctorch_gpu_batched",      # GPU dt.cc (we have the RTX 6000)

    mainshock_event_id=None,                   # UF subregion has no single mainshock (M5.8 is OUTSIDE the box)
    cuspid_offset=200000,
    num_cores=16,
)
# QC after HYPOINVERSE is applied SEPARATELY with your canonical data/hypoinv/uf_cluster.apply_qc
# (erh<5 & erz<5 & gap<270 & num>5 & rms<1.0) before HypoDD — NOT re-implemented here.
