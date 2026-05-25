"""KS_KG seismicity-catalog pipeline (detection -> association -> PHS -> HYPOINVERSE).

Importable from notebooks:

    import sys; sys.path.insert(0, "/home/msseo/works/02.Ulsan_Fault_detection/KS_KG/models/pipeline")
    import core, config
    core.run_detection_year("original", 2024, days=range(1, 6))

or driven from the CLIs (detection.py, association.py, make_phs.py,
run_hypoinverse.py, run_pipeline.py).
"""
