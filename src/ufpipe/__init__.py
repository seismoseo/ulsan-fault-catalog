"""ufpipe — Ulsan-Fault seismicity-catalog pipeline (detection -> association -> PHS -> HYPOINVERSE).

Renamed from the former outputs/models/pipeline (avoids the `pipeline` name collision with the external
korea-cluster-relocation package). Installable: `pip install -e .` (env base), then:

    from ufpipe import core, config
    core.run_detection_year("original", 2024, days=range(1, 6))

or driven from the CLIs:  python -m ufpipe.run_pipeline --model original --years 2024
(the module files detection.py / association.py / make_phs.py / run_hypoinverse.py / run_pipeline.py
also run directly, e.g. `python run_pipeline.py ...`, via their self-insert).
"""
