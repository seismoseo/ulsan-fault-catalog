"""uflib — shared Ulsan-Fault analysis library.

Three modules, formerly in KS_KG/HypoInv/ and imported everywhere via a hard-coded
`sys.path.insert(".../KS_KG/HypoInv")`. Now a proper package: `pip install -e .` (env `ulsan`),
then `from uflib import uf_cluster` / `import uflib.uf_cluster` from any cwd.

  uf_cluster              spatial/temporal quarry-blast decluster + map helpers (QC dict, read_sum, apply_qc)
  uf_waveform_similarity  waveform-feature blast screening (imports uf_cluster)
  event_sac_export        event-idx-keyed SAC store writer (fully path-parameterized; portable)
"""
from . import uf_cluster, uf_waveform_similarity, event_sac_export  # noqa: F401

__all__ = ["uf_cluster", "uf_waveform_similarity", "event_sac_export"]
