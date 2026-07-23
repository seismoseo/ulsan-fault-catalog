"""Post-batch: correct the stale '510' coherence-QC count -> 133 in notebook markdown (no re-exec)."""
import json, glob
OLD1="510 PhaseNet+ edge-mislocations removed"
NEW1="133 PhaseNet+ edge-mislocations removed"
OLD2="510 such edge-mislocations were dropped catalog-wide"
NEW2=("133 such edge-mislocations were dropped catalog-wide (the densest available pick archive, "
      "detection_location; an earlier note cited 510 from an unreproducible legacy pick set — see "
      "CLEAN_CATALOG_PROVENANCE.md)")
for f in ["03.Magnitude_summary.ipynb","10.Magnitude_summary_homogenised.ipynb","06.Magnitude_summary_sheen.ipynb"]:
    try: nb=json.load(open(f))
    except FileNotFoundError: continue
    ch=0
    for c in nb["cells"]:
        if c["cell_type"]!="markdown": continue
        src="".join(c["source"])
        if OLD1 in src or OLD2 in src:
            src=src.replace(OLD1,NEW1).replace(OLD2,NEW2)
            c["source"]=src.splitlines(keepends=True); ch+=1
    if ch: json.dump(nb,open(f,"w"),indent=1); print(f"{f}: updated {ch} markdown cell(s)")
    else: print(f"{f}: no 510 text found")
