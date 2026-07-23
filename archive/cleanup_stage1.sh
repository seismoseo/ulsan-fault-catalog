#!/usr/bin/env bash
# =============================================================================
# cleanup_stage1.sh — archive the Stage-1 (KS/KG initial exploration) results
# into stage1_KS_KG_exploration/ so the Gyeongju-catalog stage starts clean.
#
# USAGE (run it yourself, from 02.Ulsan_Fault_detection/):
#   bash cleanup_stage1.sh            # DRY RUN — prints what would move, moves nothing
#   bash cleanup_stage1.sh --apply    # actually perform the moves
#
# WHAT IT DOES
#   * Creates stage1_KS_KG_exploration/ and MOVES the Stage-1 items into it
#     (git mv when the item is tracked, plain mv otherwise). NOTHING is deleted.
#   * Leaves in place everything the new stage will reuse:
#       GJ/                    2016 Gyeongju aftershock temporary arrays (GJ/PK/SS/TP waveforms)
#       NS/                    GHBSN dense-network waveform archive
#       GHBSN_metadata/        NS station epochs + coordinates (+ orientation table)
#       GHBSN_catalog_Heoetal/ reference catalog
#       KIGAM_metadata/        KG StationXML (63 stations, channel epochs + responses)
#       papers/, tools/, environment.yml, requirements.txt, README.md, CLAUDE.md
#
# WARNINGS (read before --apply)
#   * Stage-1 notebooks/builders hard-code absolute paths (e.g.
#     /home/msseo/works/02.Ulsan_Fault_detection/KS_KG/...). After the move the
#     archived notebooks remain fully VIEWABLE, but RE-RUNNING them would need
#     path fixes. That is intended: the archive is a frozen record.
#   * Nothing outside this repository is touched. The relocation run data live
#     in 15.PocketQuake/external/korea-cluster-relocation and are unaffected.
#   * The moves are reversible (mv them back, or git mv back).
#   * CLAUDE.md at the repo root still describes Stage 1 — revise it for the
#     new stage when convenient (not done automatically).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ARCHIVE="stage1_KS_KG_exploration"
ITEMS=(
  "KS_KG"                             # the entire Stage-1 working tree (notebooks nb21-40, HypoDD runs, ML, picks, ...)
  "00.Summary_figures_Zhigang.ipynb"  # Stage-1 summary notebook
  "build_summary_zhigang_nb.py"       # its builder
  "figures_zhigang"                   # exported summary figures
  "01.PhaseNet_detection_test.ipynb"  # early detection tests
  "02.Multi-station_detection.ipynb"
  "_dens.cpt"                         # GMT color tables used by Stage-1 maps
  "_depth.cpt"
  "docs"                              # Stage-1 pipeline documentation
  "gmt.history"
)

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1
if [[ $APPLY -eq 0 ]]; then
  echo "================ DRY RUN (nothing will be moved) ================"
  echo "Re-run with:  bash cleanup_stage1.sh --apply"
  echo "================================================================="
fi

echo "[plan] archive directory: $ARCHIVE/"
[[ $APPLY -eq 1 ]] && mkdir -p "$ARCHIVE"

for item in "${ITEMS[@]}"; do
  if [[ ! -e "$item" ]]; then
    echo "[skip]  $item (not present)"
    continue
  fi
  # choose git mv when tracked so history follows the file
  if git ls-files --error-unmatch "$item" >/dev/null 2>&1 \
     || [[ -n "$(git ls-files "$item" 2>/dev/null)" ]]; then
    echo "[git mv] $item  ->  $ARCHIVE/$item"
    [[ $APPLY -eq 1 ]] && git mv "$item" "$ARCHIVE/$item"
  else
    echo "[mv]     $item  ->  $ARCHIVE/$item   (untracked/ignored)"
    [[ $APPLY -eq 1 ]] && mv "$item" "$ARCHIVE/$item"
  fi
done

# new-stage working directory
echo "[plan] new stage directory: Gyeongju_catalog/"
[[ $APPLY -eq 1 ]] && mkdir -p Gyeongju_catalog

if [[ $APPLY -eq 1 ]]; then
  echo
  echo "Done. Review with:  git status"
  echo "Commit yourself when satisfied, e.g.:"
  echo "  git add -A && git commit -m 'Archive stage-1 KS/KG exploration; start Gyeongju catalog stage'"
else
  echo
  echo "DRY RUN complete — nothing was moved."
fi
