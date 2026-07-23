#!/bin/bash
# Systematic 4-picker comparison — DETECTION stage for the 3 new SeisBench pickers on 2016.
# PN+ (phasenet_plus) is already done. The other three run through the IDENTICAL detection path
# (same 2016 station caches, same 100 Hz anti-alias decimation via gj_config, same consistent P=S
# threshold 0.2) so the picker is the only variable:
#   (b) original -> PhaseNet 'original' (NCEDC-trained)
#   (c) stead    -> PhaseNet 'stead'    (STEAD-retrained)
#   (d) eqt      -> EQTransformer 'stead' (Mousavi 2020, STEAD)
# Station-checkpointed + resumable (re-run this script to continue after any interruption).
set -u
HERE=/home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test
cd "$HERE/lib" || exit 1
for picker in original stead eqt; do
  for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo "########## $picker 2016-$mm  START $(date '+%F %T') ##########"
    python3 -u run_seisbench_picker.py --model "$picker" --month "2016-$mm"
    echo "########## $picker 2016-$mm  DONE  $(date '+%F %T') ##########"
  done
done
echo "########## ALL DETECTION DONE $(date '+%F %T') ##########"
