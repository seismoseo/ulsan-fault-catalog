#!/bin/bash
# One picker's 2016 detection (12 months). Launch three of these concurrently (original/stead/eqt) to
# saturate the under-utilised GPU (detection is CPU/IO-bound; one picker uses only ~27% GPU). Station
# checkpoints make each month resumable, so re-launching continues instantly past completed months.
set -u
picker=$1
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test/lib || exit 1
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
  echo "########## $picker 2016-$mm START $(date '+%F %T') ##########"
  python3 -u run_seisbench_picker.py --model "$picker" --month "2016-$mm"
  echo "########## $picker 2016-$mm DONE  $(date '+%F %T') ##########"
done
echo "########## $picker ALL DONE $(date '+%F %T') ##########"
