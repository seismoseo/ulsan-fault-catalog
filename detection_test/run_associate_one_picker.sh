#!/bin/bash
# One picker's 2016 PyOcto association (daily-chunked, all 12 months). Consistent gate (gj_config 4/2/2),
# kim2011 velocity, min-coverage 0 (all stations) — identical to the PN+ run; the picker is the only variable.
# Writes catalogs/catalog_<picker>_2016_<mm>_pyocto.csv + assign_<picker>_2016_<mm>_pyocto.parquet.
# Association is PyOcto-internally-threaded, so run pickers SEQUENTIALLY here (not 3 pools oversubscribing).
set -u
picker=$1
cd /home/msseo/works/02.Ulsan_Fault_detection/Gyeongju_catalog/detection_test/lib || exit 1
for mm in 01 02 03 04 05 06 07 08 09 10 11 12; do
  echo "########## assoc $picker 2016-$mm START $(date '+%F %T') ##########"
  python3 -u associate_daily.py --picker "$picker" --month "2016-$mm"
  echo "########## assoc $picker 2016-$mm DONE  $(date '+%F %T') ##########"
done
echo "########## assoc $picker ALL DONE $(date '+%F %T') ##########"
