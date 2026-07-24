#!/bin/bash
cd /home/msseo/works/02.Ulsan_Fault_detection/analysis/local_magnitudes
for nb in 03.Magnitude_summary 10.Magnitude_summary_homogenised 11.Mc_completeness_investigation \
          12.EMR_completeness 13.UF_secular_changes_artifacts 14.UF_ETAS_declustering_comparison \
          15.Time_dependent_station_corrections 16.Magnitude_timedependence_synthesis \
          17.Response_epoch_corrected_catalog 18.Why_epoch_station_correction \
          04.Catalog_quality_audit 19.HDB_2015_amplitude_diagnosis; do
  echo "### $(date +%H:%M) start $nb"
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb --ExecutePreprocessor.timeout=1500 > logs_$nb.log 2>&1
  echo "### $(date +%H:%M) end $nb exit=$?"
done
echo "ALL DONE $(date +%H:%M)"
