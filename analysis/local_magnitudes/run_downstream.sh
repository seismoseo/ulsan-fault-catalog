#!/bin/bash
cd /home/msseo/works/02.Ulsan_Fault_detection/KS_KG/local_magnitudes
for nb in 03.Magnitude_summary 10.Magnitude_summary_homogenised 11.Mc_completeness_investigation 12.EMR_completeness 04.Catalog_quality_audit 13.UF_secular_changes_artifacts 14.UF_ETAS_declustering_comparison 19.HDB_2015_amplitude_diagnosis; do
  echo "### $(date +%H:%M) running $nb"
  jupyter nbconvert --to notebook --execute --inplace $nb.ipynb --ExecutePreprocessor.timeout=900 > logs_$nb.log 2>&1
  echo "### $(date +%H:%M) $nb exit=$?"
done
echo "ALL DOWNSTREAM DONE"
