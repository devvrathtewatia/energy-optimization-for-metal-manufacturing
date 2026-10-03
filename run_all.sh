#!/usr/bin/env bash
# Reproduce everything from scratch (about 1 hour on a 14-core laptop; the tuning in E06/E07 dominates).
# Usage:  python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && bash run_all.sh
set -euo pipefail
PY=${PY:-python}

$PY -m src.download_real                  # verify/download the real public data (md5 checked)
$PY -m src.synth_generator                # regenerate the synthetic data (bit-for-bit identical: sha256 in metadata)
$PY -m src.data_prep                      # cleaning rules + splits
$PY -m pytest -q tests                    # leakage / split / physics sanity tests

for e in e00_physics_baseline e01_eda e02_linear_poly e03_kmeans_regimes e04_e06_trees \
         e07_feature_engineering e08_neural_network e10_hybrid_physics_ml e09_explainability \
         e11_model_selection e12_anomaly_detection e13_optimization e14_real_data_probes; do
  echo "=== $e"; $PY -u -m experiments.$e
done
$PY -m experiments.summarize
$PY notebooks/build_notebooks.py          # re-executes the three notebooks against the results
