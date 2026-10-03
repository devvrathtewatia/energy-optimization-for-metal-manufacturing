# Experimental protocol (written BEFORE the modelling experiments were run)

This file fixes the rules of the investigation so that conclusions come from the results, not the other way
round. Nothing in here was changed after seeing model results; `LOG.md` reports what happened against
these expectations.

> **Disclosure.** Before writing this file a smoke test of the harness was run on four models (mean,
> zero-fit handbook, 2-parameter SEC, fitted parametric physics) on the synthetic data to make sure the
> code executes. It showed the fitted physics model reaching validation R² ≈ 0.92. Those numbers were not
> used to alter the generator, splits, metrics or the rules below.

## 1. Data

| | Real (public) | Synthetic (simulated) |
|---|---|---|
| Source | CFAA milling tests, Zenodo 14445879 (CC BY 4.0), Ibarmia THR-16 + GMTK VR 2.4 | `src/synth_generator.py`, seed 42, 2500 jobs |
| Rows | 6 000 per machine; 5 199 / 5 200 after de-duplication | 27 027 (26 967 after rule R1) |
| Frozen | md5 checked against Zenodo | sha256 `f8d297ff3a2ef85f…` (`data/synthetic/synthetic_metadata.json`) |
| Target | `powerDrive_SPINDLE` [kW] | `power_kw` = mean electrical power of a 10 s window [kW]; energy of a window = P·Δt |
| Role | sanity-check / what real data can and cannot support | main experimental engine, anomaly labels, optimisation |

Everything concluded on synthetic data describes how *methods behave on a controlled, physics-generated
problem*. It is **not** evidence about a real factory. Everything on real data is limited by the dataset's
missing process variables (no feed, depth of cut, spindle torque or timestamps).

## 2. Splits (no leakage)

* **Synthetic:** by JOB. Jobs are ranked by commanded material-removal rate (MRR). Top 6 % of jobs = `ood_test`,
  next 6 % (88-94th percentile) = `ood_val`; both are never seen in training. The rest is split 60/20/20 into
  `train` / `val` / `test` at random by job. Test and OOD-test numbers are computed for every model for
  completeness but are **not used for any decision**.
* **Real:** exact duplicates removed, then a random 60/20/20 split. The files have no run/time ids, so a
  grouped split is impossible: near-neighbour leakage between train and test cannot be excluded (stated limitation).
* All hyper-parameters are tuned on `train` only (grouped 4-fold CV). Scalers, K-Means, physics coefficients
  and neural-network early stopping are fitted inside `fit(train)`.
* Features for planning/optimisation are *commanded* conditions only. The drive's own load signals and the
  electrical measurements (V, I, cos φ) are excluded from predictors; E01 quantifies the leakage.

## 3. Metrics

MAE, RMSE, R² and MAPE (MAPE only on rows with y ≥ 1 kW synthetic, ≥ 0.5 kW real, because it explodes near zero).
Reported on all rows and, for the synthetic data, on *engaged* (cutting) windows, where the process-dependent
part of the power lives. A "nominal" subset (no sensor glitches, uses ground truth) is a diagnostic only.

**Why MAE drives model selection.** 0.7 % of synthetic rows are meter glitches (spikes up to ×3, dropouts).
A hypothetical oracle that knows the true noise-free power still has validation RMSE 0.88 kW (MAE 0.165 kW;
RMSE on non-glitch rows 0.154 kW). Squared error is therefore dominated by irreducible glitches and
compresses real differences between models; MAE is robust to them.

## 4. Final-model selection rule (frozen)

Candidate pool = every model entry from E00-E10 trained on the synthetic data (all variants, including the
ones that failed; the rule simply ranks them). The multi-seed neural network enters with its mean over seeds.

```
score(m) = 0.5 * MAE_val_all(m)/min_pool MAE_val_all  +  0.5 * MAE_oodval_engaged(m)/min_pool MAE_oodval_engaged
winner   = model with the lowest score, except: among all models with score <= 1.02 * best score,
           choose the one with the lowest `complexity` rank (0 mean < 1 handbook/SEC < 2 linear < 3 poly/physics-param
           < 4 K-Means+model < 5 tree < 6 forest/boosting < 7 hybrid < 8 neural net).
```
`test` and `ood_test` are then read once for the winner and reported as the final, un-tuned estimate.

## 5. Expectations recorded in advance (hypotheses, not conclusions)

| Exp | Expectation |
|---|---|
| E00 | Mean ≈ R² 0. 2-parameter SEC (P₀ + k·MRR) explains only part of the variance because speed-dependent idle loss and coolant dominate many windows (R² roughly 0.4-0.7). Zero-fit handbook: right order of magnitude, biased. |
| E01 | Cutting power is a modest increment on large fixed loads. Adding V, I, cos φ gives R² ≈ 1 (leakage). Glitches/outliers visible. |
| E02 | Linear fails on multiplicative structure (MRR = aₚ·aₑ·f_z·z·n); polynomial helps in-distribution but extrapolates badly (OOD worse than in-distribution by a wide margin). |
| E03 | K-Means with k ≈ 4 recovers standby / rapid / air-cut / cutting with high ARI. Cluster one-hot or per-cluster models help linear/polynomial models clearly; little or no gain for tree ensembles. |
| E04-E06 | Depth-limited tree < Random Forest < boosting in-distribution. All tree models flat-line outside the training MRR range, so OOD error ≫ in-distribution error. |
| E07 | Physics-based features (Kienzle power, idle-loss term) help every model class; they help linear models the most, trees mainly in-distribution. |
| E08 | MLP roughly matches tuned boosting in-distribution on 14 k rows but does not clearly beat it; seed variance is non-trivial; extrapolation is smoother than trees. Deep learning is **not** expected to win. |
| E09 | Top SHAP drivers: spindle speed, coolant mode, cutting-power/MRR-related features, material. Group-wise SHAP should resemble the true power decomposition stored in the generator. |
| E10 | A fitted physics-structured model is competitive in-distribution and best out-of-distribution; a residual hybrid adds a small in-distribution gain. |
| E12 | Residual thresholding catches large additive inefficiencies (stuck coolant, aux leak) with high recall; subtle ones (axis drag, bearing at low speed) are missed; spike glitches create false positives. |
| E13 | Simulated savings come mainly from raising MRR (amortising fixed loads) under power/tool-life limits and from speed choices that reduce idle loss; the ML-predicted saving is optimistic relative to the noise-free simulator (optimiser exploits model error). |
| Real data | No model exceeds moderate R²: spindle power depends on cutting load, which is not in the files. Neural network does not beat boosting. K-Means regimes may describe speed/override states but add little predictive value. |
