# Experiment log

How to read this: every experiment answers **Why / Expected / Happened / Better or worse? / Learned / Next**.
"Expected" was written in advance in [`PROTOCOL.md`](PROTOCOL.md). Failed experiments and wrong expectations are kept.

* **S** = synthetic (simulated) data, main engine. **R** = real public data (CFAA, Zenodo 14445879), supporting role.
* Metrics are on the **validation** split (all windows) unless stated: MAE / RMSE in kW, R², MAPE in %. "OOD" = MAE on *engaged
  (cutting)* windows of the extrapolation band `ood_val` (jobs with MRR above anything seen in training).
  Test and `ood_test` are not used for decisions; they appear in E11 only.
* Reference floor: an oracle that knows the true noise-free power still has validation MAE **0.165 kW**, RMSE 0.876 kW
  (glitches), RMSE 0.154 kW on non-glitch rows (E01). Nothing can beat that.
* Raw numbers for every model: `results/tables/all_experiments_long.csv`, `results/tables/leaderboard_*.csv`.

---

## E00 - Physics baselines
**Why.** Set the bar a physical law gives *before* any machine learning. Three references: the mean (null), the 2-parameter
specific-energy model `P = P0 + k·MRR` (Gutowski), and a zero-fit handbook calculation from catalogue constants (no fitting at all).
**Expected.** Mean R² ≈ 0; SEC R² 0.4-0.7 (idle, coolant and speed effects are not in MRR); handbook right order of magnitude but biased.
**Happened (S).**

| model | MAE | RMSE | R² | MAPE | OOD MAE |
|---|---|---|---|---|---|
| mean | 2.895 | 4.097 | 0.000 | 42.0 | 11.01 |
| SEC `P0 + k·MRR` (2 params, fitted) | 1.775 | 2.462 | 0.639 | 26.5 | 3.79 |
| handbook, zero fit | 1.809 | 2.430 | 0.648 | 18.7 | 4.80 |

**Verdict.** Big improvement over the null; expectation met. The unfitted handbook equals the fitted 2-parameter law.
**Learned.** MRR alone explains ~64 % of variance; the rest is fixed/speed/coolant load. Both physics baselines extrapolate roughly linearly (OOD MAE 3.8-4.8), not catastrophically.
**Happened (R).** Real data has no cutting parameters, so the physics baseline is the spindle no-load loss curve `c0 + c1|n| + c2 n²`:
Ibarmia MAE 1.348 → 0.959 (R² 0.197); GMTK 0.906 → 0.619 (R² 0.223). Weak, as expected: load is missing.
**Next.** Look at the data (E01).

## E01 - EDA, data audit, leakage audit
**Why.** Understand structure, find traps (leakage, outliers, label definitions) before fitting.
**Expected.** Cutting power is a modest increment on large fixed loads; V/I/cos φ leak; glitches visible.
**Happened (S).**
* Mean power: standby 2.8 kW, rapid 7.5, air-cut 6.6, cutting 11.6. Inside a cutting window the mean 11.6 kW splits into
  base 2.4 + coolant/conveyor 2.5 + spindle idle loss 1.7 + **actual cutting (incl. copper loss) 4.6** + axes 0.4 kW. Fixed/speed-dependent
  loads are ~60 % of the power of a window in which metal is being cut.
* 4.4 % of windows carry an injected inefficiency. 0.74 % of raw windows were meter glitches; rule R1 (power < 1 kW, physically impossible) removed the 60 dropouts, leaving 0.45 % spike glitches that no physical rule can identify.
* Leakage audit (linear model, validation): commanded features only R² **0.713**; + drive load signals **0.941**;
  + V, I, cos φ **0.953**; the identity √3·V·I·cos φ alone gives 0.954 (the gap to 1.0 is meter noise + glitches).
  → V/I/cos φ are the target in disguise; the drive load signals are nearly as revealing. Both are **excluded** from planning models.
**Happened (R).** The `power_consumption` Low/Medium/High label is *not* a function of spindle power (class means explain R² 0.19 / 0.11):
its definition is undocumented, so it is excluded. 13.4 % exact duplicates; rows are shuffled; GMTK spindle speed is within ±100 rpm for
99.6 % of rows (not rotating, or different units); 1.7 % / 5.5 % of spindle-power values are slightly negative.
**Verdict.** Expectation met (the leakage is only ≈ 0.95 rather than 1.0 because of noise).
**Learned.** Any R² above ~0.95 with electrical inputs is meaningless; planning models must use commanded conditions.
**Next.** Simplest learners: linear and polynomial.

## E02 - Linear and polynomial regression
**Why.** Test whether the problem is low-order. **Expected.** Linear fails (multiplicative MRR); polynomial helps in-distribution but extrapolates badly.
**Happened (S).**

| model | train MAE | val MAE | RMSE | R² | MAPE | OOD MAE |
|---|---|---|---|---|---|---|
| ridge linear | 1.398 | 1.492 | 2.194 | 0.713 | 17.4 | 5.59 (R² −0.94) |
| ridge poly-2 | 0.678 | 0.705 | 1.398 | 0.884 | 7.2 | 2.50 |
| ridge poly-3 | 0.509 | 0.607 | 1.305 | 0.899 | 6.2 | 2.21 |

**Verdict.** Linear fails as expected (worse than the 2-parameter SEC baseline on OOD). Polynomials help a lot. OOD error is 3-4× the in-distribution error, yet polynomials extrapolate *better* than trees later (E04-E06): a smooth global form fails gracefully.
**Learned.** The relationship is low-order but multiplicative. **Happened (R).** Linear/poly on 5 raw inputs: Ibarmia R² 0.19-0.20 (no gain from polynomials); GMTK poly-2 R² 0.50 but poly-3 collapses to 0.04 (overfitting to outliers).
**Next.** Are there discrete operating regimes (E03)?

## E03 - K-Means operating regimes
**Why.** Industrial processes have regimes (idle, positioning, cutting). If K-Means finds them, per-regime models may fit better.
Fitted on *train only*, on unsupervised descriptors (speed, axis speed, log MRR, coolant mode); the true state is used only to grade clusters.
**Expected.** k≈4 recovers the four states with high ARI; regime features help linear/poly, not trees.
**Happened (S).**
* Silhouette generally rises with k (0.35 at k=3, 0.50 at k=10, its maximum; no elbow), i.e. it does **not** pick the physical k=4. For k=4: ARI **0.39**, NMI 0.56, purity 0.85.
  *Standby* and *rapid* windows are recovered perfectly (own clusters); *air-cut* and *cutting* windows are split by spindle-speed band and mixed (see `results/figures/e03_confusion_k4_synthetic.png`).
* Effect on prediction (val MAE):

| base model | global | + regime one-hot (k=4 / k=10) | per-regime models (k=4 / k=10) |
|---|---|---|---|
| ridge linear | 1.492 | 1.411 / 1.375 | 1.166 / **0.904** |
| ridge poly-2 | 0.704 | 0.682 / **0.645** | 0.689 / 0.771 |
| LightGBM (default) | 0.684 | 0.695 / 0.701 | 0.733 / 0.837 |

**Verdict.** Expectation about "high ARI at k=4" was **wrong**: K-Means finds speed bands, not machine states. Regimes help the weakest model a lot (linear: −39 % MAE at k=10), the polynomial a little, and **hurt** boosting (per-regime models starve on data).
**Learned.** Regime structure is real but mostly reflects *which speed/material group the job is in*; a flexible model learns that itself. K-Means is useful as EDA, not as a route to better accuracy here.
**Happened (R).** Clusters explain only 18-20 % of power variance (Ibarmia k=9, GMTK k=2 by silhouette). Regime models: Ibarmia no gain (linear 0.985 → 0.979, LightGBM 0.613 → 0.616); GMTK poly-2 per-cluster 0.663 → 0.461 (a tiny high-power cluster of 106 rows is isolated), LightGBM 0.329 → 0.304.
**Next.** Non-linear learners: trees.

## E04 - Decision tree · E05 - Random forest · E06 - XGBoost / LightGBM
**Why.** Standard non-parametric learners; test whether flexibility beats polynomials. Hyper-parameters tuned on train only (grouped 4-fold CV).
**Expected.** Tree < forest < boosting in-distribution; all flat-line outside the training range.
**Happened (S).**

| model | train MAE | val MAE | RMSE | R² | MAPE | OOD MAE |
|---|---|---|---|---|---|---|
| decision tree, unconstrained | **0.000** | 1.274 | 2.550 | 0.613 | 12.0 | 4.07 |
| decision tree, tuned | 0.712 | 1.033 | 1.888 | 0.788 | 9.7 | 3.53 |
| random forest, default | 0.113 | 0.855 | 1.625 | 0.843 | 8.1 | 3.22 |
| random forest, tuned | 0.137 | 0.833 | 1.575 | 0.852 | 7.9 | 3.53 |
| XGBoost, default / tuned | 0.225 / 0.326 | 0.861 / **0.672** | 1.608 / 1.387 | 0.846 / 0.885 | 8.5 / 6.4 | 2.97 / 2.83 |
| LightGBM, default / tuned | 0.340 / 0.440 | 0.692 / **0.665** | 1.413 / 1.380 | 0.881 / 0.887 | 6.5 / 6.4 | 2.83 / 2.98 |
| LightGBM tuned, Huber / L1 loss | 0.694 / 0.493 | 0.842 / 0.683 | 1.844 / 1.545 | 0.797 / 0.858 | 7.0 / 5.8 | 5.95 / 4.10 |

**Verdict.** Ordering as expected, but the best tree model (MAE 0.665) is **worse than cubic polynomial regression (0.607)** and far worse than physics-structured models later. Unconstrained tree memorises (train MAE 0) - kept as the overfitting demonstration. Robust losses did not help (worse on both splits). OOD error is 3-5× in-distribution for every tree model (flat-line extrapolation).
**Learned.** On a smooth, physically structured target with ~14 k rows, generic tree ensembles are not the strongest tool; boosting needs the structure handed to it (E07). Tuning moved LightGBM MAE 0.692 → 0.665 only.
**Happened (R).** Trees are *much* better than linear: Ibarmia RF/LightGBM val R² 0.74-0.75 (MAE ≈ 0.51) vs 0.20 linear; GMTK RF R² 0.81. **This is suspicious; see E14.**
**Next.** Hand the physics to the models (E07).

## E07 - Feature engineering and re-tuning
**Why.** Does giving models process knowledge help more than tuning? Levels: raw commanded → generic process features (feed velocity, MRR, cutting speed, flags) → physics features (Kienzle power, torque, idle loss, coolant, wear fraction, warm-up decay; catalogue constants).
**Expected.** Physics features help all; linear most; trees mostly in-distribution.
**Happened (S), val MAE (OOD MAE):**

| model | raw commanded | + generic FE | + physics FE |
|---|---|---|---|
| ridge linear | 1.492 (5.59) | 1.009 (3.10) | **0.562** (2.06) |
| ridge poly-2 | 0.705 (2.50) | 0.514 (2.08) | **0.462** (1.61) |
| random forest (tuned params) | 0.833 (3.53) | 0.676 (2.52) | 0.578 (2.01) |
| LightGBM default | 0.692 (2.83) | 0.584 (2.21) | 0.515 (2.05) |
| LightGBM tuned params | 0.665 (2.98) | 0.555 (2.17) | 0.481 (1.89) |
| LightGBM re-tuned on physics FE | - | - | 0.486 (1.95) (**no gain** over reusing params) |

Physics FE on the best variants: val RMSE 1.157-1.238, R² 0.909-0.920, MAPE 4.6-5.9 %.
**Diagnostic (not a planning model).** Adding the drive's own load signals to LightGBM: MAE 0.366, R² 0.937, OOD 1.14 - a *monitoring* gain that would be unusable for choosing operating conditions.
**Verdict.** Feature engineering beat tuning by a wide margin (LightGBM: tuning −4 %, physics features −28 %). Expectation met for ordering; **wrong** that trees would only gain in-distribution: OOD error also fell (2.98 → 1.89).
**Learned.** Encoding the known structure is worth more than any search over hyper-parameters, and the gain reaches extrapolation. Linear + physics FE (MAE 0.562) now beats tuned boosting on raw features.
**Happened (R).** Engineered real inputs (|speed|, speed², |power_Z|): Ibarmia nothing changes (ridge 0.198 R²; RF 0.747); GMTK ridge R² 0.37 → 0.67 (|power_Z| is informative).
**Next.** Neural network (E08), explainability (E09), then a physics-structured model (E10).

## E08 - Neural network (PyTorch MLP)
**Why.** Test whether deep learning actually helps. MLP, early stopping on a held-out slice of *training* jobs, 8-config random search on that slice, **5 seeds**, same features and splits as the classical models.
**Expected.** About equal to tuned boosting in-distribution, high seed variance, no clear win.
**Happened (S).** Best config both times: 2 hidden layers × 128, lr 1e-3, no dropout.

| model | val MAE (±seed std) | RMSE | R² | MAPE | OOD MAE |
|---|---|---|---|---|---|
| MLP, raw commanded | 0.616 ± 0.022 | 1.349 | 0.892 | 6.0 | 2.46 ± 0.15 |
| 5-seed ensemble | 0.559 | 1.311 | 0.898 | 5.3 | 2.37 |
| MLP, physics features | 0.485 ± 0.025 | 1.176 | 0.918 | 4.8 | 1.63 ± 0.07 |
| 5-seed ensemble | 0.449 | 1.158 | 0.920 | 4.3 | 1.59 |
| *for comparison:* LightGBM tuned, raw / physics FE | 0.665 / 0.481 | 1.380 / 1.178 | 0.887 / 0.917 | 6.4 / 4.6 | 2.98 / 1.89 |
| *for comparison:* fitted physics equation (E10) | **0.377** | 1.161 | 0.920 | 3.2 | **1.43** |

**Verdict.** Expectation partly **wrong**: on raw features the MLP *does* beat tuned boosting and random forests (MAE 0.616 vs 0.665 / 0.833; better OOD too) - a modest, seed-stable win. With physics features they tie (0.485 vs 0.481). It never beats the physics-structured equation (0.377), and it needs 5× the engineering (search, seeds, scaling).
**Learned.** Smooth function + continuous inputs is the regime where small MLPs are competitive. Deep learning is *not* worse than classical ML here, but it is not better than knowing the physics.
**Happened (R).** Clear negative result: Ibarmia MLP MAE 0.686 (R² 0.60; ensemble 0.645) and 0.641 with engineered inputs vs 0.505 for tuned LightGBM; GMTK ensemble 0.256-0.276 vs RF 0.221. With ~3 k training rows DL loses to trees.
**Next.** Explain what the models learned (E09).

## E09 - Explainability (SHAP + permutation importance)
**Why.** Check that the model uses the physically right variables. SHAP TreeExplainer on the tuned LightGBM (raw commanded features, 3 000 validation rows); compare group importances with the generator's *true* power components.
**Expected.** Top drivers: spindle speed, coolant, cutting-power/MRR variables, material; group shares resemble the truth.
**Happened (S).** Mean |SHAP| [kW]: axial depth 1.52, radial width 1.22, coolant mode 0.88, axis speed 0.80, spindle speed 0.54, material C45 0.38, flutes 0.19. Permutation importance gives the same ranking (Spearman ρ = 0.99). On physics features: handbook cutting power 1.57, coolant 0.63, MRR 0.53, spindle speed 0.41.

| physical group | SHAP share | true share |
|---|---|---|
| cutting load (depths, feed, tool, material, wear) | 51.0 % | 52.6 % |
| coolant | 19.0 % | 24.6 % |
| spindle speed / idle loss | 11.6 % | 17.1 % |
| feed axes | **17.3 %** | **4.3 %** |
| ambient / base | 1.2 % | 1.4 % |

**Verdict.** Mostly met: depth of cut and width of cut (not speed) dominate - the ordering I wrote in advance was off. Group shares track the truth except *feed axes*, which SHAP over-credits because `axis_speed` doubles as a machine-state indicator (24 000 mm/min = rapid): correlated inputs share credit.
**Learned.** SHAP recovers the physics well enough to trust, but group-level attribution is blurred by proxy features; use the physics-feature model (cutting power ranks first) for cleaner stories.
**Happened (R).** Ibarmia: speed 0.97, power_Z 0.22, load_X 0.22, load_Z 0.19. GMTK: power_Z 0.61, speed 0.35 (on a machine whose "speed" barely moves).
**Next.** Build the model with the physics as its skeleton (E10).

## E10 - Physics-structured and hybrid models
**Why.** The brief is to understand the physical relationship, not maximise R². Fit the governing equation (Section 4 of the README: 21 coefficients - base load, ambient, coolant steps, conveyor, idle-loss polynomial, warm-up, copper loss, per-material Kienzle `k_c1.1` and `m_c`, wear slope) by robust non-linear least squares, **starting from catalogue values, not the generator's truth**. Then try correcting it with ML.
**Expected.** Competitive in-distribution, best out-of-distribution; a residual hybrid adds a small in-distribution gain.
**Happened (S).**

| model | val MAE | RMSE | R² | MAPE | OOD MAE |
|---|---|---|---|---|---|
| **physics-parametric (robust loss)** | **0.377** | 1.161 | 0.920 | 3.2 | **1.43** |
| same, ordinary least squares | 0.413 | 1.140 | 0.923 | 3.8 | 1.46 |
| hybrid: physics + LightGBM on residuals | 0.419 | 1.149 | 0.921 | 3.9 | 1.48 |
| hybrid: LightGBM given physics prediction as a feature | 0.458 | 1.190 | 0.916 | 4.3 | 1.78 |

Parameter recovery (`results/tables/e10_param_recovery.csv`): 15 of 21 coefficients within ±10 % of the generator's effective values (base+servo 0.5 %, flood/high-pressure coolant 2 %/0.5 %, warm-up τ −5.5 %, wear slope 2 %, `k_c1.1` Al +7 % / C45 −5 %); weakly identified ones: copper loss +56 %, `m_c` Ti +19 % / SS316L +15 %, `k_c1.1` Ti −16 % / SS316L −11 % (`k_c1.1` and `m_c` trade off against each other).
**Verdict.** Expectation met for the physics model (best model in-distribution *and* out-of-distribution). The residual hybrid expectation was **wrong**: letting ML correct the equation made it *worse* (0.377 → 0.419), the stacked variant worse still (0.458). Robust loss helped MAE (0.413 → 0.377) and cost a little RMSE.
**Learned.** With 14 k noisy rows and glitches, the 21-parameter structural model is already at the point where extra flexibility fits noise. **Caveat that matters:** the synthetic world is generated from the same functional forms, so this experiment shows that the structure is *recoverable from noisy data by a method started from imperfect catalogue values* - it does not show that physics models beat ML on real machines.
**Happened (R).** No cutting parameters, so no physics model is possible; a no-load-curve + LightGBM-residual hybrid gets Ibarmia MAE 0.629 / GMTK 0.331 (worse than plain tuned trees).
**Next.** Apply the frozen selection rule (E11).

## E11 - Final model selection (frozen rule, PROTOCOL §4)
**Rule.** `score = 0.5·MAE_val/min + 0.5·MAE_oodval_engaged/min`; among models within 2 % of the best score pick the simplest. Pool = all 50 synthetic model entries from E00-E10 (diagnostic monitoring model excluded).
**Result.** Best score and winner: **`physics_parametric_fitted`** (score 1.000). Runners-up: same model with squared loss 1.058; residual hybrid 1.072; MLP-physics-FE ensemble 1.150; ridge poly-2 physics FE 1.175; best boosting (physics FE) 1.298; best random forest 1.468; best decision tree far behind. Nothing else fell within the 2 % tie band, so complexity did not matter.
**One-shot test evaluation** (refit on train+val, hyper-parameters unchanged; `test` = 440 unseen jobs, `ood_test` = 150 jobs with MRR above everything seen):

| split | subset | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|---|
| test | all windows | **0.365** | 1.010 | **0.931** | 3.29 |
| test | non-glitch rows (diagnostic) | 0.322 | 0.681 | 0.967 | 3.06 |
| ood_test | engaged windows | 1.292 | 2.327 | 0.667 | 6.65 |

Reference points on the same `ood_test` engaged windows (train-only models): MLP raw-feature ensemble MAE 2.06 (R² 0.40), ridge poly-3 raw 2.11 (R² 0.40), zero-fit handbook 2.31 (R² 0.24), LightGBM physics FE 2.35 (R² 0.34), ridge poly-2 physics FE 2.87 (R² 0.05), MLP physics-FE ensemble 3.08 (R² 0.04), hybrid 1.38 (R² 0.65), best tree 4.2-6.1 (R² < 0). The ordering chosen on `ood_val` was confirmed on the untouched `ood_test` band.
**Learned.** Test MAE is 2.2× the noise floor (0.165); the remainder is hidden job-level factors (lot scatter, tool-specific life, injected inefficiencies) that cannot be predicted from commanded conditions. Extrapolation is where structure pays: ML models that look equal in-distribution (R² 0.92-0.93 on `test`) differ hugely beyond the training range (R² 0.04-0.67).

## E12 - Energy-residual anomaly detection
**Why.** `residual = actual − predicted` should expose windows that use far more energy than their commanded conditions justify - *potential inefficiency*, not confirmed faults. The final model is applied **out-of-fold** (5-fold, grouped by job) so a window is never scored by a model that saw it. Score: residual divided by a robust (MAD) scale estimated in bins of predicted power; flag if `z > 3.5` and excess > 0.4 kW. (The 0.4 kW practical-significance threshold equals the generator's labelling threshold by design; without it precision would fall.)
**Expected.** High recall for large additive losses, misses subtle ones, glitch spikes create false positives.
**Happened (S, 23 678 in-distribution windows, 4.3 % truly inefficient).**

| | precision | recall | F1 | average precision |
|---|---|---|---|---|
| windows, in-distribution (OOF) | 0.815 | 0.906 | 0.858 | 0.723 (chance 0.043) |
| windows, extrapolation jobs (`ood`) | 0.697 | 0.610 | 0.651 | 0.666 |
| jobs (≥ 50 % of cutting windows flagged) | 0.960 | 0.681 | - | - |

Recall by injected cause: stuck high-pressure coolant 0.94, aux leak 0.92, bearing friction 0.90, excess tool wear 0.82, axis drag 1.00 (only 13 windows). Of the 207 false positives, 106 are meter glitches (unavoidable without a glitch filter) and 101 are ordinary noise. The flagged windows hold 5.97 of the 6.66 kWh of true excess energy (90 %). Threshold sweep: z > 2 → P 0.63 / R 0.96; z > 5 → P 0.85 / R 0.75.
**Verdict.** Expectation partly wrong: recall was high even for "subtle" causes because every injected inefficiency is ≥ 0.4 kW by construction; the genuinely hard case (tool wear, partly absorbed by the model's own wear term) is the lowest at 0.82. Recall drops to 0.61 on extrapolation jobs, where the model's own error rises - residual detection is only as good as the predictor in that region.
**Learned.** The residual is a usable screening signal; job-level aggregation gives 96 % precision. Detection quality has to be re-earned per operating range.
**Happened (R).** No ground truth. Ibarmia (tuned LightGBM, OOF): 148 rows (2.85 %) flagged, their median Z-axis power is 43.7 vs 6.6 for all rows; GMTK (random forest): 132 rows (2.54 %), Z-axis power −59 vs −30. These are *candidates* for inspection (`results/tables/e12_flagged_windows_real_*.csv`); the model may simply be mis-fitting rare `power_Z` combinations (E14).

## E13 - Optimisation (all savings are PREDICTED or SIMULATED)
**Why.** Minimise predicted energy of a milling operation subject to production and machine constraints; compare current and recommended conditions.
**Formulation.** Per operation: variables spindle speed, feed per tooth, axial depth, radial width. Job fixed: material, tool, coolant, volume to remove `V`, context. Minimise the energy over the **fixed takt window** `E_win = E_op + P_standby·(t_window − t_op)` where `E_op = P_cut·t_cut + N_pass·t_pos·P_rapid` (states predicted by the model). Constraints: operation time ≤ takt (same part, same time); spindle power/torque with catalogue `k_c` and 1.25× margin ≤ 85 % of the limit at that speed; tool life ≥ 1.2× cutting time; cutting-speed window of the material; feed/depth/width ranges of the training data; MRR ≤ training maximum (196 cm³/min; the *trust region*). Search: Sobol sampling + two local refinements. **Current conditions** = the recorded settings of 440 unseen test jobs.
**A first attempt was wrong and is kept here:** counting only `E_op` gave a 58.6 % mean saving because finishing early looked free. Once the machine's standby energy for the remaining takt is included, the headline falls to **37 %**. Both are reported; only the window version is a defensible headline.
**Expected.** Savings mostly from higher MRR (amortising fixed loads); ML-predicted saving optimistic vs the simulator.
**Happened (S, 440 unseen jobs, final model).**

| | mean saving (simulator) | 95 % CI | model's own prediction | median | P10-P90 |
|---|---|---|---|---|---|
| energy over the takt window (headline) | **37.2 %** | 35.8-38.6 | 39.6 % | 37.9 % | 17.8-55.7 % |
| operation-only (upper bound) | 58.6 % | 56.4-60.8 | 60.8 % | 60.8 % | 24.3-88.5 % |

All 440 jobs improve; total 1 521 → 994 kWh (−34.7 %). The recommended settings run the operation in **34 %** of the baseline time (median MRR ×3.9) with a lower spindle speed (median ×0.79, which cuts the idle loss) but a larger chip load (median: feed per tooth ×1.66, axial depth ×1.49, radial width ×1.78); no recommendation overloads the *true* spindle in the simulator. By material: Al6061 38.6 %, C45 33.0 %, SS316L 34.4 %, Ti6Al4V 43.7 %. Constraint activity at the optimum: trust-region MRR limit 42 %, spindle power 10 %, tool life 0 %. 12 % of baselines already violate at least one optimiser constraint.

Which model finds the optimum that *actually* works (120 jobs, window objective; predicted vs simulated saving):

| model used by the optimiser | in-support: predicted / simulated | extended range (MRR to machine limit): predicted / simulated |
|---|---|---|
| physics-parametric (final) | 39.1 / 36.8 | 40.5 / 38.4 |
| handbook, zero fit | 37.1 / 37.6 | 38.9 / 39.4 |
| MLP ensemble | 39.1 / 36.9 | 41.5 / 38.9 |
| ridge poly-3 | 41.7 / 36.2 | 42.7 / 38.0 |
| LightGBM tuned | 41.8 / 35.1 | 45.8 / 38.3 |
| random forest tuned | 44.0 / 35.1 | 47.9 / 38.3 |
| SEC `P0 + k·MRR` | 0.0 / 13.6 | 0.0 / 14.1 |

Relaxing the takt (×1.25 / ×1.5 / ×2) lowers the relative saving 36.8 → 34.2 → 32.0 → 28.3 % because both schedules idle longer.
**Verdict.** Expectation met: the saving comes from compressing cycle time (higher MRR), and tree-based optimisers are over-optimistic by 6-9 points while the physics model is within 2.3. All reasonable models reach similar *realised* savings (35-38 %) - the gain is a property of the problem, the model mainly tells you how much to believe it. The 2-parameter SEC model is a bad optimiser (it believes nothing can be gained).
**Learned.** The headline is a **simulation result under a deliberately broad baseline** (randomly sampled practice, often far below the machine's capability). Real baselines are already constrained by chatter, surface finish, fixture rigidity and tool-change rules that this simulation omits, so the number is an upper bound on what the simulated machine allows, not a forecast for a plant. Only *relative* conclusions are robust: raise MRR within power/tool-life limits because fixed loads dominate; do not trust tree models to pick operating points near or beyond the edge of their data.

## E14 - Reliability probes on the real data
**Why.** E04-E06 showed trees reaching R² 0.74-0.85 on real data where linear models get 0.2 - with no run IDs to prevent neighbouring rows of the same cut sitting in train and test.
**Happened (R).**

| probe | Ibarmia | GMTK |
|---|---|---|
| tuned LightGBM, 5 inputs, random split, val R² | 0.744 | 0.733 |
| 1-nearest-neighbour regressor, val R² | 0.643 | 0.717 |
| validation rows with a training neighbour closer than 0.05 std | 84 % | 70 % |
| 5-fold CV, random, pooled R² | 0.846 | 0.851 |
| 5-fold CV, **grouped by 30 input-space clusters**, pooled R² | **−0.130** | **−1.429** |
| val R², speed + override only | 0.365 | 0.344 |
| val R², + both axis loads | 0.707 | 0.770 |

**Verdict.** The high real-data scores are mostly **interpolation between near-identical records** (same cut, repeated samples), not a learned relationship: predict a new region of input space and R² goes negative. This downgrades every tree/NN/anomaly conclusion on the real data to "descriptive only".
**Learned.** With unshuffled data, timestamps and run IDs this would be checkable; without them the real dataset can support a sanity check on the *shape* of spindle-power versus speed, not a predictive claim.

---

## Scorecard of the pre-registered expectations

| Exp | Expectation | Outcome |
|---|---|---|
| E00 | SEC R² 0.4-0.7; handbook biased | **met** (0.64; handbook 0.65) |
| E01 | V/I/cos φ leak | **met** (R² 0.95 limited by noise) |
| E02 | linear fails, poly better ID, poor OOD | **met**; poly extrapolates better than trees |
| E03 | k≈4 recovers states (high ARI) | **not met** (ARI 0.39; silhouette picks k=10) |
| E03 | regimes help linear, not trees | **met** (linear −39 % MAE; boosting worse) |
| E04-06 | tree < forest < boosting; flat OOD | **met**, but all lose to polynomials and physics |
| E07 | physics features help most for linear; trees mainly ID | **partly** (linear most; trees' OOD also improved) |
| E08 | MLP ≈ boosting, no clear win | **partly not met**: MLP beat trees on raw features, tied on physics features; lost to physics |
| E09 | drivers: speed, coolant, MRR, material | **partly** (depth/width of cut ranked first) |
| E10 | physics best OOD; residual hybrid small gain | **physics met; hybrid gain not met** (hybrid worse) |
| E12 | subtle causes missed | **not met** (recall ≥ 0.82 for all; injected effects ≥ 0.4 kW) |
| E13 | gain from MRR; ML optimistic | **met** (trees +6-9 points, physics +2) |
| Real | moderate R² at best; NN ≯ boosting; regimes add little | **mostly met**, but R² was high for a bad reason (E14) |
