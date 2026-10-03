"""E13 - Constrained minimisation of predicted energy.  ALL savings are PREDICTED or SIMULATED, not measured.
python -m experiments.e13_optimization"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import config as C, data_prep as dp, optimize as O, plotstyle as ps

OUT = {}
COMPARE = ["handbook_physics_zero_fit", "sec_P0+k*MRR_fitted", "ridge_poly3", "random_forest_tuned", "lightgbm_tuned",
           "mlp_commanded_ensemble5", "physics_parametric_fitted", "hybrid_physics+lgbm_residual"]
TAG = "SIMULATED / PREDICTED - not measured savings"


def run_jobs(model, jobs, **kw):
    rows = Parallel(n_jobs=8)(delayed(O.optimise_job)(model, j, **kw) for _, j in jobs.iterrows())
    r = pd.DataFrame(rows)
    out = pd.concat([jobs.reset_index(drop=True), r], axis=1)
    out["save_pred_pct"] = 100 * (out.e_pred_base - out.e_pred_opt) / out.e_pred_base
    out["save_true_pct"] = 100 * (out.e_true_base - out.e_true_opt) / out.e_true_base
    out["save_true_op_pct"] = 100 * (out.eop_true_base - out.eop_true_opt) / out.eop_true_base
    return out


def summ(r, label):
    rng = np.random.default_rng(C.SEED)
    boot = [rng.choice(r.save_true_pct, len(r)).mean() for _ in range(2000)]
    return dict(label=label, jobs=len(r), mean_saving_true_pct=float(r.save_true_pct.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                median_saving_true_pct=float(r.save_true_pct.median()), p10=float(r.save_true_pct.quantile(.1)), p90=float(r.save_true_pct.quantile(.9)),
                mean_saving_pred_pct=float(r.save_pred_pct.mean()), mean_saving_true_operation_only_pct=float(r.save_true_op_pct.mean()),
                share_power_limit_active=float((r.power_util_opt > 0.97).mean()), share_tool_life_limit_active=float((r.life_ratio_opt < 1.03).mean()),
                share_trust_limit_active=float((r.trust_ratio_opt > 0.97).mean()), mean_vars_at_upper_bound=float(r.at_upper_bound.mean()),
                mean_time_ratio_opt_over_base=float((r.t_op_opt_min / r.t_op_base_min).mean()), share_improved_true=float((r.save_true_pct > 0.5).mean()),
                share_worse_true=float((r.save_true_pct < -0.5).mean()), baseline_infeasible_share=float((~r.baseline_feasible).mean()),
                no_feasible_found_share=float((~r.feasible_found).mean()), true_overload_opt_share=float(r.true_overload_opt.mean()),
                true_overload_base_share=float(r.true_overload_base.mean()), total_kwh_base_true=float(r.e_true_base.sum()), total_kwh_opt_true=float(r.e_true_opt.sum()),
                total_saving_true_pct=float(100 * (1 - r.e_true_opt.sum() / r.e_true_base.sum())), mrr_ratio_median=float((r.mrr_opt_cm3 / r.mrr_base_cm3_min).median()))


def fig_parameter_shift(A):
    fig, axs = plt.subplots(1, 5, figsize=(14, 3.0))
    for ax, (col_o, col_b, ttl) in zip(axs, [("n_opt", "n_base", "spindle speed"), ("fz_opt", "fz_base", "feed per tooth"), ("ap_opt", "ap_base", "axial depth"),
                                             ("ae_opt", "ae_base", "radial width"), ("mrr_opt_cm3", "mrr_base_cm3_min", "removal rate (MRR)")]):
        ratio = (A[col_o] / A[col_b]).clip(0.2, 5)
        ax.hist(ratio, bins=40, color=ps.PALETTE[0], alpha=0.85); ax.axvline(1, color=ps.INK2, linestyle="--", linewidth=1.2)
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("ratio")
    fig.suptitle(TAG + " - ratio of recommended to current value (clipped at 5; dashed line = unchanged)", x=0.01, ha="left", fontsize=9, color=ps.INK2, y=1.04)
    fig.savefig(C.FIGURES / "e13_parameter_shift.png"); plt.close(fig)


def fig_model_comparison(cdf):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, regime in zip(axs, cdf.regime.unique()):
        s = cdf[cdf.regime == regime].reset_index(drop=True)
        y = np.arange(len(s))
        ax.barh(y - 0.19, s.mean_saving_pred_pct, 0.36, color=ps.PALETTE[1], label="what the model predicted")
        ax.barh(y + 0.19, s.mean_saving_true_pct, 0.36, color=ps.PALETTE[0], label="what the simulator says")
        ax.set_yticks(y, [m.replace("__FINAL", " (FINAL)") for m in s.model]); ax.invert_yaxis(); ax.axvline(0, color=ps.INK2, linewidth=1)
        ax.set_title(regime); ax.set_xlabel("mean energy reduction [%]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(TAG, x=0.01, ha="left", fontsize=9, color=ps.INK2)
    fig.savefig(C.FIGURES / "e13_model_comparison.png"); plt.close(fig)


def main():
    d = dp.load_synthetic()
    jobs = O.job_specs(d)
    trust = float(d[d.split == "train"].job_mrr_cm3_min.max())
    OUT["trust_region_mrr_cm3_min"] = trust
    test_jobs = jobs[jobs.split == "test"].reset_index(drop=True)
    final = joblib.load(C.MODELS / "final_model.joblib")
    OUT["final_model"] = final.name

    # ---- A. main result: final model, inside the training support ----------------------------------------------
    A = run_jobs(final, test_jobs, takt_mult=1.0, trust_mrr_cm3=trust, objective="window")
    A.to_csv(C.TABLES / "e13_job_results_final_model.csv", index=False)
    OUT["A_final_model_window_objective"] = summ(A, "final model, in-support, energy over the fixed takt window (standby included)")
    Aop = run_jobs(final, test_jobs, takt_mult=1.0, trust_mrr_cm3=trust, objective="operation")
    Aop.to_csv(C.TABLES / "e13_job_results_final_model_operation_only.csv", index=False)
    OUT["A_final_model_operation_objective"] = summ(Aop, "final model, in-support, operation-only energy (upper bound)")
    by_mat = A.groupby("material").apply(lambda g: pd.Series(dict(jobs=len(g), mean_true_pct=g.save_true_pct.mean(), mean_pred_pct=g.save_pred_pct.mean(),
                                                                  median_mrr_ratio=(g.mrr_opt_cm3 / g.mrr_base_cm3_min).median())), include_groups=False)
    OUT["A_by_material"] = by_mat.round(2).to_dict("index")
    print(json.dumps(OUT["A_final_model_window_objective"], indent=1)); print(json.dumps(OUT["A_final_model_operation_objective"], indent=1)); print(by_mat.round(2), flush=True)

    # ---- B/C. does the optimum found with each model survive the simulator? -----------------------------------
    sub = test_jobs.sample(min(120, len(test_jobs)), random_state=C.SEED).reset_index(drop=True)
    comp = []
    for name in COMPARE + [final.name + "__FINAL"]:
        mdl = final if name.endswith("__FINAL") else joblib.load(C.CACHE / f"{name}.joblib")
        for regime, tr in [("in-support (MRR <= training max)", trust), ("extended (MRR up to machine limits)", None)]:
            r = run_jobs(mdl, sub, takt_mult=1.0, trust_mrr_cm3=tr, objective="window")
            s = summ(r, f"{name} | {regime}")
            s.update(model=name, regime=regime)
            comp.append(s)
            print(f"{name:42s} {regime:38s} predicted {s['mean_saving_pred_pct']:6.1f}%   simulated {s['mean_saving_true_pct']:6.1f}%  overload {s['true_overload_opt_share']:.2f}")
    OUT["B_C_model_comparison"] = comp
    pd.DataFrame(comp).drop(columns=["ci95"]).to_csv(C.TABLES / "e13_model_comparison.csv", index=False)

    # ---- D. production-requirement sensitivity ---------------------------------------------------------------
    sens = []
    for tm in [1.0, 1.25, 1.5, 2.0]:
        r = run_jobs(final, sub, takt_mult=tm, trust_mrr_cm3=trust, objective="window")
        s = summ(r, f"takt x{tm}"); s["takt_mult"] = tm
        sens.append(s)
        print("takt", tm, round(s["mean_saving_true_pct"], 2), round(s["mean_saving_pred_pct"], 2))
    OUT["D_takt_sensitivity"] = sens

    # ---- figures ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for i, m in enumerate(C.MATERIAL_LIST):
        s = A[A.material == m]
        ax.scatter(s.e_true_base, s.e_true_opt, s=16, alpha=0.7, color=ps.PALETTE[i], marker=ps.MARKERS[i], label=m, linewidths=0)
    lo, hi = A.e_true_base.min() * 0.8, A.e_true_base.max() * 1.2
    ax.plot([lo, hi], [lo, hi], color=ps.INK2, linewidth=1.2, linestyle="--"); ax.text(hi * 0.4, hi * 0.5, "no change", color=ps.INK2, fontsize=9, rotation=35)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("baseline energy over the takt window [kWh]  (simulator)"); ax.set_ylabel("recommended energy over the takt window [kWh]  (simulator)")
    ax.set_title(f"{TAG}\nBaseline vs recommended conditions, {len(A)} unseen jobs"); ax.legend()
    fig.savefig(C.FIGURES / "e13_before_after.png"); plt.close(fig)

    fig_parameter_shift(A)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = [A[A.material == m].save_true_pct for m in C.MATERIAL_LIST]
    bp = ax.boxplot(data, tick_labels=C.MATERIAL_LIST, patch_artist=True, widths=0.5, showfliers=False)
    for p in bp["boxes"]:
        p.set(facecolor=ps.PALETTE[0], alpha=0.35, edgecolor=ps.INK2)
    for l in bp["medians"]:
        l.set(color=ps.INK, linewidth=2)
    ax.axhline(0, color=ps.INK2, linewidth=1); ax.set_ylabel("energy reduction over the takt window [%]  (simulator)")
    ax.set_title(f"{TAG}\nSimulated saving by material")
    fig.savefig(C.FIGURES / "e13_saving_by_material.png"); plt.close(fig)

    fig_model_comparison(pd.DataFrame(comp))

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    sd = pd.DataFrame(sens)
    ax.plot(sd.takt_mult, sd.mean_saving_true_pct, color=ps.PALETTE[0], marker="o", markersize=6, label="simulator")
    ax.plot(sd.takt_mult, sd.mean_saving_pred_pct, color=ps.PALETTE[1], marker="s", markersize=6, label="model prediction")
    ax.set_xlabel("allowed cycle time / baseline cycle time"); ax.set_ylabel("mean energy reduction [%]"); ax.legend()
    ax.set_title(f"{TAG}\nTrade-off between cycle time and energy")
    fig.savefig(C.FIGURES / "e13_takt_tradeoff.png"); plt.close(fig)

    (C.EXP_RESULTS / "E13_optimization.json").write_text(json.dumps(OUT, indent=1, default=str))


if __name__ == "__main__":
    main()
