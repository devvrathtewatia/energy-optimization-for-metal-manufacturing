"""E09 - Explainability: SHAP (TreeExplainer) + permutation importance, compared with the generator's true
power decomposition.  python -m experiments.e09_explainability"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from src import config as C, data_prep as dp, features as F, plotstyle as ps

OUT = {}
GROUPS = {
    "spindle speed / idle loss": ["spindle_speed_rpm", "machine_on_min"],
    "cutting load": ["feed_per_tooth_mm", "axial_depth_mm", "radial_width_mm", "tool_diameter_mm", "n_flutes", "tool_age_min",
                     "mat_Al6061", "mat_C45", "mat_SS316L", "mat_Ti6Al4V"],
    "coolant": ["coolant_mode"],
    "feed axes": ["axis_speed_mm_min"],
    "ambient / base load": ["ambient_temp_c"],
}
TRUTH = {"spindle speed / idle loss": "gt_p_idle", "cutting load": "gt_p_cut", "coolant": "gt_p_coolant",
         "feed axes": "gt_p_axes", "ambient / base load": "gt_p_base"}


def bar(ax, names, vals, color=ps.PALETTE[0]):
    ax.barh(names[::-1], vals[::-1], color=color, height=0.6)
    ax.set_xlabel("mean |SHAP|  [kW]")


def explain(model_name, d, n=3000, tag=""):
    m = joblib.load(C.CACHE / f"{model_name}.joblib")
    s = d[d.split == "val"].sample(n, random_state=C.SEED)
    X = m.fb(s)[m.cols]
    sv = shap.TreeExplainer(m.estimator).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
    return m, s, X, sv, imp


def synthetic():
    d = dp.load_synthetic()
    m, s, X, sv, imp = explain("lightgbm_tuned", d)
    OUT["synthetic_top_features"] = imp.round(4).head(12).to_dict()

    plt.figure(figsize=(7.5, 5))
    shap.summary_plot(sv, X, show=False, max_display=12, plot_size=None)
    plt.title("SHAP summary - tuned LightGBM, commanded features (synthetic)")
    plt.savefig(C.FIGURES / "e09_shap_beeswarm_synthetic.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bar(ax, list(imp.index[:12]), imp.values[:12])
    ax.set_title("Mean |SHAP| (synthetic, commanded features)")
    fig.savefig(C.FIGURES / "e09_shap_bar_synthetic.png"); plt.close(fig)

    for feat, color in [("spindle_speed_rpm", "coolant_mode"), ("axial_depth_mm", "material") if False else ("axial_depth_mm", "radial_width_mm")]:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(feat, sv, X, interaction_index=color, show=False)
        plt.savefig(C.FIGURES / f"e09_shap_dependence_{feat}.png"); plt.close()

    # ---- group-wise SHAP vs the generator's true decomposition ------------------------------
    cols = list(X.columns)
    g_shap = {g: np.abs(sv[:, [cols.index(c) for c in cs]].sum(1)).mean() for g, cs in GROUPS.items()}
    g_true = {g: np.abs(s[t] - s[t].mean()).mean() for g, t in TRUTH.items()}
    tab = pd.DataFrame({"shap_mean_abs_kw": g_shap, "true_component_mean_abs_dev_kw": g_true})
    tab["shap_share"] = tab.shap_mean_abs_kw / tab.shap_mean_abs_kw.sum()
    tab["true_share"] = tab.true_component_mean_abs_dev_kw / tab.true_component_mean_abs_dev_kw.sum()
    tab.round(4).to_csv(C.TABLES / "e09_shap_groups_vs_truth.csv")
    OUT["group_shares"] = tab.round(4).to_dict("index")
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(tab))
    ax.barh(y - 0.19, tab.true_share, 0.36, color=ps.PALETTE[1], label="true power component (generator)")
    ax.barh(y + 0.19, tab.shap_share, 0.36, color=ps.PALETTE[0], label="SHAP group importance (model)")
    ax.set_yticks(y, tab.index); ax.invert_yaxis(); ax.set_xlabel("share of total variation attributed"); ax.legend(loc="lower right")
    ax.set_title("Does SHAP recover the physical drivers?")
    fig.savefig(C.FIGURES / "e09_shap_vs_truth_groups.png"); plt.close(fig)

    # ---- permutation importance (second method) -----------------------------------------------
    pi = permutation_importance(m.estimator, X, s.power_kw, n_repeats=5, random_state=C.SEED, scoring="neg_mean_absolute_error", n_jobs=4)
    pimp = pd.Series(pi.importances_mean, index=X.columns)
    rho = float(spearmanr(imp.reindex(X.columns), pimp.reindex(X.columns)).statistic)
    OUT["permutation_top"] = pimp.sort_values(ascending=False).round(4).head(8).to_dict()
    OUT["spearman_shap_vs_permutation"] = rho

    # ---- physics-feature model -----------------------------------------------------------------
    m2, s2, X2, sv2, imp2 = explain("lightgbm_retuned__physics_fe", d)
    OUT["synthetic_physics_fe_top_features"] = imp2.round(4).head(12).to_dict()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bar(ax, list(imp2.index[:12]), imp2.values[:12], ps.PALETTE[2])
    ax.set_title("Mean |SHAP| - LightGBM on physics features")
    fig.savefig(C.FIGURES / "e09_shap_bar_physics_fe_synthetic.png"); plt.close(fig)


def real():
    import lightgbm as lgb
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axs, ["ibarmia", "gmtk"]):
        d = dp.load_real(key)
        tr, va = d[d.split == "train"], d[d.split == "val"]
        import src.experiment as E
        p = E.get_params("E06", key, "lightgbm_tuned")
        est = lgb.LGBMRegressor(n_jobs=8, random_state=C.SEED, verbose=-1, **p).fit(F.real_base(tr), tr.power_kw)
        X = F.real_base(va)
        sv = shap.TreeExplainer(est).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
        OUT[f"{key}_shap"] = imp.round(4).to_dict()
        bar(ax, list(imp.index), imp.values)
        ax.set_title(f"Real ({key}): mean |SHAP|")
    fig.savefig(C.FIGURES / "e09_shap_real.png"); plt.close(fig)


if __name__ == "__main__":
    synthetic()
    real()
    (C.EXP_RESULTS / "E09_explainability.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps(OUT, indent=1))
