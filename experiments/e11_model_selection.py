"""E11 - Final-model selection by the frozen rule in experiments/PROTOCOL.md, then one-shot test/ood_test evaluation.
python -m experiments.e11_model_selection"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C, data_prep as dp, experiment as E, plotstyle as ps
from src.metrics import regression_metrics

FAMILY_ORDER = ["null", "physics", "linear", "kmeans+model", "tree", "forest", "boosting", "neural_network", "hybrid"]


def leaderboard():
    rows = []
    for e in E.load_all():
        if e["dataset"] != "synthetic" or e["family"] == "diagnostic" or e["params"].get("diagnostic"):
            continue
        m = e["metrics"]
        g = lambda sp, sub, k: m.get(sp, {}).get(sub, {}).get(k, np.nan)
        rows.append(dict(
            exp_id=e["exp_id"], model=e["model"], family=e["family"], complexity=e["complexity"],
            val_mae=g("val", "all", "mae"), val_rmse=g("val", "all", "rmse"), val_r2=g("val", "all", "r2"), val_mape=g("val", "all", "mape"),
            val_eng_mae=g("val", "engaged", "mae"), val_eng_r2=g("val", "engaged", "r2"),
            oodval_eng_mae=g("ood_val", "engaged", "mae"), oodval_eng_rmse=g("ood_val", "engaged", "rmse"), oodval_eng_r2=g("ood_val", "engaged", "r2"),
            test_mae=g("test", "all", "mae"), test_rmse=g("test", "all", "rmse"), test_r2=g("test", "all", "r2"), test_mape=g("test", "all", "mape"),
            oodtest_eng_mae=g("ood_test", "engaged", "mae"), oodtest_eng_rmse=g("ood_test", "engaged", "rmse"), oodtest_eng_r2=g("ood_test", "engaged", "r2"),
            train_mae=g("train", "all", "mae"),
        ))
    df = pd.DataFrame(rows).drop_duplicates("model", keep="last").reset_index(drop=True)
    df["score"] = 0.5 * df.val_mae / df.val_mae.min() + 0.5 * df.oodval_eng_mae / df.oodval_eng_mae.min()
    return df.sort_values("score").reset_index(drop=True)


def main():
    df = leaderboard()
    best = df.score.min()
    near = df[df.score <= 1.02 * best].sort_values(["complexity", "score"])
    winner = near.iloc[0]
    df["within_2pct_of_best"] = df.score <= 1.02 * best
    df.to_csv(C.TABLES / "leaderboard_synthetic.csv", index=False)
    print(df[["exp_id", "model", "family", "complexity", "val_mae", "oodval_eng_mae", "score"]].head(15).to_string(index=False))
    print("\nwithin 2 % of best score:\n", near[["model", "complexity", "score"]].to_string(index=False))
    print("\nWINNER (rule):", winner.model)

    # ---- one-shot final evaluation: refit on train+val with unchanged hyper-parameters ------------------------
    d = dp.load_synthetic()
    model = joblib.load(C.CACHE / f"{winner.model}.joblib")
    dev = d[d.split.isin(["train", "val"])]
    model.fit(dev)
    C.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, C.MODELS / "final_model.joblib")
    final = {}
    eng = d["engaged"].to_numpy()
    for sp in ["test", "ood_test"]:
        s = d[d.split == sp]
        p = model.predict(s)
        final[sp] = {"all": regression_metrics(s.power_kw, p, 1.0), "engaged": regression_metrics(s.power_kw[s.engaged], p[s.engaged.to_numpy()], 1.0),
                     "nominal_rows": regression_metrics(s.power_kw[~s.gt_is_glitch], p[~s.gt_is_glitch.to_numpy()], 1.0)}
    # reference: all other 'family bests' on test for context are in the leaderboard (trained on train only)
    out = dict(rule="score=0.5*MAE_val/min + 0.5*MAE_oodval_engaged/min; ties (<=2% of best) -> lowest complexity",
               best_score_model=df.iloc[0].model, winner=winner.model, winner_family=winner.family, winner_complexity=int(winner.complexity),
               candidates_within_2pct=near.model.tolist())
    out["final_metrics_trainval_refit"] = final
    (C.EXP_RESULTS / "E11_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(final, indent=1))
    if hasattr(model, "params"):   # fitted-physics winner: compare coefficients with the simulator's effective truth
        from experiments.e10_hybrid_physics_ml import truth_table
        truth_table(model).to_csv(C.TABLES / "final_model_param_recovery.csv", index=False)

    # ---- figures ----------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    fams = [f for f in FAMILY_ORDER if f in set(df.family)]
    colors = {"null": "#8a8984", **{f: ps.PALETTE[i] for i, f in enumerate([f for f in fams if f != "null"])}}
    for i, f in enumerate(fams):
        s = df[df.family == f]
        ax.scatter(s.val_mae, s.oodval_eng_mae, s=38, color=colors[f], marker=ps.MARKERS[i % 4], label=f, alpha=0.85, linewidths=0)
    w = df[df.model == winner.model].iloc[0]
    ax.scatter([w.val_mae], [w.oodval_eng_mae], s=170, facecolors="none", edgecolors=ps.INK, linewidths=2)
    ax.annotate(f"selected: {winner.model}", (w.val_mae, w.oodval_eng_mae), textcoords="offset points", xytext=(16, 30), fontsize=9, arrowprops=dict(arrowstyle="-", color=ps.INK2))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("validation MAE, all windows [kW]  (in-distribution)")
    ax.set_ylabel("OOD-validation MAE, engaged windows [kW]  (extrapolation)")
    ax.set_title("Every synthetic-data model: in-distribution vs extrapolation error")
    ax.legend(title="family", fontsize=8, ncol=2, loc="lower right")
    fig.savefig(C.FIGURES / "e11_model_landscape.png"); plt.close(fig)

    best_fam = df.sort_values("score").groupby("family").head(1).sort_values("score")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    names = best_fam.model.tolist()[::-1]
    for ax, col, ttl in [(axs[0], "val_mae", "Validation MAE [kW] (all windows)"), (axs[1], "oodval_eng_mae", "OOD-validation MAE [kW] (engaged)")]:
        ax.barh(names, best_fam[col].tolist()[::-1], color=ps.PALETTE[0], height=0.6)
        ax.set_title(ttl); ax.set_xlabel("kW")
    fig.suptitle("Best model of each family (by selection score)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(C.FIGURES / "e11_family_best.png"); plt.close(fig)


if __name__ == "__main__":
    main()
