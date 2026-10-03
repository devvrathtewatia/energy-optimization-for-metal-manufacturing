"""E12 - Energy-residual anomaly detection:  residual = actual - predicted  (out-of-fold).
python -m experiments.e12_anomaly_detection"""
import copy
import json

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold, KFold

from src import config as C, data_prep as dp, experiment as E, features as F, plotstyle as ps

OUT = {}
Z_THR, MIN_EXCESS_KW = 3.5, 0.4


def oof_predict(model, d, groups, n_splits=5):
    pred = np.zeros(len(d))
    cv = GroupKFold(n_splits) if groups is not None else KFold(n_splits, shuffle=True, random_state=C.SEED)
    for tr, te in cv.split(d, groups=groups):
        m = copy.deepcopy(model).fit(d.iloc[tr])
        pred[te] = m.predict(d.iloc[te])
    return pred


def robust_z(resid, yhat, n_bins=10, ref=None):
    """Standardise residuals by a heteroscedastic robust scale (MAD) estimated in bins of the predicted power."""
    ref = (resid, yhat) if ref is None else ref
    edges = np.unique(np.quantile(ref[1], np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b_ref, b = np.digitize(ref[1], edges[1:-1]), np.digitize(yhat, edges[1:-1])
    sig = np.array([1.4826 * np.median(np.abs(ref[0][b_ref == k] - np.median(ref[0][b_ref == k]))) if (b_ref == k).sum() > 20 else np.nan for k in range(len(edges) - 1)])
    sig = np.where(np.isnan(sig), np.nanmedian(sig), sig)
    return (resid - 0) / np.maximum(sig[b], 1e-3), sig


def synthetic():
    d = dp.load_synthetic()
    model = joblib.load(C.MODELS / "final_model.joblib")
    idm = d.split.isin(["train", "val", "test"]).to_numpy()
    did = d[idm].reset_index(drop=True)
    pred_id = oof_predict(model, did, did.job_id.to_numpy())
    did["pred"], did["resid"] = pred_id, did.power_kw - pred_id
    z, sig = robust_z(did.resid.to_numpy(), did.pred.to_numpy())
    did["z"] = z
    did["flag"] = (did.z > Z_THR) & (did.resid > MIN_EXCESS_KW)
    dood = d[~idm].reset_index(drop=True)
    dood["pred"] = model.predict(dood); dood["resid"] = dood.power_kw - dood.pred
    dood["z"], _ = robust_z(dood.resid.to_numpy(), dood.pred.to_numpy(), ref=(did.resid.to_numpy(), did.pred.to_numpy()))
    dood["flag"] = (dood.z > Z_THR) & (dood.resid > MIN_EXCESS_KW)

    def evaluate(x, name):
        y, f = x.gt_is_inefficient.to_numpy(), x.flag.to_numpy()
        tp, fp, fn = int((y & f).sum()), int((~y & f).sum()), int((y & ~f).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        kwh = lambda v: float(v.sum() * C.WINDOW_S / 3600)
        res = dict(rows=len(x), true_inefficient_windows=int(y.sum()), flagged=int(f.sum()), tp=tp, fp=fp, fn=fn,
                   precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                   average_precision=float(average_precision_score(y, x.z)), base_rate=float(y.mean()),
                   fp_glitch=int((~y & f & x.gt_is_glitch.to_numpy()).sum()), fp_other=int((~y & f & ~x.gt_is_glitch.to_numpy()).sum()),
                   true_excess_kwh=kwh(x.gt_extra_kw[y]), flagged_true_excess_kwh=kwh(x.gt_extra_kw[y & f]),
                   flagged_residual_kwh=kwh(x.resid[f]))
        bytype = {}
        for t in ["coolant_stuck_hp", "excess_tool_wear", "spindle_bearing_friction", "aux_leak", "axis_drag"]:
            m = y & (x.gt_job_anomaly == t).to_numpy()
            bytype[t] = dict(windows=int(m.sum()), recall=float(f[m].mean()) if m.any() else None,
                             median_extra_kw=float(x.gt_extra_kw[m].median()) if m.any() else None)
        res["recall_by_type"] = bytype
        bins = pd.cut(x.gt_extra_kw[y], [0.4, 1, 2, 4, 100])
        res["recall_by_true_extra_kw"] = {str(k): float(v) for k, v in pd.Series(f[y]).groupby(bins.to_numpy()).mean().items()}
        OUT[name] = res
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if not isinstance(v, dict)})
        return res

    evaluate(did, "in_distribution_oof")
    evaluate(dood, "extrapolation_ood_jobs")
    # threshold sweep
    sweep = []
    for zt in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        f = (did.z > zt) & (did.resid > MIN_EXCESS_KW); y = did.gt_is_inefficient
        tp = int((f & y).sum())
        sweep.append(dict(z_threshold=zt, flagged=int(f.sum()), precision=tp / max(f.sum(), 1), recall=tp / y.sum()))
    OUT["threshold_sweep"] = sweep
    # job level: flag a job if >= 50 % of its cutting windows are flagged
    cut = did[did.engaged]
    jl = cut.groupby("job_id").agg(frac=("flag", "mean"), anomalous=("gt_job_anomaly", lambda s: (s != "none").any()))
    jl_flag = jl.frac >= 0.5
    OUT["job_level"] = dict(jobs=len(jl), anomalous_jobs=int(jl.anomalous.sum()), flagged_jobs=int(jl_flag.sum()),
                            precision=float((jl_flag & jl.anomalous).sum() / max(jl_flag.sum(), 1)), recall=float((jl_flag & jl.anomalous).sum() / jl.anomalous.sum()))
    print("job level", OUT["job_level"])
    did[did.flag].drop(columns=[c for c in did.columns if c.startswith("gt_p_")]).to_csv(C.TABLES / "e12_flagged_windows_synthetic.csv", index=False)

    # ---- figures -------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    lim = (-4, 8)
    ax.hist(did.resid[~did.gt_is_inefficient & ~did.gt_is_glitch].clip(*lim), bins=120, color=ps.PALETTE[0], alpha=0.75, label="nominal windows", density=True)
    ax.hist(did.resid[did.gt_is_inefficient].clip(*lim), bins=120, color=ps.PALETTE[1], alpha=0.75, label="injected inefficiency", density=True)
    ax.set_yscale("log"); ax.set_xlabel("residual = actual - predicted [kW]  (clipped to [-4, 8])"); ax.set_ylabel("density (log)")
    ax.set_title("Out-of-fold energy residuals (synthetic)"); ax.legend()
    p, r, _ = precision_recall_curve(did.gt_is_inefficient, did.z)
    axs[1].plot(r, p, color=ps.PALETTE[0]); axs[1].axhline(did.gt_is_inefficient.mean(), color=ps.INK2, linestyle=":", linewidth=1.2)
    axs[1].text(0.02, did.gt_is_inefficient.mean() + 0.02, "chance", color=ps.INK2, fontsize=9)
    axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision"); axs[1].set_title(f"Precision-recall (AP = {OUT['in_distribution_oof']['average_precision']:.2f})")
    fig.savefig(C.FIGURES / "e12_residuals_pr_synthetic.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bt = OUT["in_distribution_oof"]["recall_by_type"]
    names = [k for k in bt if bt[k]["recall"] is not None]
    ax.barh(names[::-1], [bt[k]["recall"] for k in names][::-1], color=ps.PALETTE[0], height=0.6)
    for i, k in enumerate(names[::-1]):
        ax.text(bt[k]["recall"] + 0.01, i, f"{bt[k]['recall']:.2f}   (median excess {bt[k]['median_extra_kw']:.1f} kW)", va="center", fontsize=9)
    ax.set_xlim(0, 1.25); ax.set_xlabel("recall of injected inefficiency (window level)")
    ax.set_title("Which inefficiencies does the residual detect?")
    fig.savefig(C.FIGURES / "e12_recall_by_type.png"); plt.close(fig)


def real():
    import lightgbm as lgb
    res = {}
    for key in ["ibarmia", "gmtk"]:
        d = dp.load_real(key).reset_index(drop=True)
        cands = {"random_forest_tuned": ("E05", RandomForestRegressor), "lightgbm_tuned": ("E06", lgb.LGBMRegressor)}
        best = min(cands, key=lambda n: E.get_entry(cands[n][0], key, n)["metrics"]["val"]["all"]["mae"])
        p = E.get_params(cands[best][0], key, best)
        kw = dict(n_jobs=8, random_state=C.SEED, **({"verbose": -1} if best.startswith("light") else {"n_estimators": 300}))
        model = __import__("src.models", fromlist=["x"]).SkModel(best, cands[best][1](**{**kw, **p}), F.real_base)
        pred = oof_predict(model, d, None)
        d["pred"], d["resid"] = pred, d.power_kw - pred
        d["z"], _ = robust_z(d.resid.to_numpy(), d.pred.to_numpy())
        d["flag"] = (d.z > Z_THR) & (d.resid > 0.5)
        f = d[d.flag]
        res[key] = dict(model=best, rows=len(d), flagged=int(len(f)), flagged_pct=float(100 * len(f) / len(d)),
                        excess_kw_flagged_mean=float(f.resid.mean()) if len(f) else None,
                        profile=pd.DataFrame({"flagged_median": f[F.REAL_BASE].median(), "all_median": d[F.REAL_BASE].median()}).round(3).to_dict("index"))
        f.sort_values("z", ascending=False).head(200).to_csv(C.TABLES / f"e12_flagged_windows_real_{key}.csv", index=False)
        print(key, res[key]["model"], res[key]["flagged"], "flagged")
    OUT["real"] = res
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axs, ["ibarmia", "gmtk"]):
        d = dp.load_real(key).reset_index(drop=True)
        f = pd.read_csv(C.TABLES / f"e12_flagged_windows_real_{key}.csv")
        ax.scatter(d.speed_SPINDLE, d.power_kw, s=6, alpha=0.25, color=ps.PALETTE[0], linewidths=0, label="all rows")
        ax.scatter(f.speed_SPINDLE, f.power_kw, s=14, color=ps.PALETTE[1], marker="s", linewidths=0, label="top flagged (potentially inefficient)")
        ax.set_xlabel("spindle speed [rpm]"); ax.set_ylabel("spindle drive power [kW]"); ax.set_title(f"Real ({key}): rows with unusually high power"); ax.legend(fontsize=8)
    fig.savefig(C.FIGURES / "e12_real_flagged.png"); plt.close(fig)


if __name__ == "__main__":
    synthetic()
    real()
    (C.EXP_RESULTS / "E12_anomaly.json").write_text(json.dumps(OUT, indent=1, default=str))
