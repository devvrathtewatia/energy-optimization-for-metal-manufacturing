"""E14 - Why do tree models do so much better than linear models on the REAL CFAA data?  Is it leakage?
Probes: (1) which inputs carry the signal, (2) near-duplicate / nearest-neighbour check, (3) cluster-grouped CV
(generalisation to unseen regions of input space) vs random CV.   python -m experiments.e14_real_data_probes"""
import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config as C, data_prep as dp, experiment as E, features as F, plotstyle as ps
from src.metrics import regression_metrics

OUT = {}


def fit_eval(p, cols, tr, te):
    m = lgb.LGBMRegressor(n_jobs=8, random_state=C.SEED, verbose=-1, **p).fit(tr[cols], tr.power_kw)
    return regression_metrics(te.power_kw, m.predict(te[cols]), 0.5)


def plot(out):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, key in zip(axs, ["ibarmia", "gmtk"]):
        sub = out[key]["feature_subsets_val"]
        names = list(sub)[::-1]
        ax.barh(names, [sub[k]["r2"] for k in names], color=ps.PALETTE[0], height=0.6)
        ax.set_title(f"Real ({key}): validation R²"); ax.set_xlabel("R² of tuned LightGBM, random split")
    fig.savefig(C.FIGURES / "e14_real_feature_subsets.png"); plt.close(fig)


def main():
    for key in ["ibarmia", "gmtk"]:
        d = dp.load_real(key)
        tr, va = d[d.split == "train"], d[d.split == "val"]
        p = E.get_params("E06", key, "lightgbm_tuned")
        res = {}
        subsets = {"speed+override": ["speed_SPINDLE", "override_SPINDLE"],
                   "speed+override+load_X+load_Z": ["speed_SPINDLE", "override_SPINDLE", "load_X", "load_Z"],
                   "speed+override+power_Z": ["speed_SPINDLE", "override_SPINDLE", "power_Z"],
                   "all five inputs": F.REAL_BASE,
                   "power_Z only": ["power_Z"], "load_Z only": ["load_Z"]}
        res["feature_subsets_val"] = {k: fit_eval(p, c, tr, va) for k, c in subsets.items()}
        # nearest-neighbour structure
        sc = StandardScaler().fit(tr[F.REAL_BASE])
        Xtr, Xva = sc.transform(tr[F.REAL_BASE]), sc.transform(va[F.REAL_BASE])
        nn = NearestNeighbors(n_neighbors=2).fit(Xtr)
        d_val = nn.kneighbors(Xva, n_neighbors=1)[0][:, 0]
        d_loo = nn.kneighbors(Xtr, n_neighbors=2)[0][:, 1]
        idx = nn.kneighbors(Xva, n_neighbors=1)[1][:, 0]
        res["nn"] = dict(median_val_to_train=float(np.median(d_val)), median_train_loo=float(np.median(d_loo)),
                         share_val_with_nn_dist_lt_0p05=float((d_val < 0.05).mean()),
                         one_nn_regressor_val=regression_metrics(va.power_kw, tr.power_kw.to_numpy()[idx], 0.5))
        # cluster-grouped vs random CV on all unique rows
        X = StandardScaler().fit_transform(d[F.REAL_BASE])
        grp = KMeans(30, n_init=10, random_state=C.SEED).fit_predict(X)
        oof_g, oof_r = np.zeros(len(d)), np.zeros(len(d))
        for tri, tei in GroupKFold(5).split(X, groups=grp):
            oof_g[tei] = lgb.LGBMRegressor(n_jobs=8, random_state=C.SEED, verbose=-1, **p).fit(d.iloc[tri][F.REAL_BASE], d.power_kw.iloc[tri]).predict(d.iloc[tei][F.REAL_BASE])
        for tri, tei in KFold(5, shuffle=True, random_state=C.SEED).split(X):
            oof_r[tei] = lgb.LGBMRegressor(n_jobs=8, random_state=C.SEED, verbose=-1, **p).fit(d.iloc[tri][F.REAL_BASE], d.power_kw.iloc[tri]).predict(d.iloc[tei][F.REAL_BASE])
        res["cv_random_5fold"] = regression_metrics(d.power_kw, oof_r, 0.5)
        res["cv_cluster_grouped_5fold"] = regression_metrics(d.power_kw, oof_g, 0.5)
        OUT[key] = res
        print(key, json.dumps({k: (v if k != "feature_subsets_val" else {kk: round(vv["r2"], 3) for kk, vv in v.items()}) for k, v in res.items()}, indent=1, default=lambda o: round(o, 3)))
    (C.EXP_RESULTS / "E14_real_data_probes.json").write_text(json.dumps(OUT, indent=1))

    plot(OUT)


if __name__ == "__main__":
    main()
