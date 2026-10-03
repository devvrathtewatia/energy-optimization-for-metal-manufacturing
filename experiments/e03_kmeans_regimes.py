"""E03 - Do K-Means operating regimes exist, and does modelling them help?  python -m experiments.e03_kmeans_regimes"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src import config as C, data_prep as dp, experiment as E, features as F, models as M, plotstyle as ps

EXP = "E03"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def lgbm():
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=8, random_state=C.SEED)


def ridge(deg):
    s = [StandardScaler()] + ([PolynomialFeatures(deg, include_bias=False), StandardScaler()] if deg > 1 else [])
    return make_pipeline(*s, Ridge(alpha=1.0))


def select_k(tr, regime_fb, truth=None, kmax=10):
    X = StandardScaler().fit_transform(regime_fb(tr))
    rows = []
    for k in range(2, kmax + 1):
        km = KMeans(k, n_init=20, random_state=C.SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=4000, random_state=C.SEED)
        row = dict(k=k, inertia=float(km.inertia_), silhouette=float(sil))
        if truth is not None:
            row.update(ari=float(adjusted_rand_score(truth, km.labels_)), nmi=float(normalized_mutual_info_score(truth, km.labels_)))
        rows.append(row)
    return pd.DataFrame(rows)


def model_grid(d, dataset, fb, regime_fb, ks):
    entries = []
    bases = {"ridge_linear": (ridge(1), 2), "ridge_poly2": (ridge(2), 3), "lightgbm_default": (lgbm(), 6)}
    for bname, (est, cx) in bases.items():
        entries.append(E.run_model(f"{EXP}b", dataset, M.SkModel(f"{bname}__global", est, fb, "no-regime baseline", cx), d))
        for k in ks:
            for mode in ["onehot", "per_cluster"]:
                m = M.RegimeModel(f"{bname}__kmeans{k}_{mode}", est, fb, regime_fb, k, mode)
                entries.append(E.run_model(f"{EXP}c", dataset, m, d, params=dict(k=k, mode=mode, base=bname)))
    return entries


def synthetic():
    d = dp.load_synthetic()
    tr = d[d.split == "train"]
    sel = select_k(tr, F.regime_features, tr.gt_state)
    k_sil = int(sel.loc[sel.silhouette.idxmax(), "k"])
    print(sel.round(3).to_string(index=False), "\nk by max silhouette:", k_sil)
    ks = sorted({k_sil, 4})
    # cluster vs ground-truth state (training rows)
    reg = M.KMeansRegimes(F.regime_features, 4).fit(tr)
    lab = reg.predict(tr)
    conf = pd.crosstab(tr.gt_state, lab).reindex(["standby", "rapid", "aircut", "cutting"])
    print(conf)
    pur = float(conf.max(axis=0).sum() / conf.values.sum())
    entries = model_grid(d, "synthetic", F.commanded, F.regime_features, ks)
    extra = dict(selection=sel.to_dict("records"), k_by_silhouette=k_sil, confusion_k4=conf.to_dict(), purity_k4=pur,
                 ari_k4=float(adjusted_rand_score(tr.gt_state, lab)), nmi_k4=float(normalized_mutual_info_score(tr.gt_state, lab)))
    E.save(EXP, "synthetic", entries, extra)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    axs[0].plot(sel.k, sel.inertia, color=ps.PALETTE[0], marker="o", markersize=5)
    axs[0].set_title("Inertia (elbow)"); axs[0].set_xlabel("k")
    axs[1].plot(sel.k, sel.silhouette, color=ps.PALETTE[0], marker="o", markersize=5, label="silhouette")
    axs[1].set_title("Silhouette"); axs[1].set_xlabel("k")
    axs[2].plot(sel.k, sel.ari, color=ps.PALETTE[1], marker="s", markersize=5, label="ARI vs true state")
    axs[2].plot(sel.k, sel.nmi, color=ps.PALETTE[2], marker="^", markersize=5, label="NMI vs true state")
    axs[2].set_title("Agreement with ground-truth states"); axs[2].set_xlabel("k"); axs[2].legend()
    fig.savefig(C.FIGURES / "e03_kmeans_selection_synthetic.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    im = ax.imshow(conf.values, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(conf.shape[1]), [f"cluster {c}" for c in conf.columns]); ax.set_yticks(range(conf.shape[0]), conf.index)
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            v = conf.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > conf.values.max() / 2 else ps.INK, fontsize=9)
    ax.grid(False); ax.set_title("K-Means (k=4) vs true machine state (train)")
    fig.savefig(C.FIGURES / "e03_confusion_k4_synthetic.png"); plt.close(fig)


def real():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axs, ["ibarmia", "gmtk"]):
        d = dp.load_real(key)
        tr = d[d.split == "train"]
        sel = select_k(tr, F.real_regime_features)
        k_sil = int(sel.loc[sel.silhouette.idxmax(), "k"])
        print(key, "k by silhouette", k_sil, "\n", sel.round(3).to_string(index=False))
        ks = sorted({k_sil, 4})
        reg = M.KMeansRegimes(F.real_regime_features, k_sil).fit(tr)
        lab = reg.predict(tr)
        stats = tr.assign(c=lab).groupby("c").power_kw.agg(["count", "mean", "std", "min", "max"]).round(3)
        print(stats)
        tot_var = tr.power_kw.var()
        within = tr.assign(c=lab).groupby("c").power_kw.apply(lambda s: s.var() * len(s)).sum() / len(tr)
        entries = model_grid(d, key, F.real_base, F.real_regime_features, ks)
        E.save(EXP, key, entries, dict(selection=sel.to_dict("records"), k_by_silhouette=k_sil,
                                       cluster_power_stats=stats.to_dict("index"), variance_explained_by_clusters=float(1 - within / tot_var)))
        groups = [tr.power_kw[lab == c] for c in sorted(set(lab))]
        bp = ax.boxplot(groups, tick_labels=[f"c{c}" for c in sorted(set(lab))], patch_artist=True, showfliers=False, widths=0.55)
        for p in bp["boxes"]:
            p.set(facecolor=ps.PALETTE[0], alpha=0.35, edgecolor=ps.INK2)
        for l in bp["medians"]:
            l.set(color=ps.INK, linewidth=2)
        ax.set_title(f"Real ({key}): spindle power by K-Means cluster (k={k_sil})"); ax.set_ylabel("kW")
    fig.savefig(C.FIGURES / "e03_real_cluster_power.png"); plt.close(fig)


if __name__ == "__main__":
    synthetic()
    real()
