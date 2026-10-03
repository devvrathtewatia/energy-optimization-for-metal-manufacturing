"""E01 - EDA, data audit and leakage audit.  python -m experiments.e01_eda"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config as C, data_prep as dp, features as F, plotstyle as ps
from src.metrics import regression_metrics
import matplotlib.pyplot as plt

OUT = {}


def save(fig, name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / name)
    plt.close(fig)


def synthetic():
    d = dp.load_synthetic()
    tr, va = d[d.split == "train"], d[d.split == "val"]
    OUT["synthetic_summary"] = dict(
        rows=len(d), jobs=int(d.job_id.nunique()), power_kw=d.power_kw.describe().round(3).to_dict(),
        share_by_state=d.gt_state.value_counts(normalize=True).round(3).to_dict(),
        mean_power_by_state=d.groupby("gt_state").power_kw.mean().round(2).to_dict(),
        inefficient_share=float(d.gt_is_inefficient.mean()), glitch_share=float(d.gt_is_glitch.mean()),
        anomaly_jobs=d.groupby("job_id").gt_job_anomaly.first().value_counts().to_dict(),
        # how much of a cutting window's power is NOT cutting
        cutting_power_decomposition_kw=d[d.gt_state == "cutting"][["gt_p_base", "gt_p_coolant", "gt_p_idle", "gt_p_cut", "gt_p_axes"]].mean().round(2).to_dict(),
        oracle_floor_val=regression_metrics(va.power_kw, va.gt_power_true_kw),
        oracle_floor_val_nominal_rows=regression_metrics(va.power_kw[~va.gt_is_glitch], va.gt_power_true_kw[~va.gt_is_glitch]),
    )
    # ---- leakage audit --------------------------------------------------------------
    rec = np.sqrt(3) * va.voltage_v * va.current_a * va.power_factor / 1000
    audit = {"identity_sqrt3*V*I*cosphi (no fit)": regression_metrics(va.power_kw, rec)}
    sets = {
        "commanded features only (planning set)": F.commanded,
        "commanded + drive load signals": lambda x: pd.concat([F.commanded(x), x[C.SIGNAL_COLS]], axis=1),
        "commanded + V, I, cos(phi)  [LEAKAGE]": lambda x: pd.concat([F.commanded(x), x[C.LEAKY_COLS]], axis=1),
        "V, I, cos(phi) only  [LEAKAGE]": lambda x: x[C.LEAKY_COLS].assign(VI=x.voltage_v * x.current_a * x.power_factor),
    }
    for k, fb in sets.items():
        m = LinearRegression().fit(fb(tr), tr.power_kw)
        audit[f"linear: {k}"] = regression_metrics(va.power_kw, m.predict(fb(va)))
    OUT["leakage_audit_synthetic_val"] = audit

    # ---- figures ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    order = ["standby", "rapid", "aircut", "cutting"]
    data = [d.loc[(d.gt_state == s) & ~d.gt_is_glitch, "power_kw"] for s in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55, showfliers=False)
    for p in bp["boxes"]:
        p.set(facecolor=ps.PALETTE[0], alpha=0.35, edgecolor=ps.INK2)
    for k in ("medians",):
        for l in bp[k]:
            l.set(color=ps.INK, linewidth=2)
    ax.set_ylabel("electrical power [kW]")
    ax.set_title("Synthetic: power by machine state (glitches hidden)")
    save(fig, "e01_power_by_state.png")

    e = d[(d.gt_state == "cutting") & ~d.gt_is_glitch].copy()
    e["mrr"] = e.axial_depth_mm * e.radial_width_mm * e.feed_per_tooth_mm * e.n_flutes * e.spindle_speed_rpm / 1000
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, m in enumerate(C.MATERIAL_LIST):
        s = e[e.material == m].sample(min(1500, (e.material == m).sum()), random_state=1)
        ax.scatter(s.mrr, s.power_kw, s=10, alpha=0.45, color=ps.PALETTE[i], marker=ps.MARKERS[i], label=m, linewidths=0)
    ax.set_xlabel("material removal rate [cm³/min]")
    ax.set_ylabel("electrical power [kW]")
    ax.set_title("Synthetic: cutting power is a modest increment on large fixed loads")
    ax.legend(markerscale=2, loc="upper left")
    save(fig, "e01_power_vs_mrr.png")

    a = d[(d.gt_state == "aircut") & ~d.gt_is_glitch]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, (cm, lab) in enumerate({0: "coolant off", 1: "flood", 2: "high-pressure"}.items()):
        s = a[a.coolant_mode == cm]
        ax.scatter(s.spindle_speed_rpm, s.power_kw, s=10, alpha=0.5, color=ps.PALETTE[i], marker=ps.MARKERS[i], label=lab, linewidths=0)
    ax.set_xlabel("spindle speed [rpm]")
    ax.set_ylabel("electrical power [kW]")
    ax.set_title("Synthetic: air-cutting power = speed-dependent spindle loss + coolant step")
    ax.legend(markerscale=2)
    save(fig, "e01_aircut_power_vs_speed.png")

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    jobs = d.groupby("job_id").agg(mrr=("job_mrr_cm3_min", "first"), split=("split", "first"))
    bins = np.linspace(0, jobs.mrr.quantile(0.995), 50)
    for i, sp in enumerate(["train", "val", "test", "ood_val", "ood_test"]):
        ax.hist(jobs.mrr[jobs.split == sp].clip(upper=bins[-1]), bins=bins, histtype="step", linewidth=2, color=ps.PALETTE[i], label=sp, density=True)
    ax.set_xlabel("job MRR [cm³/min]")
    ax.set_ylabel("density")
    ax.set_title("Split design: ood_val / ood_test are extrapolation bands in MRR")
    ax.legend()
    save(fig, "e01_split_mrr.png")


def real():
    panels = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axs, ["ibarmia", "gmtk"]):
        raw = pd.read_csv(C.DATA_RAW / {"ibarmia": "IBARMIA_dataset.csv", "gmtk": "GMTK_dataset.csv"}[key])
        d = dp.load_real(key)
        lab = raw.groupby("power_consumption").powerDrive_SPINDLE.agg(["min", "max", "mean", "count"]).round(3)
        class_mean = raw.groupby("power_consumption").powerDrive_SPINDLE.transform("mean")
        panels[key] = dict(
            rows_raw=len(raw), duplicates=int(raw.duplicated().sum()),
            describe=raw.describe().T[["mean", "std", "min", "max"]].round(3).to_dict("index"),
            label_vs_power=lab.to_dict("index"),
            r2_of_class_means=float(1 - ((raw.powerDrive_SPINDLE - class_mean) ** 2).sum() / ((raw.powerDrive_SPINDLE - raw.powerDrive_SPINDLE.mean()) ** 2).sum()),
            corr_with_power=raw.drop(columns="power_consumption").corr()["powerDrive_SPINDLE"].round(3).to_dict(),
            frac_rows_spindle_abs_speed_gt_100rpm=float((raw.speed_SPINDLE.abs() > 100).mean()),
            frac_rows_power_lt_0=float((raw.powerDrive_SPINDLE < 0).mean()),
        )
        ax.scatter(d.speed_SPINDLE, d.powerDrive_SPINDLE, s=8, alpha=0.35, color=ps.PALETTE[0], linewidths=0)
        ax.set_xlabel("spindle speed [rpm]")
        ax.set_ylabel("spindle drive power [kW]")
        ax.set_title(f"Real ({key}): power vs speed")
    OUT["real"] = panels
    save(fig, "e01_real_power_vs_speed.png")


if __name__ == "__main__":
    synthetic()
    real()
    (C.EXP_RESULTS / "E01_eda.json").write_text(json.dumps(OUT, indent=1, default=str))
    print(json.dumps(OUT["leakage_audit_synthetic_val"], indent=1))
    print(json.dumps(OUT["synthetic_summary"]["cutting_power_decomposition_kw"]))
    print(json.dumps({k: {kk: v[kk] for kk in ["r2_of_class_means", "frac_rows_spindle_abs_speed_gt_100rpm", "frac_rows_power_lt_0", "corr_with_power"]} for k, v in OUT["real"].items()}, indent=1))
