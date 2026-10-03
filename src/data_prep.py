"""Cleaning and splitting. Every rule here is applied BEFORE any modelling and documented in the report.

Synthetic data
  * R1: drop meter dropouts (power_kw < 1.0 kW). Physically impossible: the machine's base load alone
        is > 2 kW whenever it is on. This is a domain rule, not an outlier filter fitted on the target
        distribution. Spike glitches are NOT removed (they cannot be identified by a physical rule).
  * Splits are by JOB (all windows of a job stay together) to prevent leakage between windows of the
    same job. Jobs are first ordered by commanded MRR: the top ~12 % form two extrapolation bands
    (ood_val: 88-94th pct, ood_test: >94th pct) that are never seen in training; the remaining jobs are
    split 60/20/20 into train/val/test at random.

Real data (CFAA / Zenodo 14445879)
  * exact duplicate rows removed (about 13 %), then a random 60/20/20 split. The files have no
    timestamps or run ids, so grouping is impossible (documented limitation).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C

REAL_FILES = {"ibarmia": "IBARMIA_dataset.csv", "gmtk": "GMTK_dataset.csv"}


def prepare_synthetic(seed: int = C.SEED) -> pd.DataFrame:
    df = pd.read_csv(C.DATA_SYNTH / "synthetic_machining_windows.csv")
    n0 = len(df)
    df = df[df["power_kw"] >= 1.0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    jobs = df.groupby("job_id")["job_mrr_cm3_min"].first()
    q_val, q_test = jobs.quantile(0.88), jobs.quantile(0.94)
    split = pd.Series("", index=jobs.index)
    split[jobs > q_test] = "ood_test"
    split[(jobs > q_val) & (jobs <= q_test)] = "ood_val"
    rest = split.index[split == ""].to_numpy().copy()
    rng.shuffle(rest)
    n = len(rest)
    split[rest[: int(0.6 * n)]] = "train"
    split[rest[int(0.6 * n): int(0.8 * n)]] = "val"
    split[rest[int(0.8 * n):]] = "test"
    df["split"] = df["job_id"].map(split)
    df["engaged"] = df["axial_depth_mm"] > 0
    df.to_csv(C.DATA_PROCESSED / "synthetic_clean.csv", index=False)
    rep = dict(rows_raw=n0, rows_clean=len(df), dropped_dropouts=n0 - len(df),
               mrr_q88=float(q_val), mrr_q94=float(q_test),
               rows_per_split=df["split"].value_counts().to_dict(),
               jobs_per_split=split.value_counts().to_dict(),
               max_train_mrr=float(df.loc[df.split == "train", "job_mrr_cm3_min"].max()))
    (C.DATA_PROCESSED / "synthetic_prep_report.json").write_text(json.dumps(rep, indent=2))
    return df


def prepare_real(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    out, rep = {}, {}
    for key, fname in REAL_FILES.items():
        raw = pd.read_csv(C.DATA_RAW / fname)
        df = raw.drop_duplicates().reset_index(drop=True)
        df["power_kw"] = df["powerDrive_SPINDLE"]  # alias so the same harness/target name works
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        n = len(df)
        split = np.empty(n, dtype=object)
        split[idx[: int(0.6 * n)]] = "train"
        split[idx[int(0.6 * n): int(0.8 * n)]] = "val"
        split[idx[int(0.8 * n):]] = "test"
        df["split"] = split
        df.to_csv(C.DATA_PROCESSED / f"real_{key}_clean.csv", index=False)
        out[key] = df
        rep[key] = dict(rows_raw=len(raw), rows_unique=n, duplicates_removed=len(raw) - n,
                        rows_per_split=df["split"].value_counts().to_dict())
    (C.DATA_PROCESSED / "real_prep_report.json").write_text(json.dumps(rep, indent=2))
    return out


def load_synthetic() -> pd.DataFrame:
    return pd.read_csv(C.DATA_PROCESSED / "synthetic_clean.csv")


def load_real(key: str) -> pd.DataFrame:
    return pd.read_csv(C.DATA_PROCESSED / f"real_{key}_clean.csv")


if __name__ == "__main__":
    s = prepare_synthetic()
    r = prepare_real()
    print(json.dumps(json.loads((C.DATA_PROCESSED / "synthetic_prep_report.json").read_text()), indent=2))
    print(json.dumps(json.loads((C.DATA_PROCESSED / "real_prep_report.json").read_text()), indent=2))
