"""Experiment harness: evaluate a fitted model on every split/subset and persist a JSON record."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .metrics import regression_metrics

SPLITS = ["train", "val", "test", "ood_val", "ood_test"]


def subsets_for(dataset: str):
    if dataset == "synthetic":
        return {"all": lambda d: np.ones(len(d), bool), "engaged": lambda d: d["engaged"].to_numpy(),
                "nominal": lambda d: ~d["gt_is_glitch"].to_numpy()}  # diagnostic only (uses ground truth)
    return {"all": lambda d: np.ones(len(d), bool)}


MAPE_FLOOR = {"synthetic": 1.0, "ibarmia": 0.5, "gmtk": 0.5}


def evaluate(model, data: pd.DataFrame, dataset: str, target: str = C.TARGET) -> dict:
    out = {}
    subs = subsets_for(dataset)
    for sp in SPLITS:
        d = data[data["split"] == sp]
        if d.empty:
            continue
        p = np.asarray(model.predict(d), float)
        out[sp] = {}
        for sname, fn in subs.items():
            m = fn(d)
            if m.sum() > 1:
                out[sp][sname] = regression_metrics(d[target].to_numpy()[m], p[m], MAPE_FLOOR[dataset])
    return out


def run_model(exp_id: str, dataset: str, model, data: pd.DataFrame, params: dict | None = None,
              notes: str = "", fit: bool = True, fit_df: pd.DataFrame | None = None) -> dict:
    train = data[data["split"] == "train"] if fit_df is None else fit_df
    t0 = time.time()
    if fit:
        model.fit(train)
    fit_s = time.time() - t0
    if dataset == "synthetic":
        import joblib
        C.CACHE.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(model, C.CACHE / f"{model.name}.joblib")
        except Exception as ex:  # a model that cannot be pickled is simply not available to later stages
            print(f"   (not cached: {model.name}: {type(ex).__name__})")
    entry = dict(exp_id=exp_id, dataset=dataset, model=model.name, family=model.family,
                 complexity=int(model.complexity), params=params or {}, fit_seconds=round(fit_s, 2),
                 metrics=evaluate(model, data, dataset), notes=notes)
    v = entry["metrics"]["val"]["all"]
    print(f"[{exp_id}] {dataset:9s} {model.name:44s} val MAE={v['mae']:.3f} RMSE={v['rmse']:.3f} R2={v['r2']:.3f}"
          + (f" | oodval(eng) RMSE={entry['metrics']['ood_val']['engaged']['rmse']:.3f}" if "ood_val" in entry["metrics"] else ""))
    return entry


def save(exp_id: str, dataset: str, entries: list[dict], extra: dict | None = None) -> Path:
    C.EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    path = C.EXP_RESULTS / f"{exp_id}_{dataset}.json"
    path.write_text(json.dumps(dict(exp_id=exp_id, dataset=dataset, entries=entries, extra=extra or {}),
                               indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return path


def load_all() -> list[dict]:
    rows = []
    for p in sorted(C.EXP_RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        rows += d.get("entries", [])
    return rows


def tune_and_run(exp_id, dataset, d, name, estimator, dist, fb, family, complexity, n_iter=20, notes=""):
    """Tune on TRAIN (grouped CV), refit on train, evaluate on all splits."""
    from sklearn.base import clone
    from . import models as M
    tr = d[d["split"] == "train"]
    best, cv = M.tune(estimator, dist, fb, tr, n_iter=n_iter)
    est = clone(estimator).set_params(**best)
    return run_model(exp_id, dataset, M.SkModel(name, est, fb, family, complexity), d,
                     params={**best, "cv_rmse_train": cv}, notes=notes)


def get_entry(exp_id: str, dataset: str, model: str) -> dict:
    d = json.loads((C.EXP_RESULTS / f"{exp_id}_{dataset}.json").read_text())
    for e in d["entries"]:
        if e["model"] == model:
            return e
    raise KeyError(model)


def get_params(exp_id: str, dataset: str, model: str) -> dict:
    return {k: v for k, v in get_entry(exp_id, dataset, model)["params"].items() if k not in ("cv_rmse_train",)}
