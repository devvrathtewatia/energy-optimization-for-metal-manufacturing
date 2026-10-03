"""Regression metrics. MAPE is only meaningful away from zero, so it is computed on rows with y >= floor."""
from __future__ import annotations

import numpy as np


def regression_metrics(y, p, mape_floor: float = 1.0) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss_res, ss_tot = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    m = y >= mape_floor
    return dict(
        mae=float(np.abs(err).mean()),
        rmse=float(np.sqrt((err ** 2).mean())),
        r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        mape=float(np.mean(np.abs(err[m]) / y[m]) * 100) if m.any() else float("nan"),
        n=int(len(y)),
    )
