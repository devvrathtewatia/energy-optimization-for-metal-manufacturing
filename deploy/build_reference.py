"""One-off script (run on your laptop, from the repo root): builds deploy/reference.json.
It stores (1) the anomaly-detection noise scale per power bin, computed from out-of-fold residuals exactly like
experiments/e12_anomaly_detection.py, and (2) the optimizer's trust-region limit (max removal rate seen in training).
    python -m deploy.build_reference
"""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from src import config as C, data_prep as dp

d = dp.load_synthetic()
model = joblib.load(C.MODELS / "final_model.joblib")
did = d[d.split.isin(["train", "val", "test"])].reset_index(drop=True)

pred = np.zeros(len(did))
for tr, te in GroupKFold(5).split(did, groups=did.job_id.to_numpy()):
    pred[te] = copy.deepcopy(model).fit(did.iloc[tr]).predict(did.iloc[te])
resid = did.power_kw.to_numpy() - pred

edges = np.unique(np.quantile(pred, np.linspace(0, 1, 11)))
inner = edges[1:-1]
b = np.digitize(pred, inner)
sig = np.array([1.4826 * np.median(np.abs(resid[b == k] - np.median(resid[b == k]))) if (b == k).sum() > 20 else np.nan
                for k in range(len(edges) - 1)])
sig = np.where(np.isnan(sig), np.nanmedian(sig), sig)

ref = dict(bin_edges=[float(x) for x in inner], sigma=[float(x) for x in sig], z_threshold=3.5, min_excess_kw=0.4,
           trust_mrr_cm3_min=float(d[d.split == "train"].job_mrr_cm3_min.max()))
(Path(__file__).parent / "reference.json").write_text(json.dumps(ref, indent=2))
print(json.dumps(ref, indent=2))
