"""Small web API that serves the fitted physics-parametric CNC power model.

Run locally:   uvicorn deploy.api:app --port 8000      (from the repo root)
Run on EC2:    same command, with --host 0.0.0.0
"""
import json
import logging
import os
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import PhysParamModel  # noqa: E402  (the model class from this repo)
from src import optimize as O  # noqa: E402
from deploy.iot_listener import Listener  # noqa: E402

# Rebuild the fitted model from the saved coefficients (no pickle, so no version problems).
_saved = json.loads((Path(__file__).parent / "model_params.json").read_text())
model = PhysParamModel()
model.theta_ = np.array([_saved["params"][n] for n in PhysParamModel.NAMES])

_ref = json.loads((Path(__file__).parent / "reference.json").read_text())
_edges, _sigma = np.array(_ref["bin_edges"]), np.array(_ref["sigma"])

LIVE = deque(maxlen=500)          # most recent live readings, newest last
_listener = None


@asynccontextmanager
async def lifespan(app):
    """Start listening to the machine feed only if it is configured (environment variables)."""
    global _listener
    if os.environ.get("IOT_ENDPOINT") or os.environ.get("IOT_LOCAL_HOST"):
        _listener = Listener(_on_reading, endpoint=os.environ.get("IOT_ENDPOINT"), cert=os.environ.get("IOT_CERT"),
                             key=os.environ.get("IOT_KEY"), ca=os.environ.get("IOT_CA"),
                             local_host=os.environ.get("IOT_LOCAL_HOST")).start()
    yield


app = FastAPI(title="CNC milling energy model (simulated data)", lifespan=lifespan)


class Window(BaseModel):
    material: str              # Al6061, C45, SS316L or Ti6Al4V
    spindle_speed_rpm: float
    axis_speed_mm_min: float
    feed_per_tooth_mm: float
    axial_depth_mm: float
    radial_width_mm: float
    tool_diameter_mm: float
    n_flutes: int
    coolant_mode: int          # 0 off, 1 flood, 2 high-pressure
    tool_age_min: float
    machine_on_min: float
    ambient_temp_c: float


class Request(BaseModel):
    windows: list[Window]


def _score(df):
    """Predicted power, excess over prediction, robust z-score and anomaly flag for each row (needs a power_kw column)."""
    pred = model.predict(df)
    resid = df["power_kw"].to_numpy() - pred
    z = resid / np.maximum(_sigma[np.digitize(pred, _edges)], 1e-3)
    flag = (z > _ref["z_threshold"]) & (resid > _ref["min_excess_kw"])
    return pred, resid, z, flag


@app.get("/health")
def health():
    return {"status": "ok", "model": _saved["model_name"]}


@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame([w.model_dump() for w in req.windows])
    kw = model.predict(df)
    return {"predicted_power_kw": [round(float(v), 4) for v in kw]}


class MeasuredWindow(Window):
    power_kw: float            # measured electrical power for this 10 s window


class AnomalyRequest(BaseModel):
    windows: list[MeasuredWindow]


@app.post("/anomaly")
def anomaly(req: AnomalyRequest):
    """Flag windows that use clearly more power than the model expects (same rule as experiments/e12)."""
    df = pd.DataFrame([w.model_dump() for w in req.windows])
    pred, resid, z, flag = _score(df)
    return {
        "predicted_power_kw": [round(float(v), 3) for v in pred],
        "excess_kw": [round(float(v), 3) for v in resid],
        "z_score": [round(float(v), 2) for v in z],
        "flagged": [bool(v) for v in flag],
        "n_flagged": int(flag.sum()),
        "flagged_excess_kwh": round(float(resid[flag].sum() * 10 / 3600), 3),
    }


class Job(BaseModel):
    material: str
    tool_diameter_mm: float
    n_flutes: int
    coolant_mode: int
    tool_age_min: float
    machine_on_min: float
    ambient_temp_c: float
    spindle_speed_rpm: float       # current settings
    feed_per_tooth_mm: float
    axial_depth_mm: float
    radial_width_mm: float
    current_cut_time_min: float = 15.0   # how long the operation takes today


@app.post("/optimize")
def optimize(job: Job):
    """Recommend spindle speed / feed / depths that cut predicted energy in the same cycle time.
    All savings are PREDICTED by the model or SIMULATED; they are not measured."""
    mrr = job.axial_depth_mm * job.radial_width_mm * job.feed_per_tooth_mm * job.n_flutes * job.spindle_speed_rpm
    spec = pd.Series(dict(
        material=job.material, tool_diameter_mm=job.tool_diameter_mm, n_flutes=job.n_flutes, coolant_mode=job.coolant_mode,
        tool_age_min=job.tool_age_min, machine_on_min=job.machine_on_min, ambient_temp_c=job.ambient_temp_c,
        n_base=job.spindle_speed_rpm, fz_base=job.feed_per_tooth_mm, ap_base=job.axial_depth_mm, ae_base=job.radial_width_mm,
        volume_mm3=mrr * job.current_cut_time_min))
    r = O.optimise_job(model, spec, takt_mult=1.0, trust_mrr_cm3=_ref["trust_mrr_cm3_min"], objective="window")
    return {
        "recommended": {"spindle_speed_rpm": round(float(r["n_opt"]), 0), "feed_per_tooth_mm": round(float(r["fz_opt"]), 4),
                        "axial_depth_mm": round(float(r["ap_opt"]), 2), "radial_width_mm": round(float(r["ae_opt"]), 2)},
        "predicted_energy_kwh": {"current": round(r["e_pred_base"], 3), "recommended": round(r["e_pred_opt"], 3)},
        "predicted_saving_pct": round(100 * (r["e_pred_base"] - r["e_pred_opt"]) / r["e_pred_base"], 1),
        "simulated_saving_pct": round(100 * (r["e_true_base"] - r["e_true_opt"]) / r["e_true_base"], 1),
        "cycle_time_min": {"current": round(r["t_op_base_min"], 1), "recommended": round(r["t_op_opt_min"], 1)},
        "feasible_solution_found": bool(r["feasible_found"]),
        "note": "Predicted/simulated values only, not measured on a real machine.",
    }


def _on_reading(payload):
    """Called for every machine reading that arrives: score it and keep it for the dashboard."""
    w = MeasuredWindow.model_validate(payload)
    df = pd.DataFrame([w.model_dump()])
    pred, resid, z, flag = _score(df)
    LIVE.append({
        "seq": payload.get("seq"), "ts": payload.get("ts"), "job_id": payload.get("job_id"), "material": w.material,
        "spindle_speed_rpm": w.spindle_speed_rpm, "measured_kw": round(w.power_kw, 3),
        "predicted_kw": round(float(pred[0]), 3), "excess_kw": round(float(resid[0]), 3),
        "z_score": round(float(z[0]), 2), "flagged": bool(flag[0]),
    })


@app.get("/live")
def live(limit: int = 120):
    """Latest machine readings received from the IoT feed, already scored by the model."""
    items = list(LIVE)[-limit:]
    return {
        "enabled": _listener is not None,
        "connected": bool(_listener and _listener.connected),
        "received_total": _listener.received if _listener else 0,
        "last_error": _listener.last_error if _listener else None,
        "flagged_in_view": sum(1 for i in items if i["flagged"]),
        "items": items,
    }
