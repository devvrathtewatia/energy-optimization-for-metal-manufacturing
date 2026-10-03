"""Small web API that serves the fitted physics-parametric CNC power model.

Run locally:   uvicorn deploy.api:app --port 8000      (from the repo root)
Run on EC2:    same command, with --host 0.0.0.0
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import PhysParamModel  # noqa: E402  (the model class from this repo)

# Rebuild the fitted model from the saved coefficients (no pickle, so no version problems).
_saved = json.loads((Path(__file__).parent / "model_params.json").read_text())
model = PhysParamModel()
model.theta_ = np.array([_saved["params"][n] for n in PhysParamModel.NAMES])

app = FastAPI(title="CNC milling power model (simulated data)")


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


@app.get("/health")
def health():
    return {"status": "ok", "model": _saved["model_name"]}


@app.post("/predict")
def predict(req: Request):
    df = pd.DataFrame([w.model_dump() for w in req.windows])
    kw = model.predict(df)
    return {"predicted_power_kw": [round(float(v), 4) for v in kw]}
