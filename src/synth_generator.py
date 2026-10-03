"""Physics-informed SYNTHETIC dataset of a machining centre (NOT real industrial data).

Each row is a 10 s steady-state window of a milling job. Power comes from ``physics.power_components``
with hidden factors a model cannot observe directly (workpiece lot variation, tool-specific life,
cold-start friction, injected inefficiencies) plus sensor noise, quantisation and glitches.

Columns prefixed ``gt_`` are ground truth for evaluation ONLY and must never be used as features.

Run:  python -m src.synth_generator
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C
from . import physics as ph

STATE_NAMES = {0: "standby", 1: "rapid", 2: "aircut", 3: "cutting"}
ANOMALY_TYPES = ["coolant_stuck_hp", "excess_tool_wear", "spindle_bearing_friction", "aux_leak", "axis_drag"]


def _sample_jobs(rng: np.random.Generator, n_jobs: int) -> pd.DataFrame:
    M = C.MATERIALS
    mats = np.array(C.MATERIAL_LIST)
    mat = rng.choice(mats, size=n_jobs, p=[M[m]["p"] for m in mats])
    g = lambda key: np.array([M[m][key] for m in mat], dtype=float)
    D = rng.choice(C.TOOL_DIAMETERS, size=n_jobs, p=C.TOOL_D_P).astype(float)
    z = np.where(mat == "Al6061", rng.choice([2, 3], size=n_jobs, p=[0.3, 0.7]),
                 rng.choice([3, 4, 5, 6], size=n_jobs, p=[0.15, 0.35, 0.30, 0.20])).astype(float)
    vc = g("vc_min") + (g("vc_max") - g("vc_min")) * rng.beta(2.2, 2.2, n_jobs)
    n = np.clip(1000.0 * vc / (np.pi * D), 600.0, C.MACHINE_TRUE["n_max_rpm"])
    fz = D * rng.uniform(0.003, 0.013, n_jobs) * g("fz_scale")
    ap = D * rng.uniform(0.25, 1.5, n_jobs) * g("ap_scale")
    ae = D * rng.uniform(0.08, 0.65, n_jobs)

    # keep every job feasible on the machine (spindle power/torque limit with margin for wear/lot)
    kc11, mc = g("kc11"), g("mc")
    for _ in range(80):
        vf = ph.feed_velocity(fz, z, n)
        comp = ph.power_components(
            C.MACHINE_TRUE, n, vf, fz, z, ap, ae, D, kc11, mc, np.zeros(n_jobs, int),
            np.full(n_jobs, 300.0), np.full(n_jobs, 20.0), force_mult=1.25)
        over = comp["p_cut_mech"] > 0.85 * ph.spindle_power_limit_kw(n)
        if not over.any():
            break
        ap = np.where(over, ap * 0.88, ap)
        ae = np.where(over, ae * 0.95, ae)

    vc = ph.cutting_speed_m_min(n, D)
    vc_ref = g("vc_ref")
    t_nom = ph.taylor_life_min(vc, fz, D, vc_ref)
    life_factor = rng.lognormal(0.0, 0.25, n_jobs)
    tool_age = t_nom * rng.beta(1.4, 1.6, n_jobs)
    vb = ph.wear_vb_mm(tool_age, t_nom * life_factor)

    cool = np.zeros(n_jobs, int)
    for i, m in enumerate(mat):
        p = {"Al6061": [0.35, 0.65, 0.0], "C45": [0.0, 0.85, 0.15],
             "SS316L": [0.0, 0.60, 0.40], "Ti6Al4V": [0.0, 0.30, 0.70]}[m]
        cool[i] = rng.choice(3, p=p)

    on0 = np.where(rng.random(n_jobs) < 0.25, rng.uniform(0, 60, n_jobs), rng.uniform(60, 600, n_jobs))
    jobs = pd.DataFrame(dict(
        job_id=np.arange(n_jobs), material=mat, tool_diameter_mm=D, n_flutes=z, n_cmd=n, fz=fz, ap=ap, ae=ae,
        coolant_mode=cool, tool_age0=tool_age, on0=on0,
        ambient_temp_c=np.clip(rng.normal(24, 4, n_jobs), 14, 36),
        kc11=kc11, mc=mc, lot=rng.lognormal(0, 0.06, n_jobs), sharp=rng.lognormal(0, 0.05, n_jobs),
        wear_mult=ph.wear_force_multiplier(vb), vb=vb,
    ))
    jobs["job_mrr_cm3_min"] = ap * ae * ph.feed_velocity(fz, z, n) / 1000.0

    # injected inefficiencies (job-level, persistent)
    anom = np.array(["none"] * n_jobs, dtype=object)
    has = rng.random(n_jobs) < 0.07
    choice = rng.choice(ANOMALY_TYPES, size=n_jobs, p=[0.25, 0.25, 0.20, 0.20, 0.10])
    choice = np.where((choice == "coolant_stuck_hp") & (cool >= 2), "aux_leak", choice)
    anom[has] = choice[has]
    jobs["job_anomaly"] = anom
    jobs["anom_force"] = np.where(anom == "excess_tool_wear", rng.uniform(1.35, 1.8, n_jobs), 1.0)
    jobs["anom_bearing"] = np.where(anom == "spindle_bearing_friction", rng.uniform(1.8, 2.8, n_jobs), 1.0)
    jobs["anom_aux_kw"] = np.where(anom == "aux_leak", rng.uniform(0.9, 2.0, n_jobs), 0.0)
    jobs["anom_drag"] = np.where(anom == "axis_drag", rng.uniform(2.0, 3.5, n_jobs), 1.0)
    jobs["anom_cool"] = anom == "coolant_stuck_hp"
    return jobs


def _expand_windows(rng: np.random.Generator, jobs: pd.DataFrame) -> pd.DataFrame:
    counts = np.c_[
        rng.choice([0, 1, 2], size=len(jobs), p=[0.4, 0.4, 0.2]),   # standby
        rng.integers(1, 3, len(jobs)),                               # rapid
        rng.integers(1, 3, len(jobs)),                               # air cut
        rng.integers(4, 11, len(jobs)),                              # cutting
    ]
    reps = counts.sum(1)
    idx = np.repeat(np.arange(len(jobs)), reps)
    w = jobs.iloc[idx].reset_index(drop=True)
    state = np.concatenate([np.repeat([0, 1, 2, 3], c) for c in counts])
    w["gt_state"] = state
    w["window_idx"] = np.concatenate([np.arange(r) for r in reps])
    return w


def generate(n_jobs: int = 2500, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    jobs = _sample_jobs(rng, n_jobs)
    w = _expand_windows(rng, jobs)
    N = len(w)
    st = w["gt_state"].to_numpy()
    spind_on, feeding, cutting = st > 0, st >= 2, st == 3
    mach = C.MACHINE_TRUE

    n_cmd = np.where(spind_on, w["n_cmd"].to_numpy(), 0.0)
    vf = ph.feed_velocity(w["fz"].to_numpy(), w["n_flutes"].to_numpy(), n_cmd)
    axis = np.where(st == 1, mach["rapid_mm_min"], np.where(feeding, vf, 0.0))
    fz_w = np.where(feeding, w["fz"].to_numpy(), 0.0)
    ap_w = np.where(cutting, w["ap"].to_numpy(), 0.0)
    ae_w = np.where(cutting, w["ae"].to_numpy(), 0.0)
    cool_w = np.where(spind_on, w["coolant_mode"].to_numpy(), 0)
    on_w = w["on0"].to_numpy() + w["window_idx"].to_numpy() * C.WINDOW_S / 60.0
    age_w = w["tool_age0"].to_numpy() + np.where(cutting, w["window_idx"].to_numpy() * C.WINDOW_S / 60.0, 0.0)
    engagement = np.where(cutting, rng.normal(1.0, 0.04, N), 1.0)

    force_true = w["lot"].to_numpy() * w["sharp"].to_numpy() * w["wear_mult"].to_numpy() * w["anom_force"].to_numpy()
    force_nominal = w["lot"].to_numpy() * w["sharp"].to_numpy() * w["wear_mult"].to_numpy()
    common = dict(n=n_cmd, axis_speed=axis, fz=fz_w, z=w["n_flutes"].to_numpy(), ap=ap_w, ae=ae_w,
                  diameter=w["tool_diameter_mm"].to_numpy(), kc11=w["kc11"].to_numpy(), mc=w["mc"].to_numpy(),
                  coolant_mode=cool_w, machine_on_min=on_w, ambient_c=w["ambient_temp_c"].to_numpy(),
                  engagement=engagement)
    cool_kw = np.array([mach["p_coolant_kw"][0], mach["p_coolant_kw"][1], mach["p_coolant_kw"][2]])[cool_w.astype(int)]
    extra_cool = np.where(w["anom_cool"].to_numpy() & spind_on, np.maximum(mach["p_coolant_kw"][2] - cool_kw, 0.0), 0.0)
    comp = ph.power_components(mach, force_mult=force_true, bearing_mult=w["anom_bearing"].to_numpy(),
                               drag_mult=w["anom_drag"].to_numpy(), extra_aux_kw=w["anom_aux_kw"].to_numpy(),
                               extra_coolant_kw=extra_cool, **common)
    nominal = ph.power_components(mach, force_mult=force_nominal, **common)
    p_true = comp["total"]
    extra = p_true - nominal["total"]

    # --- sensors -------------------------------------------------------------------------
    p_meas = p_true * (1 + rng.normal(0, 0.015, N)) + rng.normal(0, 0.04, N)
    glitch = rng.random(N) < 0.007
    is_dropout = glitch & (rng.random(N) < 0.3)
    p_meas = np.where(glitch & ~is_dropout, p_meas * rng.uniform(1.6, 3.0, N), p_meas)
    p_meas = np.where(is_dropout, 0.0, p_meas)
    p_meas = np.round(np.maximum(p_meas, 0.0), 2)

    volt = 400.0 * (1 + rng.normal(0, 0.012, N))
    pf = np.clip(0.42 + 0.5 * (1 - np.exp(-p_true / 6.0)) + rng.normal(0, 0.01, N), 0.3, 0.95)
    cur = p_true * 1000.0 / (np.sqrt(3) * volt * pf) * (1 + rng.normal(0, 0.005, N))
    load_sp = 100.0 * (comp["p_cut_mech"] + comp["idle"] * mach["eta_inv"]) / mach["p_rated_kw"]
    load_sp = np.where(spind_on, load_sp * (1 + rng.normal(0, 0.03, N)) + rng.normal(0, 0.3, N), 0.0)
    load_ax = np.maximum(100.0 * (comp["axes"] - mach["p_servo_kw"]) / 4.0 + rng.normal(0, 0.5, N), 0.0)

    out = pd.DataFrame(dict(
        job_id=w["job_id"], window_idx=w["window_idx"],
        material=w["material"],
        spindle_speed_rpm=np.where(spind_on, n_cmd * (1 + rng.normal(0, 0.003, N)), 0.0),
        axis_speed_mm_min=np.where(axis > 0, axis * (1 + rng.normal(0, 0.01, N)), 0.0),
        feed_per_tooth_mm=fz_w, axial_depth_mm=ap_w, radial_width_mm=ae_w,
        tool_diameter_mm=w["tool_diameter_mm"], n_flutes=w["n_flutes"].astype(int),
        coolant_mode=cool_w.astype(int), tool_age_min=age_w, machine_on_min=on_w,
        ambient_temp_c=w["ambient_temp_c"],
        spindle_load_pct=load_sp, feed_axes_load_pct=load_ax,
        voltage_v=volt, current_a=cur, power_factor=pf,
        power_kw=p_meas,
        job_mrr_cm3_min=w["job_mrr_cm3_min"],
        gt_state=[STATE_NAMES[s] for s in st], gt_job_anomaly=w["job_anomaly"],
        gt_extra_kw=extra, gt_power_true_kw=p_true, gt_power_nominal_kw=nominal["total"],
        gt_is_inefficient=(extra >= np.maximum(0.4, 0.10 * nominal["total"])),
        gt_is_glitch=glitch, gt_wear_vb=w["vb"],
        gt_p_base=comp["base"], gt_p_coolant=comp["coolant"] + comp["conveyor"],
        gt_p_idle=comp["idle"] + np.where(spind_on, 0.0, mach["p_spindle_off_kw"]),
        gt_p_cut=comp["cut"] + comp["cu"], gt_p_axes=comp["axes"],
    ))
    for c in ["tool_age_min", "machine_on_min", "spindle_speed_rpm", "axis_speed_mm_min", "spindle_load_pct",
              "feed_axes_load_pct", "voltage_v", "current_a", "power_factor", "gt_extra_kw",
              "gt_power_true_kw", "gt_power_nominal_kw", "gt_p_base", "gt_p_coolant", "gt_p_idle", "gt_p_cut",
              "gt_p_axes", "job_mrr_cm3_min", "ambient_temp_c"]:
        out[c] = out[c].round(4)
    out["energy_wh"] = (out["power_kw"] * C.WINDOW_S / 3.6).round(3)   # E = P * dt
    return out


def main(n_jobs: int = 2500, seed: int = C.SEED) -> None:
    C.DATA_SYNTH.mkdir(parents=True, exist_ok=True)
    df = generate(n_jobs, seed)
    path = C.DATA_SYNTH / "synthetic_machining_windows.csv"
    df.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = dict(kind="SYNTHETIC - physics-informed simulation, not industrial data", seed=seed, n_jobs=n_jobs,
                n_rows=len(df), window_seconds=C.WINDOW_S, sha256=digest,
                machine_true=C.MACHINE_TRUE, materials=C.MATERIALS,
                note="Columns prefixed gt_ are ground truth for evaluation only.")
    (C.DATA_SYNTH / "synthetic_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {path} rows={len(df)} sha256={digest[:12]}")


if __name__ == "__main__":
    main()
