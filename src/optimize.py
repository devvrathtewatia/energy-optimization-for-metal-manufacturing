"""Constrained energy minimisation for a milling operation.  ALL savings computed here are
PREDICTED (by an ML/physics model) or SIMULATED (by the noise-free synthetic-world simulator). They are not
measurements from a real factory.

Problem (per job):
    decision variables x = (n [rpm], f_z [mm/tooth], a_p [mm], a_e [mm])
    fixed by the job: material, tool (D, z), coolant mode, volume to remove V, machine/tool context
    E_op(x)  = P_cut(x) * t_cut(x) + N_pass(x) * t_pos * P_rapid(x)                 [kWh, the operation itself]
    E_win(x) = E_op(x) + P_standby * max(0, t_window - t_op(x))                       [kWh, over the fixed takt window]
       t_cut  = V / MRR(x),  N_pass = V / (a_p * a_e * L_pass),  t_op = t_cut + N_pass * t_pos
       t_window = takt_mult * t_op(baseline)
    minimise E_win (headline: finishing early does NOT make the machine free, it idles at standby) or E_op
    (upper bound: valid only if the freed time is used productively).
    subject to
       t_op(x) <= t_window                       (production requirement: same part within the same takt)
       P_cut,mech(x) * 1.25 <= 0.85 * P_spindle_limit(n)      (spindle power & torque, catalogue k_c + wear margin)
       T_life(v_c, f_z) >= 1.2 * t_cut           (tool must survive the operation)
       bounds: v_c window of the material, f_z, a_p, a_e ranges (= the ranges covered by the training data)
       [trust region] MRR(x) <= max MRR seen in training  (optional; removed in the extended-range study)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

from . import config as C
from . import physics as ph

T_POS_S = 2.5          # repositioning (rapid) time between passes [s]
L_PASS_MM = 400.0      # tool path length of one pass [mm]
LIFE_MARGIN = 1.2
PCUT_SAFETY = 1.25
POWER_FRAC = 0.85
HB = C.MATERIALS_HANDBOOK


def job_specs(df: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """One row per job with its CURRENT (baseline) operating point, derived from the recorded windows."""
    cut = df[df["engaged"]]
    g = cut.groupby("job_id")
    jobs = pd.DataFrame(dict(
        material=g["material"].first(), tool_diameter_mm=g["tool_diameter_mm"].first(), n_flutes=g["n_flutes"].first(),
        coolant_mode=g["coolant_mode"].first(), tool_age_min=g["tool_age_min"].median(), machine_on_min=g["machine_on_min"].median(),
        ambient_temp_c=g["ambient_temp_c"].first(), n_base=g["spindle_speed_rpm"].median(), fz_base=g["feed_per_tooth_mm"].first(),
        ap_base=g["axial_depth_mm"].first(), ae_base=g["radial_width_mm"].first(), split=g["split"].first(),
    )).reset_index()
    rng = np.random.default_rng(seed)
    jobs["t_base_min"] = rng.uniform(5, 30, len(jobs))                       # baseline cutting time of the operation
    mrr = jobs.ap_base * jobs.ae_base * jobs.fz_base * jobs.n_flutes * jobs.n_base   # mm3/min
    jobs["mrr_base_cm3_min"] = mrr / 1000.0
    jobs["volume_mm3"] = mrr * jobs.t_base_min
    return jobs


def bounds(job) -> np.ndarray:
    M, D = C.MATERIALS[job["material"]], job["tool_diameter_mm"]
    n_lo = max(600.0, 1000 * M["vc_min"] / (np.pi * D))
    n_hi = min(C.MACHINE_TRUE["n_max_rpm"], 1000 * M["vc_max"] / (np.pi * D))
    if n_lo >= n_hi:                                   # tool too small to reach vc window: allow around the baseline
        n_lo, n_hi = max(600.0, 0.8 * job["n_base"]), min(C.MACHINE_TRUE["n_max_rpm"], 1.2 * job["n_base"])
    fz = D * np.array([0.003, 0.013]) * M["fz_scale"]
    ap = D * np.array([0.25, 1.5]) * M["ap_scale"]
    ae = D * np.array([0.08, 0.65])
    return np.array([[n_lo, n_hi], fz, ap, ae])


def _unit(n_pts, seed, d=4):
    return qmc.Sobol(d, scramble=True, seed=seed).random(n_pts)


def _frames(job, X, state):
    n, fz, ap, ae = X.T
    N = len(X)
    base = dict(material=job["material"], tool_diameter_mm=float(job["tool_diameter_mm"]), n_flutes=int(job["n_flutes"]),
                coolant_mode=int(job["coolant_mode"]), tool_age_min=float(job["tool_age_min"]),
                machine_on_min=float(job["machine_on_min"]), ambient_temp_c=float(job["ambient_temp_c"]))
    df = pd.DataFrame({k: np.full(N, v) for k, v in base.items()})
    df["spindle_speed_rpm"] = n
    if state == "cut":
        df["feed_per_tooth_mm"], df["axial_depth_mm"], df["radial_width_mm"] = fz, ap, ae
        df["axis_speed_mm_min"] = fz * job["n_flutes"] * n
    elif state == "standby":  # spindle stopped, axes still, coolant off
        df["spindle_speed_rpm"], df["coolant_mode"] = 0.0, 0
        df["feed_per_tooth_mm"], df["axial_depth_mm"], df["radial_width_mm"], df["axis_speed_mm_min"] = 0.0, 0.0, 0.0, 0.0
    else:  # rapid repositioning between passes, spindle keeps turning
        df["feed_per_tooth_mm"], df["axial_depth_mm"], df["radial_width_mm"] = 0.0, 0.0, 0.0
        df["axis_speed_mm_min"] = C.MACHINE_TRUE["rapid_mm_min"]
    return df


def _geometry(job, X):
    n, fz, ap, ae = X.T
    vf = fz * job["n_flutes"] * n
    mrr = ap * ae * vf                                   # mm3/min
    t_cut_s = job["volume_mm3"] / mrr * 60.0
    n_pass = job["volume_mm3"] / (ap * ae * L_PASS_MM)
    return vf, mrr, t_cut_s, n_pass


def t_op_s(job, X):
    _, _, t_cut_s, n_pass = _geometry(job, X)
    return t_cut_s + n_pass * T_POS_S


def t_window_s(job, takt_mult=1.0):
    x0 = np.array([[job["n_base"], job["fz_base"], job["ap_base"], job["ae_base"]]])
    return takt_mult * float(t_op_s(job, x0)[0])


def _assemble(job, X, p_cut, p_rapid, p_stby, win_s, objective):
    _, _, t_cut_s, n_pass = _geometry(job, X)
    e = (p_cut * t_cut_s + n_pass * T_POS_S * p_rapid) / 3600.0        # kWh, operation only
    if objective == "window":
        e = e + p_stby * np.maximum(win_s - (t_cut_s + n_pass * T_POS_S), 0.0) / 3600.0
    return e


def ml_energy(model, job, X, win_s=None, objective="window") -> np.ndarray:
    win_s = t_window_s(job) if win_s is None else win_s
    p_stby = float(np.asarray(model.predict(_frames(job, X[:1], "standby")))[0])
    return _assemble(job, X, np.asarray(model.predict(_frames(job, X, "cut"))), np.asarray(model.predict(_frames(job, X, "rapid"))), p_stby, win_s, objective)


def truth_energy(job, X, win_s=None, objective="window") -> np.ndarray:
    """Noise-free nominal simulator (lot = sharpness = 1, no anomaly); tool wear taken from the job's current tool."""
    win_s = t_window_s(job) if win_s is None else win_s
    mach, M = C.MACHINE_TRUE, C.MATERIALS[job["material"]]
    D, z = float(job["tool_diameter_mm"]), float(job["n_flutes"])
    vc0 = ph.cutting_speed_m_min(job["n_base"], D)
    life0 = ph.taylor_life_min(vc0, job["fz_base"], D, M["vc_ref"])
    wear = ph.wear_force_multiplier(ph.wear_vb_mm(job["tool_age_min"], life0))
    out = []
    for state in ("cut", "rapid", "standby"):
        f = _frames(job, X if state != "standby" else X[:1], state)
        comp = ph.power_components(
            mach, f.spindle_speed_rpm.to_numpy(), f.axis_speed_mm_min.to_numpy(), f.feed_per_tooth_mm.to_numpy(), z,
            f.axial_depth_mm.to_numpy(), f.radial_width_mm.to_numpy(), D, M["kc11"], M["mc"], f.coolant_mode.to_numpy(int),
            float(job["machine_on_min"]), float(job["ambient_temp_c"]), force_mult=wear)
        out.append(comp["total"])
    return _assemble(job, X, out[0], out[1], float(out[2][0]), win_s, objective)


def feasibility(job, X, takt_mult=1.0, trust_mrr_cm3=None):
    n, fz, ap, ae = X.T
    D, z = float(job["tool_diameter_mm"]), float(job["n_flutes"])
    M, H = C.MATERIALS[job["material"]], HB[job["material"]]
    vf, mrr, t_cut_s, _ = _geometry(job, X)
    hm = ph.mean_chip_thickness(fz, ae, D)
    p_mech = ph.cutting_power_kw(ph.kienzle_kc(H["kc11"], H["mc"], hm), mrr)
    vc = ph.cutting_speed_m_min(n, D)
    life = ph.taylor_life_min(vc, fz, D, M["vc_ref"])
    ok = t_op_s(job, X) <= t_window_s(job, takt_mult) * (1 + 1e-9)
    ok &= PCUT_SAFETY * p_mech <= POWER_FRAC * ph.spindle_power_limit_kw(n)
    ok &= life >= LIFE_MARGIN * t_cut_s / 60.0
    ok &= ap <= 1.5 * D
    if trust_mrr_cm3 is not None:
        ok &= mrr <= trust_mrr_cm3 * 1000.0
    return ok


def true_overload(job, X):
    """Would the TRUE (generator) spindle power exceed 100 % of the limit? (verification only)"""
    n, fz, ap, ae = X.T
    M = C.MATERIALS[job["material"]]
    D = float(job["tool_diameter_mm"])
    mrr = ap * ae * fz * job["n_flutes"] * n
    p = ph.cutting_power_kw(ph.kienzle_kc(M["kc11"], M["mc"], ph.mean_chip_thickness(fz, ae, D)), mrr)
    return p > ph.spindle_power_limit_kw(n)


def optimise_job(model, job, takt_mult=1.0, trust_mrr_cm3=None, objective="window", n0=4096, n_refine=2048, rounds=2, seed=C.SEED) -> dict:
    lo_hi = bounds(job)
    x_base = np.array([[job["n_base"], job["fz_base"], job["ap_base"], job["ae_base"]]])
    cands = [x_base]
    lo, hi = lo_hi[:, 0].copy(), lo_hi[:, 1].copy()
    win = t_window_s(job, takt_mult)
    best_x, best_e = None, np.inf
    base_ok = bool(feasibility(job, x_base, takt_mult, trust_mrr_cm3)[0])
    pool = lo + _unit(n0, seed) * (hi - lo)
    pool = np.vstack([pool, x_base])
    for r in range(rounds + 1):
        ok = feasibility(job, pool, takt_mult, trust_mrr_cm3)
        if ok.any():
            Xf = pool[ok]
            e = ml_energy(model, job, Xf, win, objective)
            i = int(np.argmin(e))
            if e[i] < best_e:
                best_e, best_x = float(e[i]), Xf[i].copy()
        if r == rounds:
            break
        if best_x is None:
            break
        width = (lo_hi[:, 1] - lo_hi[:, 0]) * 0.15 * (0.5 ** r)
        lo2, hi2 = np.maximum(lo_hi[:, 0], best_x - width), np.minimum(lo_hi[:, 1], best_x + width)
        pool = lo2 + _unit(n_refine, seed + 1 + r) * (hi2 - lo2)
    found = best_x is not None
    x_opt = best_x if found else x_base[0]
    xb, xo = x_base, x_opt[None, :]
    n_, fz_, ap_, ae_ = x_opt
    D_, M_, H_ = float(job["tool_diameter_mm"]), C.MATERIALS[job["material"]], HB[job["material"]]
    mrr_ = ap_ * ae_ * fz_ * job["n_flutes"] * n_
    p_mech_ = ph.cutting_power_kw(ph.kienzle_kc(H_["kc11"], H_["mc"], ph.mean_chip_thickness(fz_, ae_, D_)), mrr_)
    life_ = ph.taylor_life_min(ph.cutting_speed_m_min(n_, D_), fz_, D_, M_["vc_ref"])
    out = dict(
        power_util_opt=float(PCUT_SAFETY * p_mech_ / (POWER_FRAC * ph.spindle_power_limit_kw(n_))),
        life_ratio_opt=float(life_ / (LIFE_MARGIN * job["volume_mm3"] / mrr_)),
        trust_ratio_opt=float(mrr_ / (trust_mrr_cm3 * 1000.0)) if trust_mrr_cm3 else float("nan"),
        at_lower_bound=int(np.sum(np.isclose(x_opt, lo_hi[:, 0], rtol=1e-3))), at_upper_bound=int(np.sum(np.isclose(x_opt, lo_hi[:, 1], rtol=1e-3))),
        feasible_found=found, baseline_feasible=base_ok, objective=objective,
        n_opt=x_opt[0], fz_opt=x_opt[1], ap_opt=x_opt[2], ae_opt=x_opt[3],
        e_pred_base=float(ml_energy(model, job, xb, win, objective)[0]), e_pred_opt=float(ml_energy(model, job, xo, win, objective)[0]),
        e_true_base=float(truth_energy(job, xb, win, objective)[0]), e_true_opt=float(truth_energy(job, xo, win, objective)[0]),
        eop_true_base=float(truth_energy(job, xb, win, "operation")[0]), eop_true_opt=float(truth_energy(job, xo, win, "operation")[0]),
        eop_pred_base=float(ml_energy(model, job, xb, win, "operation")[0]), eop_pred_opt=float(ml_energy(model, job, xo, win, "operation")[0]),
        t_op_base_min=float(t_op_s(job, xb)[0] / 60), t_op_opt_min=float(t_op_s(job, xo)[0] / 60),
        mrr_opt_cm3=float(x_opt[2] * x_opt[3] * x_opt[1] * job["n_flutes"] * x_opt[0] / 1000.0),
        true_overload_opt=bool(true_overload(job, xo)[0]), true_overload_base=bool(true_overload(job, xb)[0]),
    )
    return out
