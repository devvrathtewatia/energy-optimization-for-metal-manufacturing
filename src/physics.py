"""First-principles model of a machining centre's electrical power.

P_el = P_base + P_coolant + P_conveyor                       (auxiliaries)
     + [P_cut + P_idle(n) + P_cu(T)] / eta_inv               (spindle drive, mech + losses)
     + P_servo + b*v_axis + F_f*v_f                          (feed axes)

with
    MRR   = a_p * a_e * v_f,     v_f = f_z * z * n
    h_m   = f_z * sqrt(a_e / D)                              (mean chip thickness, a_e/D <~ 0.5 approx.)
    k_c   = k_c1.1 * h_m^(-m_c)                              (Kienzle specific cutting force)
    P_cut = k_c * MRR / 6e7                                  [kW; k_c in N/mm^2, MRR in mm^3/min]
    T     = 9550 * P_cut / n                                 [Nm]
    P_idle= (l0 + l1*n + l2*n^2) * warm-up factor            (no-load spindle loss)
    P_cu  = r_cu * T^2 / 1000                                (copper loss)

Everything is vectorised over numpy arrays. The same function is the (noise-free) world
model inside the synthetic generator (with hidden factors) and the zero-fit handbook model
(with catalogue constants and no hidden factors).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def taylor_life_min(vc, fz, diameter, vc_ref):
    """Extended Taylor tool life [min] (nominal tool)."""
    fz_ref = C.TAYLOR_FZ_REF_RATIO * np.asarray(diameter, dtype=float)
    fz_eff = np.where(np.asarray(fz) > 0, fz, fz_ref)
    vc = np.maximum(np.asarray(vc, dtype=float), 1e-6)
    return C.TAYLOR_T_REF_MIN * (vc_ref / vc) ** (1.0 / C.TAYLOR_N) * (fz_ref / fz_eff) ** C.TAYLOR_FZ_EXP


def cutting_speed_m_min(n, diameter):
    return np.pi * np.asarray(diameter, dtype=float) * np.asarray(n, dtype=float) / 1000.0


def feed_velocity(fz, z, n):
    return np.asarray(fz) * np.asarray(z) * np.asarray(n)


def mrr_mm3_min(ap, ae, vf):
    return np.asarray(ap) * np.asarray(ae) * np.asarray(vf)


def mean_chip_thickness(fz, ae, diameter):
    ratio = np.clip(np.asarray(ae, dtype=float) / np.asarray(diameter, dtype=float), 1e-6, 1.0)
    return np.asarray(fz) * np.sqrt(ratio)


def kienzle_kc(kc11, mc, hm):
    return kc11 * np.maximum(hm, 0.004) ** (-mc)


def cutting_power_kw(kc, mrr):
    return kc * mrr / 6.0e7


def idle_loss_kw(n, machine, warm=1.0, bearing_mult=1.0):
    n = np.asarray(n, dtype=float)
    base = machine["idle_l0"] + machine["idle_l1"] * n + machine["idle_l2"] * n ** 2
    return np.where(n > 0, base * warm * bearing_mult, 0.0)


def warm_factor(machine_on_min, machine):
    return 1.0 + machine["warm_amp"] * np.exp(-np.asarray(machine_on_min, dtype=float) / machine["warm_tau_min"])


def wear_vb_mm(tool_age_min, life_min):
    frac = np.clip(np.asarray(tool_age_min) / np.maximum(life_min, 1e-6), 0.0, 1.3)
    return 0.3 * frac ** 0.8


def wear_force_multiplier(vb_mm):
    return 1.0 + 0.45 * (np.asarray(vb_mm) / 0.3) ** 1.3


def power_components(
    machine: dict,
    n, axis_speed, fz, z, ap, ae, diameter,
    kc11, mc, coolant_mode, machine_on_min, ambient_c,
    force_mult=1.0,          # lot * wear * anomaly multipliers on k_c (hidden in the generator)
    bearing_mult=1.0, drag_mult=1.0, extra_aux_kw=0.0, extra_coolant_kw=0.0,
    engagement=1.0,          # window-level fluctuation of the cutting load
):
    """Return dict of power components [kW] (noise-free). All inputs broadcastable arrays."""
    n = np.asarray(n, dtype=float)
    vf = feed_velocity(fz, z, n)
    engaged = (np.asarray(ap) > 0) & (np.asarray(ae) > 0) & (vf > 0) & (n > 0)
    mrr = np.where(engaged, mrr_mm3_min(ap, ae, vf), 0.0)
    hm = mean_chip_thickness(fz, ae, diameter)
    kc = kienzle_kc(kc11, mc, hm) * force_mult
    p_cut = np.where(engaged, cutting_power_kw(kc, mrr) * engagement, 0.0)
    torque = np.where(n > 0, 9550.0 * p_cut / np.maximum(n, 1.0), 0.0)
    warm = warm_factor(machine_on_min, machine)
    p_idle = idle_loss_kw(n, machine, warm, bearing_mult)
    p_cu = machine["r_cu"] * torque ** 2 / 1000.0
    p_spindle = np.where(n > 0, (p_cut + p_idle + p_cu) / machine["eta_inv"], machine["p_spindle_off_kw"])

    cool_lookup = np.array([machine["p_coolant_kw"][0], machine["p_coolant_kw"][1], machine["p_coolant_kw"][2]])
    p_cool = np.where(n > 0, cool_lookup[np.asarray(coolant_mode, dtype=int)] + extra_coolant_kw, 0.0)
    p_conv = np.where(engaged, machine["p_conveyor_kw"], 0.0)
    p_base = machine["p_base_kw"] + machine["p_amb_kw_per_c"] * (np.asarray(ambient_c) - 20.0) + extra_aux_kw

    vc = cutting_speed_m_min(n, diameter)
    f_cut = np.where(engaged, p_cut * 1000.0 / np.maximum(vc / 60.0, 1e-6), 0.0)  # N
    p_feed_force = machine["feed_force_ratio"] * f_cut * vf / 60000.0 / 1000.0
    p_axes = machine["p_servo_kw"] + machine["b_axis_kw_per_mm_min"] * np.asarray(axis_speed) * drag_mult + p_feed_force

    total = p_base + p_cool + p_conv + p_spindle + p_axes
    return dict(
        total=total, base=p_base, coolant=p_cool, conveyor=p_conv, spindle=p_spindle,
        cut=p_cut / machine["eta_inv"], idle=p_idle / machine["eta_inv"], cu=p_cu / machine["eta_inv"],
        axes=p_axes, mrr=mrr, vf=vf, torque=torque, engaged=engaged, p_cut_mech=p_cut,
    )


# ---------------------------------------------------------------------------------------
# Handbook (zero-fit) prediction straight from a dataframe of commanded conditions
# ---------------------------------------------------------------------------------------
def handbook_power_kw(df, machine=None):
    """Zero-parameter forward model: catalogue constants, no wear, no warm-up, no fitting."""
    machine = machine or C.MACHINE_DATASHEET
    kc11 = df["material"].map({m: v["kc11"] for m, v in C.MATERIALS_HANDBOOK.items()}).to_numpy(float)
    mc = df["material"].map({m: v["mc"] for m, v in C.MATERIALS_HANDBOOK.items()}).to_numpy(float)
    comp = power_components(
        machine,
        n=df["spindle_speed_rpm"].to_numpy(float), axis_speed=df["axis_speed_mm_min"].to_numpy(float),
        fz=df["feed_per_tooth_mm"].to_numpy(float), z=df["n_flutes"].to_numpy(float),
        ap=df["axial_depth_mm"].to_numpy(float), ae=df["radial_width_mm"].to_numpy(float),
        diameter=df["tool_diameter_mm"].to_numpy(float), kc11=kc11, mc=mc,
        coolant_mode=df["coolant_mode"].to_numpy(int), machine_on_min=df["machine_on_min"].to_numpy(float),
        ambient_c=np.full(len(df), 20.0),
    )
    return comp["total"]


def spindle_power_limit_kw(n, machine=None):
    """Mechanical spindle power available at speed n: torque-limited below base speed, power-limited above."""
    machine = machine or C.MACHINE_TRUE
    n = np.asarray(n, dtype=float)
    return np.minimum(machine["p_rated_kw"], machine["t_rated_nm"] * n / 9550.0)
