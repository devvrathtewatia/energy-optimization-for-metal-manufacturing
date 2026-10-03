"""Feature builders. All take a raw dataframe and return a numeric DataFrame with FIXED columns
(so train/val/test/OOD always line up)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import physics as ph

HB = C.MATERIALS_HANDBOOK


def commanded(df: pd.DataFrame) -> pd.DataFrame:
    X = df[C.COMMANDED_NUM].astype(float).copy()
    for m in C.MATERIAL_LIST:
        X[f"mat_{m}"] = (df["material"] == m).astype(float)
    return X


def commanded_plus_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: commanded features + the drive's own load signals (monitoring, not planning)."""
    return pd.concat([commanded(df), df[C.SIGNAL_COLS]], axis=1)


def fe_generic(df: pd.DataFrame) -> pd.DataFrame:
    """Material-agnostic engineered features (process knowledge, no material constants)."""
    X = commanded(df)
    n = df["spindle_speed_rpm"].to_numpy(float)
    vf = df["feed_per_tooth_mm"].to_numpy(float) * df["n_flutes"].to_numpy(float) * n
    mrr = df["axial_depth_mm"].to_numpy(float) * df["radial_width_mm"].to_numpy(float) * vf / 1000.0  # cm3/min
    X["feed_velocity_mm_min"] = vf
    X["mrr_cm3_min"] = mrr
    X["log1p_mrr"] = np.log1p(mrr)
    X["cutting_speed_m_min"] = ph.cutting_speed_m_min(n, df["tool_diameter_mm"].to_numpy(float))
    X["n_sq_1e6"] = n ** 2 / 1e6
    X["spindle_on"] = (n > 0).astype(float)
    X["engaged"] = (df["axial_depth_mm"].to_numpy() > 0).astype(float)
    X["feeding"] = (df["feed_per_tooth_mm"].to_numpy() > 0).astype(float)
    X["rapid"] = (df["axis_speed_mm_min"].to_numpy() > 0.9 * C.MACHINE_TRUE["rapid_mm_min"]).astype(float)
    return X


def fe_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Adds catalogue-physics quantities (Kienzle, torque, idle loss, coolant, wear, warm-up)."""
    X = fe_generic(df)
    n = df["spindle_speed_rpm"].to_numpy(float)
    D = df["tool_diameter_mm"].to_numpy(float)
    fz = df["feed_per_tooth_mm"].to_numpy(float)
    kc11 = df["material"].map({m: v["kc11"] for m, v in HB.items()}).to_numpy(float)
    mc = df["material"].map({m: v["mc"] for m, v in HB.items()}).to_numpy(float)
    vcref = df["material"].map({m: v["vc_ref"] for m, v in C.MATERIALS.items()}).to_numpy(float)
    hm = np.where(X["engaged"] > 0, ph.mean_chip_thickness(fz, df["radial_width_mm"].to_numpy(float), D), 0.0)
    kc = np.where(X["engaged"] > 0, ph.kienzle_kc(kc11, mc, hm), 0.0)
    p_cut = ph.cutting_power_kw(kc, X["mrr_cm3_min"].to_numpy() * 1000.0)
    ds = C.MACHINE_DATASHEET
    X["mean_chip_thickness_mm"] = hm
    X["kienzle_kc_n_mm2"] = kc
    X["p_cut_handbook_kw"] = p_cut
    X["torque_est_nm"] = np.where(n > 0, 9550.0 * p_cut / np.maximum(n, 1.0), 0.0)
    X["torque_est_sq"] = X["torque_est_nm"] ** 2
    X["p_idle_handbook_kw"] = ph.idle_loss_kw(n, ds)
    X["p_coolant_handbook_kw"] = np.where(n > 0, np.array([ds["p_coolant_kw"][k] for k in (0, 1, 2)])[df["coolant_mode"].to_numpy(int)], 0.0)
    life = ph.taylor_life_min(X["cutting_speed_m_min"].to_numpy(), fz, D, vcref)
    X["life_fraction"] = np.where(X["engaged"] > 0, df["tool_age_min"].to_numpy(float) / life, 0.0)
    X["warm_decay"] = np.exp(-df["machine_on_min"].to_numpy(float) / 40.0)
    return X


def sec_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Single feature of the Gutowski specific-energy model: material removal rate [cm3/min]."""
    return fe_generic(df)[["mrr_cm3_min"]]


def regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Unsupervised descriptors used for K-Means (NO target, NO ground-truth state)."""
    X = fe_generic(df)
    return X[["spindle_speed_rpm", "axis_speed_mm_min", "log1p_mrr", "coolant_mode"]]


# ---------------------------------------------------------------------------------------
# Real (CFAA) data: only speed, override, two axis loads and Z-axis power are available
# ---------------------------------------------------------------------------------------
REAL_BASE = ["speed_SPINDLE", "override_SPINDLE", "load_X", "load_Z", "power_Z"]


def real_base(df: pd.DataFrame) -> pd.DataFrame:
    return df[REAL_BASE].astype(float).copy()


def real_fe(df: pd.DataFrame) -> pd.DataFrame:
    X = real_base(df)
    X["abs_speed"] = X["speed_SPINDLE"].abs()
    X["speed_sq_1e6"] = X["speed_SPINDLE"] ** 2 / 1e6
    X["abs_power_Z"] = X["power_Z"].abs()
    X["abs_speed_x_override"] = X["abs_speed"] * X["override_SPINDLE"] / 100.0
    return X


def real_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    X = real_fe(df)
    return X[["abs_speed", "override_SPINDLE", "load_X", "load_Z", "abs_power_Z"]]
