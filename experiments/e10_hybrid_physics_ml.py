"""E10 - Physics-informed / hybrid models.  python -m experiments.e10_hybrid_physics_ml"""
import copy

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from src import config as C, data_prep as dp, experiment as E, features as F, models as M
from experiments.e00_physics_baseline import real_noload

EXP, S = "E10", C.SEED
DIST = {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [7, 15, 31, 63], "min_child_samples": [20, 50, 100],
        "subsample": [0.7, 0.9], "subsample_freq": [1], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [1.0, 10.0]}


def truth_table(phys):
    mt, mats, eta = C.MACHINE_TRUE, C.MATERIALS, C.MACHINE_TRUE["eta_inv"]
    p = phys.params()
    rows = [
        ("p0 (base + servo) [kW]", p["p0"], mt["p_base_kw"] + mt["p_servo_kw"]),
        ("ambient coeff [kW/degC]", p["a_amb"], mt["p_amb_kw_per_c"]),
        ("flood coolant [kW]", p["c_flood"], mt["p_coolant_kw"][1]),
        ("high-pressure coolant [kW]", p["c_hp"], mt["p_coolant_kw"][2]),
        ("chip conveyor [kW]", p["p_conv"], mt["p_conveyor_kw"]),
        ("idle loss l0 [kW]", p["l0"], mt["idle_l0"] / eta),
        ("idle loss l1 [kW/rpm]", p["l1"], mt["idle_l1"] / eta),
        ("idle loss l2 [kW/rpm^2]", p["l2"], mt["idle_l2"] / eta),
        ("warm-up amplitude", p["warm_amp"], mt["warm_amp"]),
        ("warm-up tau [min]", p["tau"], mt["warm_tau_min"]),
        ("copper loss r_cu", p["r_cu"], mt["r_cu"] / eta),
    ]
    for m in C.MATERIAL_LIST:
        rows.append((f"k_c1.1 {m} [N/mm2]", p[f"kc_{m}"], mats[m]["kc11"] / eta))
    for m in C.MATERIAL_LIST:
        rows.append((f"m_c {m}", p[f"mc_{m}"], mats[m]["mc"]))
    rows.append(("wear slope", p["wear"], 0.45))
    t = pd.DataFrame(rows, columns=["parameter", "fitted", "generator_truth_(effective)"])
    t["rel_error_%"] = (100 * (t.fitted - t.iloc[:, 2]) / t.iloc[:, 2].abs()).round(1)
    return t


def synthetic():
    d = dp.load_synthetic()
    tr = d[d.split == "train"]
    ents = []
    phys = M.PhysParamModel()
    ents.append(E.run_model(f"{EXP}a", "synthetic", phys, d, notes="robust (soft-L1) non-linear least squares, 21 coefficients, started from catalogue values"))
    truth_table(phys).to_csv(C.TABLES / "e10_param_recovery.csv", index=False)
    ents.append(E.run_model(f"{EXP}b", "synthetic", M.PhysParamModel("physics_parametric_fitted_squared_loss", loss="linear"), d,
                            notes="ablation: ordinary least squares instead of robust loss"))
    # residual hybrid: tune the correction learner on the physics model's training residuals
    res_tr = tr.assign(resid=tr.power_kw - phys.predict(tr))
    best, cv = M.tune(LGBMRegressor(n_jobs=8, random_state=S, verbose=-1), DIST, F.commanded, res_tr, n_iter=20, target="resid")
    ents.append(E.run_model(f"{EXP}c", "synthetic", M.ResidualHybrid("hybrid_physics+lgbm_residual", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, **best), F.commanded), d,
                            params={**best, "cv_rmse_resid": cv}, notes="physics_parametric + LightGBM on its residuals (commanded features)"))
    lgb_p = E.get_params("E06", "synthetic", "lightgbm_tuned")
    ents.append(E.run_model(f"{EXP}d", "synthetic", M.StackHybrid("hybrid_lgbm_with_physics_feature", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, **lgb_p), F.fe_generic), d,
                            params=lgb_p, notes="LightGBM (E06d hyper-parameters) given the fitted physics prediction as an extra feature"))
    E.save(EXP, "synthetic", ents)


def real():
    for key in ["ibarmia", "gmtk"]:
        d = dp.load_real(key)
        base = M.SkModel("noload_curve", LinearRegression(), real_noload, "physics", 1)
        tr = d[d.split == "train"]
        res_tr = tr.assign(resid=tr.power_kw - base.fit(tr).predict(tr))
        best, cv = M.tune(LGBMRegressor(n_jobs=8, random_state=S, verbose=-1), DIST, F.real_base, res_tr, n_iter=20, target="resid")
        ents = [E.run_model(f"{EXP}c", key, M.ResidualHybrid("hybrid_noload_curve+lgbm_residual", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, **best), F.real_base, base=base), d,
                            params=best, notes="no-load loss curve + LightGBM on its residuals")]
        E.save(EXP, key, ents)


if __name__ == "__main__":
    C.TABLES.mkdir(parents=True, exist_ok=True)
    synthetic()
    real()
