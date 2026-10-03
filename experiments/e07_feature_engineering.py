"""E07 - Feature engineering (generic process features, then physics features) and re-tuning.
python -m experiments.e07_feature_engineering"""
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src import config as C, data_prep as dp, experiment as E, features as F, models as M

EXP, S = "E07", C.SEED


def ridge(deg):
    s = [StandardScaler()] + ([PolynomialFeatures(deg, include_bias=False), StandardScaler()] if deg > 1 else [])
    return make_pipeline(*s, Ridge())


def levels_run(dataset, d, levels):
    lgb_p = E.get_params("E06", dataset, "lightgbm_tuned")
    rf_p = E.get_params("E05", dataset, "random_forest_tuned")
    ents = []
    for lvl, fb in levels.items():
        for deg, cx in [(1, 2), (2, 3)]:
            ents.append(E.tune_and_run(f"{EXP}a", dataset, d, f"ridge_{'poly2' if deg == 2 else 'linear'}__{lvl}", ridge(deg),
                                       {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}, fb, "linear", cx, n_iter=6))
        ents.append(E.run_model(f"{EXP}b", dataset, M.SkModel(f"random_forest_tuned_params__{lvl}", RandomForestRegressor(300, n_jobs=8, random_state=S, **rf_p), fb, "forest", 6), d,
                                params=rf_p, notes="hyper-parameters tuned on the raw commanded features (E05b), reused"))
        ents.append(E.run_model(f"{EXP}c", dataset, M.SkModel(f"lightgbm_default__{lvl}", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1), fb, "boosting", 6), d))
        ents.append(E.run_model(f"{EXP}c", dataset, M.SkModel(f"lightgbm_tuned_params__{lvl}", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, **lgb_p), fb, "boosting", 6), d,
                                params=lgb_p, notes="hyper-parameters tuned on the raw commanded features (E06d), reused"))
    return ents


def main():
    d = dp.load_synthetic()
    ents = levels_run("synthetic", d, {"generic_fe": F.fe_generic, "physics_fe": F.fe_physics})
    lgb = LGBMRegressor(n_jobs=8, random_state=S, verbose=-1)
    dist = {"n_estimators": [200, 400, 800], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63, 127], "min_child_samples": [5, 20, 50],
            "subsample": [0.7, 0.9, 1.0], "subsample_freq": [1], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]}
    ents.append(E.tune_and_run(f"{EXP}d", "synthetic", d, "lightgbm_retuned__physics_fe", lgb, dist, F.fe_physics, "boosting", 6, n_iter=25,
                               notes="hyper-parameters re-tuned on the physics feature set"))
    # diagnostic (NOT a planning model): what do the drive's own load signals add?
    lgb_p = E.get_params("E06", "synthetic", "lightgbm_tuned")
    m = M.SkModel("DIAGNOSTIC_lightgbm_commanded+drive_load_signals", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, **lgb_p), F.commanded_plus_signals, "diagnostic", 6)
    ents.append(E.run_model(f"{EXP}e", "synthetic", m, d, params={"diagnostic": True}, notes="monitoring-only; not usable for planning/optimisation; excluded from the candidate pool"))
    E.save(EXP, "synthetic", ents)
    for key in ["ibarmia", "gmtk"]:
        r = dp.load_real(key)
        E.save(EXP, key, levels_run(key, r, {"real_fe": F.real_fe}))


if __name__ == "__main__":
    main()
