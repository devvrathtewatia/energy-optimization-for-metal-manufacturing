"""E04 decision tree, E05 random forest, E06 XGBoost / LightGBM.  python -m experiments.e04_e06_trees"""
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src import config as C, data_prep as dp, experiment as E, features as F, models as M

S = C.SEED


def run(dataset, d, fb):
    ents = {"E04": [], "E05": [], "E06": []}
    # ---- E04 decision tree -----------------------------------------------------------
    ents["E04"].append(E.run_model("E04a", dataset, M.SkModel("decision_tree_unconstrained", DecisionTreeRegressor(random_state=S), fb, "tree", 5), d,
                                   notes="no regularisation on purpose: shows overfitting"))
    ents["E04"].append(E.tune_and_run("E04b", dataset, d, "decision_tree_tuned", DecisionTreeRegressor(random_state=S),
                                      {"max_depth": [3, 5, 8, 12, 16, None], "min_samples_leaf": [1, 5, 20, 50, 100]}, fb, "tree", 5, n_iter=30))
    # ---- E05 random forest -----------------------------------------------------------
    ents["E05"].append(E.run_model("E05a", dataset, M.SkModel("random_forest_default", RandomForestRegressor(300, n_jobs=8, random_state=S), fb, "forest", 6), d))
    ents["E05"].append(E.tune_and_run("E05b", dataset, d, "random_forest_tuned", RandomForestRegressor(300, n_jobs=8, random_state=S),
                                      {"max_depth": [8, 12, 16, 24, None], "min_samples_leaf": [1, 2, 5, 10, 20], "max_features": [0.3, 0.5, 0.8, 1.0]}, fb, "forest", 6, n_iter=12))
    # ---- E06 gradient boosting -------------------------------------------------------
    xgb = XGBRegressor(n_jobs=8, random_state=S, tree_method="hist", verbosity=0)
    lgb = LGBMRegressor(n_jobs=8, random_state=S, verbose=-1)
    ents["E06"].append(E.run_model("E06a", dataset, M.SkModel("xgboost_default", XGBRegressor(n_jobs=8, random_state=S, tree_method="hist", verbosity=0), fb, "boosting", 6), d))
    ents["E06"].append(E.tune_and_run("E06b", dataset, d, "xgboost_tuned", xgb,
                                      {"n_estimators": [200, 400, 800], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 5, 7, 9],
                                       "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 20], "reg_lambda": [0.1, 1.0, 10.0]},
                                      fb, "boosting", 6, n_iter=25))
    ents["E06"].append(E.run_model("E06c", dataset, M.SkModel("lightgbm_default", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1), fb, "boosting", 6), d))
    lgb_dist = {"n_estimators": [200, 400, 800], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63, 127],
                "min_child_samples": [5, 20, 50], "subsample": [0.7, 0.9, 1.0], "subsample_freq": [1], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0.0, 1.0, 10.0]}
    ents["E06"].append(E.tune_and_run("E06d", dataset, d, "lightgbm_tuned", lgb, lgb_dist, fb, "boosting", 6, n_iter=25))
    best = {k: v for k, v in ents["E06"][-1]["params"].items() if k != "cv_rmse_train"}
    for obj in ["huber", "l1"]:
        m = M.SkModel(f"lightgbm_tuned_{obj}_loss", LGBMRegressor(n_jobs=8, random_state=S, verbose=-1, objective=obj, **best), fb, "boosting", 6)
        ents["E06"].append(E.run_model("E06e", dataset, m, d, params={**best, "objective": obj}, notes="same hyper-parameters as E06d, robust loss"))
    for k, v in ents.items():
        E.save(k, dataset, v)


if __name__ == "__main__":
    run("synthetic", dp.load_synthetic(), F.commanded)
    for key in ["ibarmia", "gmtk"]:
        run(key, dp.load_real(key), F.real_base)
