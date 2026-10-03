"""E00 - Physics baselines (and null baseline).  python -m experiments.e00_physics_baseline"""
import numpy as np
from sklearn.linear_model import LinearRegression

from src import config as C, data_prep as dp, experiment as E, features as F, models as M

EXP = "E00"


def real_noload(df):
    n = df["speed_SPINDLE"].abs()
    return __import__("pandas").DataFrame({"abs_speed": n, "speed_sq": n ** 2})


def main():
    dp.prepare_synthetic()
    dp.prepare_real()
    d = dp.load_synthetic()
    entries = [
        E.run_model(EXP + "a", "synthetic", M.MeanModel(), d, notes="null baseline: predict the training mean"),
        E.run_model(EXP + "b", "synthetic", M.SkModel("sec_P0+k*MRR_fitted", LinearRegression(), F.sec_feature, "physics", 1), d,
                    notes="Gutowski specific-energy model, 2 fitted parameters"),
        E.run_model(EXP + "c", "synthetic", M.HandbookModel(), d,
                    notes="zero-fit forward model from catalogue constants (no wear, no warm-up, no ambient)"),
    ]
    E.save(EXP, "synthetic", entries)
    for key in ["ibarmia", "gmtk"]:
        r = dp.load_real(key)
        ent = [
            E.run_model(EXP + "a", key, M.MeanModel(), r, notes="null baseline"),
            E.run_model(EXP + "b", key, M.SkModel("noload_loss_curve_c0+c1|n|+c2n^2", LinearRegression(), real_noload, "physics", 1), r,
                        notes="physics baseline for the real data: spindle no-load loss curve in speed only"),
        ]
        E.save(EXP, key, ent)


if __name__ == "__main__":
    main()
