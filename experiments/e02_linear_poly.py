"""E02 - Linear and polynomial regression.  python -m experiments.e02_linear_poly"""
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from src import data_prep as dp, experiment as E, features as F, models as M

EXP = "E02"


def ridge_pipe(deg):
    steps = [StandardScaler()]
    if deg > 1:
        steps += [PolynomialFeatures(deg, include_bias=False), StandardScaler()]
    return make_pipeline(*steps, Ridge())


def run(dataset, d, fb):
    tr = d[d.split == "train"]
    entries = []
    for deg, cx in [(1, 2), (2, 3), (3, 3)]:
        est = ridge_pipe(deg)
        best, cv = M.tune(est, {"ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}, fb, tr, n_iter=6)
        est.set_params(**best)
        name = {1: "ridge_linear", 2: "ridge_poly2", 3: "ridge_poly3"}[deg]
        entries.append(E.run_model(f"{EXP}{'abc'[deg-1]}", dataset, M.SkModel(name, est, fb, "linear", cx), d,
                                   params={**best, "cv_rmse_train": cv, "degree": deg}))
    E.save(EXP, dataset, entries)


def main():
    run("synthetic", dp.load_synthetic(), F.commanded)
    for k in ["ibarmia", "gmtk"]:
        run(k, dp.load_real(k), F.real_base)


if __name__ == "__main__":
    main()
