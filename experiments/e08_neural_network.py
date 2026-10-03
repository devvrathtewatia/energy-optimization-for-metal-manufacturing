"""E08 - Does deep learning beat strong classical ML?  MLP (PyTorch), 5 seeds, same train data, same splits.
python -m experiments.e08_neural_network"""
import copy
import time

import numpy as np

from src import config as C, data_prep as dp, experiment as E, features as F, models as M

EXP, SEEDS = "E08", [0, 1, 2, 3, 4]
GRID = [dict(width=w, depth=dp_, lr=lr, weight_decay=wd, dropout=do) for w, dp_, lr, wd, do in
        [(64, 2, 3e-3, 1e-5, 0.0), (128, 3, 2e-3, 1e-4, 0.0), (256, 3, 1e-3, 1e-4, 0.0), (128, 4, 2e-3, 1e-3, 0.0),
         (256, 4, 1e-3, 1e-5, 0.1), (64, 3, 3e-3, 1e-3, 0.0), (128, 2, 1e-3, 1e-5, 0.0), (256, 2, 2e-3, 1e-4, 0.1)]]


def agg(metric_list):
    """mean/std across seeds for every split/subset/metric"""
    mean, std = {}, {}
    for sp in metric_list[0]:
        mean[sp], std[sp] = {}, {}
        for sub in metric_list[0][sp]:
            mean[sp][sub], std[sp][sub] = {}, {}
            for k in metric_list[0][sp][sub]:
                v = np.array([m[sp][sub][k] for m in metric_list], float)
                mean[sp][sub][k], std[sp][sub][k] = float(v.mean()), float(v.std())
    return mean, std


def run(dataset, d, fb, tag, loss="mse"):
    tr = d[d.split == "train"]
    # --- small random search; criterion = internal early-stopping MSE on a held-out slice of TRAIN jobs
    scores = []
    for g in GRID:
        t0 = time.time()
        m = M.TorchMLP("probe", fb, seed=0, loss=loss, **g).fit(tr)
        scores.append(m.best_internal_mse)
        print(f"   probe {g} internal_mse={m.best_internal_mse:.4f} epochs={m.epochs_run} ({time.time()-t0:.0f}s)", flush=True)
    best = GRID[int(np.argmin(scores))]
    print(f"[{EXP}] {dataset} {tag}: best config {best}", flush=True)
    # --- final: 5 seeds
    per_seed, models = [], []
    for s in SEEDS:
        m = M.TorchMLP(f"mlp_{tag}", fb, seed=s, loss=loss, **best)
        m.fit(tr)
        ev = E.evaluate(m, d, dataset)
        per_seed.append(ev)
        models.append(m)
        print(f"   seed {s}: val MAE={ev['val']['all']['mae']:.3f} epochs={m.epochs_run}", flush=True)
    mean, std = agg(per_seed)
    if dataset == "synthetic":
        import joblib
        C.CACHE.mkdir(parents=True, exist_ok=True)
        joblib.dump(models[0], C.CACHE / f"mlp_{tag}.joblib")

    ens = M.MeanEnsemble(f"mlp_{tag}_ensemble5", models)
    if dataset == "synthetic":
        joblib.dump(ens, C.CACHE / f"{ens.name}.joblib")
    ent = dict(exp_id=f"{EXP}b", dataset=dataset, model=f"mlp_{tag}", family="neural_network", complexity=8,
               params={**best, "seeds": SEEDS, "probe_internal_mse": scores, "loss": loss}, fit_seconds=0.0,
               metrics=mean, metrics_std=std, notes="metrics = mean over 5 seeds; metrics_std = std over seeds")
    ent_e = dict(exp_id=f"{EXP}c", dataset=dataset, model=ens.name, family="neural_network", complexity=8,
                 params={**best, "seeds": SEEDS}, fit_seconds=0.0, metrics=E.evaluate(ens, d, dataset),
                 notes="prediction = mean of the 5 seed networks")
    for e in (ent, ent_e):
        v = e["metrics"]["val"]["all"]
        print(f"[{e['exp_id']}] {dataset:9s} {e['model']:36s} val MAE={v['mae']:.3f} RMSE={v['rmse']:.3f} R2={v['r2']:.3f}", flush=True)
    return [ent, ent_e]


def main():
    d = dp.load_synthetic()
    ents = run("synthetic", d, F.commanded, "commanded")
    ents += run("synthetic", d, F.fe_physics, "physics_fe")
    E.save(EXP, "synthetic", ents)
    for key in ["ibarmia", "gmtk"]:
        r = dp.load_real(key)
        E.save(EXP, key, run(key, r, F.real_base, "raw") + run(key, r, F.real_fe, "fe"))


if __name__ == "__main__":
    main()
