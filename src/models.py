"""Model wrappers. Every model exposes ``fit(df)`` / ``predict(df)`` on the RAW dataframe so feature
building lives inside the model (no accidental use of information from other splits)."""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import physics as ph


class DFModel:
    name = "model"
    family = "model"
    complexity = 0  # used ONLY as tie-breaker by the pre-registered selection rule

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class MeanModel(DFModel):
    name, family, complexity = "mean", "null", 0

    def fit(self, df, y=None):
        self.mu = float(np.mean(df[C.TARGET] if y is None else y))
        return self

    def predict(self, df):
        return np.full(len(df), self.mu)


class HandbookModel(DFModel):
    """Zero-fit first-principles forward model with catalogue constants."""
    name, family, complexity = "handbook_physics_zero_fit", "physics", 1

    def fit(self, df, y=None):
        return self

    def predict(self, df):
        return ph.handbook_power_kw(df)


class SkModel(DFModel):
    def __init__(self, name, estimator, fb: Callable, family="sklearn", complexity=2, target=C.TARGET):
        self.name, self.estimator, self.fb, self.family, self.complexity, self.target = name, estimator, fb, family, complexity, target

    def fit(self, df, y=None):
        X = self.fb(df)
        self.cols = list(X.columns)
        self.estimator = clone(self.estimator)
        self.estimator.fit(X, df[self.target] if y is None else y)
        return self

    def predict(self, df):
        return np.asarray(self.estimator.predict(self.fb(df)[self.cols]))


# ---------------------------------------------------------------------------------------
# Parametric physics model fitted to data
# ---------------------------------------------------------------------------------------
class PhysParamModel(DFModel):
    """Structured equation (same functional forms as physics.py) with ~21 free coefficients fitted by
    robust non-linear least squares, started from CATALOGUE values (not the generator's truth)."""
    name, family, complexity = "physics_parametric_fitted", "physics", 3
    NAMES = ["p0", "a_amb", "b_axis", "c_flood", "c_hp", "p_conv", "l0", "l1", "l2", "warm_amp", "tau",
             "r_cu", "kc_Al6061", "kc_C45", "kc_SS316L", "kc_Ti6Al4V",
             "mc_Al6061", "mc_C45", "mc_SS316L", "mc_Ti6Al4V", "wear"]
    SCALE = np.array([1, 0.1, 1e-5, 1, 1, 1, 1, 1e-4, 1e-8, 1, 10, 0.1] + [1000.0] * 4 + [1.0] * 4 + [1.0])

    def __init__(self, name=None, loss="soft_l1"):
        if name:
            self.name = name
        self.loss = loss

    def _arrays(self, df):
        mat = df["material"].map({m: i for i, m in enumerate(C.MATERIAL_LIST)}).to_numpy(int)
        vcref = np.array([C.MATERIALS[m]["vc_ref"] for m in C.MATERIAL_LIST])[mat]
        n = df["spindle_speed_rpm"].to_numpy(float)
        D = df["tool_diameter_mm"].to_numpy(float)
        fz = df["feed_per_tooth_mm"].to_numpy(float)
        vc = ph.cutting_speed_m_min(n, D)
        life = ph.taylor_life_min(vc, fz, D, vcref)
        ap, ae = df["axial_depth_mm"].to_numpy(float), df["radial_width_mm"].to_numpy(float)
        vf = fz * df["n_flutes"].to_numpy(float) * n
        return dict(
            n=n, axis=df["axis_speed_mm_min"].to_numpy(float), fz=fz, ap=ap, ae=ae, D=D, mat=mat,
            cool=df["coolant_mode"].to_numpy(int), on=df["machine_on_min"].to_numpy(float),
            amb=df["ambient_temp_c"].to_numpy(float), vf=vf,
            engaged=(ap > 0) & (ae > 0) & (vf > 0) & (n > 0),
            life_frac=np.where((ap > 0), df["tool_age_min"].to_numpy(float) / life, 0.0),
        )

    def _f(self, a, x):
        th = x * self.SCALE
        p0, a_amb, b_axis, c_fl, c_hp, p_conv, l0, l1, l2, wa, tau, rcu = th[:12]
        kc11, mc, wear = th[12:16], th[16:20], th[20]
        hm = a["fz"] * np.sqrt(np.clip(a["ae"] / a["D"], 1e-6, 1.0))
        kc = kc11[a["mat"]] * np.maximum(hm, 0.004) ** (-mc[a["mat"]]) * (1 + wear * a["life_frac"])
        mrr = a["ap"] * a["ae"] * a["vf"]
        pcut = np.where(a["engaged"], kc * mrr / 6e7, 0.0)
        on = a["n"] > 0
        torque = np.where(on, 9550.0 * pcut / np.maximum(a["n"], 1.0), 0.0)
        idle = np.where(on, (l0 + l1 * a["n"] + l2 * a["n"] ** 2) * (1 + wa * np.exp(-a["on"] / tau)), 0.0)
        spindle = np.where(on, pcut + idle + rcu * torque ** 2 / 1000.0, 0.08)
        cool = np.where(on, np.array([0.0, c_fl, c_hp])[a["cool"]], 0.0)
        conv = np.where(a["engaged"], p_conv, 0.0)
        return p0 + a_amb * (a["amb"] - 20.0) + cool + conv + spindle + b_axis * a["axis"]

    def components(self, df):
        """Decompose the fitted prediction (used by the explainability comparison)."""
        a = self._arrays(df)
        th = self.theta_
        p0, a_amb, b_axis, c_fl, c_hp, p_conv, l0, l1, l2, wa, tau, rcu = th[:12]
        kc11, mc, wear = th[12:16], th[16:20], th[20]
        hm = a["fz"] * np.sqrt(np.clip(a["ae"] / a["D"], 1e-6, 1.0))
        kc = kc11[a["mat"]] * np.maximum(hm, 0.004) ** (-mc[a["mat"]]) * (1 + wear * a["life_frac"])
        pcut = np.where(a["engaged"], kc * a["ap"] * a["ae"] * a["vf"] / 6e7, 0.0)
        on = a["n"] > 0
        torque = np.where(on, 9550.0 * pcut / np.maximum(a["n"], 1.0), 0.0)
        idle = np.where(on, (l0 + l1 * a["n"] + l2 * a["n"] ** 2) * (1 + wa * np.exp(-a["on"] / tau)), 0.0)
        return dict(
            base=np.full(len(df), p0) + a_amb * (a["amb"] - 20.0),
            coolant=np.where(on, np.array([0.0, c_fl, c_hp])[a["cool"]], 0.0) + np.where(a["engaged"], p_conv, 0.0),
            idle=idle + np.where(on, 0.0, 0.08), cut=pcut + rcu * torque ** 2 / 1000.0, axes=b_axis * a["axis"])

    def fit(self, df, y=None):
        y = df[C.TARGET].to_numpy(float) if y is None else np.asarray(y, float)
        a = self._arrays(df)
        ds = C.MACHINE_DATASHEET
        th0 = np.array([
            ds["p_base_kw"] + ds["p_servo_kw"], 0.0, ds["b_axis_kw_per_mm_min"],
            ds["p_coolant_kw"][1], ds["p_coolant_kw"][2], ds["p_conveyor_kw"],
            ds["idle_l0"], ds["idle_l1"], ds["idle_l2"], 0.1, 30.0, ds["r_cu"],
            *[C.MATERIALS_HANDBOOK[m]["kc11"] for m in C.MATERIAL_LIST],
            *[C.MATERIALS_HANDBOOK[m]["mc"] for m in C.MATERIAL_LIST], 0.3])
        lb = np.array([0.5, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0] + [200] * 4 + [0.05] * 4 + [0.0])
        ub = np.array([6, 0.2, 2e-4, 4, 8, 2, 1.5, 1e-3, 1e-7, 1.5, 200, 0.5] + [4000] * 4 + [0.6] * 4 + [2.0])
        res = least_squares(lambda x: self._f(a, x) - y, th0 / self.SCALE, bounds=(lb / self.SCALE, ub / self.SCALE),
                            loss=self.loss, f_scale=0.5, x_scale="jac", max_nfev=400)
        self.theta_ = res.x * self.SCALE
        return self

    def predict(self, df):
        return self._f(self._arrays(df), self.theta_ / self.SCALE)

    def params(self):
        return dict(zip(self.NAMES, [float(v) for v in self.theta_]))


class ResidualHybrid(DFModel):
    """base model (default: fitted parametric physics) + ML correction of its residuals."""
    family, complexity = "hybrid", 7

    def __init__(self, name, estimator, fb, base: DFModel | None = None):
        self.name, self.estimator, self.fb = name, estimator, fb
        self.base = base or PhysParamModel()

    def fit(self, df, y=None):
        self.base = copy.deepcopy(self.base).fit(df)
        r = df[C.TARGET].to_numpy(float) - self.base.predict(df)
        X = self.fb(df)
        self.cols = list(X.columns)
        self.estimator = clone(self.estimator).fit(X, r)
        return self

    def predict(self, df):
        return self.base.predict(df) + self.estimator.predict(self.fb(df)[self.cols])


class StackHybrid(DFModel):
    """ML model that receives the fitted physics prediction as an additional feature."""
    family, complexity = "hybrid", 7

    def __init__(self, name, estimator, fb, phys: PhysParamModel | None = None):
        self.name, self.estimator, self.fb = name, estimator, fb
        self.phys = phys or PhysParamModel()

    def _X(self, df):
        X = self.fb(df).copy()
        X["phys_pred_kw"] = self.phys.predict(df)
        return X

    def fit(self, df, y=None):
        self.phys = copy.deepcopy(self.phys).fit(df)
        X = self._X(df)
        self.cols = list(X.columns)
        self.estimator = clone(self.estimator).fit(X, df[C.TARGET])
        return self

    def predict(self, df):
        return self.estimator.predict(self._X(df)[self.cols])


# ---------------------------------------------------------------------------------------
# K-Means regime models
# ---------------------------------------------------------------------------------------
class KMeansRegimes:
    """Fit on TRAIN rows only; assigns any row to the nearest centroid."""

    def __init__(self, fb, k, seed=C.SEED):
        self.fb, self.k, self.seed = fb, k, seed

    def fit(self, df):
        X = self.fb(df)
        self.scaler = StandardScaler().fit(X)
        self.km = KMeans(self.k, n_init=20, random_state=self.seed).fit(self.scaler.transform(X))
        return self

    def predict(self, df):
        return self.km.predict(self.scaler.transform(self.fb(df)))


class RegimeModel(DFModel):
    """mode='onehot': base estimator gets the cluster one-hot; mode='per_cluster': one estimator per cluster."""
    family, complexity = "kmeans+model", 4

    def __init__(self, name, estimator, fb, regime_fb, k, mode, min_rows=60):
        self.name, self.estimator, self.fb, self.regime_fb, self.k, self.mode, self.min_rows = name, estimator, fb, regime_fb, k, mode, min_rows

    def _X(self, df, lab):
        X = self.fb(df).copy()
        if self.mode == "onehot":
            for j in range(self.k):
                X[f"regime_{j}"] = (lab == j).astype(float)
        return X

    def fit(self, df, y=None):
        self.reg = KMeansRegimes(self.regime_fb, self.k).fit(df)
        lab = self.reg.predict(df)
        yv = df[C.TARGET].to_numpy(float)
        X = self._X(df, lab)
        self.cols = list(X.columns)
        if self.mode == "onehot":
            self.est = clone(self.estimator).fit(X, yv)
        else:
            self.global_est = clone(self.estimator).fit(X, yv)
            self.est = {}
            for j in range(self.k):
                m = lab == j
                self.est[j] = clone(self.estimator).fit(X[m], yv[m]) if m.sum() >= self.min_rows else None
        return self

    def predict(self, df):
        lab = self.reg.predict(df)
        X = self._X(df, lab)[self.cols]
        if self.mode == "onehot":
            return np.asarray(self.est.predict(X))
        out = np.asarray(self.global_est.predict(X), float)
        for j, e in self.est.items():
            m = lab == j
            if e is not None and m.any():
                out[m] = e.predict(X[m])
        return out


# ---------------------------------------------------------------------------------------
# Neural network (PyTorch MLP)
# ---------------------------------------------------------------------------------------
class TorchMLP(DFModel):
    family, complexity = "neural_network", 8

    def __init__(self, name, fb, width=128, depth=3, lr=2e-3, weight_decay=1e-4, dropout=0.0,
                 epochs=250, patience=25, batch=256, seed=0, loss="mse", target=C.TARGET):
        self.name, self.fb = name, fb
        self.width, self.depth, self.lr, self.wd, self.dropout = width, depth, lr, weight_decay, dropout
        self.epochs, self.patience, self.batch, self.seed, self.loss, self.target = epochs, patience, batch, seed, loss, target

    def _net(self, d):
        import torch.nn as nn
        layers, k = [], d
        for _ in range(self.depth):
            layers += [nn.Linear(k, self.width), nn.ReLU()] + ([nn.Dropout(self.dropout)] if self.dropout else [])
            k = self.width
        layers.append(nn.Linear(k, 1))
        return nn.Sequential(*layers)

    def fit(self, df, y=None):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        rng = np.random.default_rng(self.seed)
        X = self.fb(df)
        self.cols = list(X.columns)
        yv = df[self.target].to_numpy(float) if y is None else np.asarray(y, float)
        # internal early-stopping holdout (10 % of jobs, or rows if no job id) - never the outer val split
        if "job_id" in df:
            jobs = df["job_id"].unique()
            hold = rng.choice(jobs, size=max(1, int(0.1 * len(jobs))), replace=False)
            m_hold = df["job_id"].isin(hold).to_numpy()
        else:
            m_hold = rng.random(len(df)) < 0.1
        self.xs = StandardScaler().fit(X[~m_hold])
        self.ym, self.ys = yv[~m_hold].mean(), yv[~m_hold].std()
        Xt = torch.tensor(self.xs.transform(X), dtype=torch.float32)
        yt = torch.tensor(((yv - self.ym) / self.ys)[:, None], dtype=torch.float32)
        tr, ho = torch.tensor(np.where(~m_hold)[0]), torch.tensor(np.where(m_hold)[0])
        self.net = self._net(Xt.shape[1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
        lossf = torch.nn.MSELoss() if self.loss == "mse" else torch.nn.HuberLoss(delta=1.0)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(self.seed)
        for ep in range(self.epochs):
            self.net.train()
            perm = tr[torch.randperm(len(tr), generator=g)]
            for i in range(0, len(perm), self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net(Xt[b]), yt[b]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(torch.mean((self.net(Xt[ho]) - yt[ho]) ** 2))
            sched.step(v)
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_internal_mse = best
        self.epochs_run = ep + 1
        return self

    def predict(self, df):
        import torch
        self.net.eval()
        X = torch.tensor(self.xs.transform(self.fb(df)[self.cols]), dtype=torch.float32)
        with torch.no_grad():
            return self.net(X).numpy().ravel() * self.ys + self.ym


# ---------------------------------------------------------------------------------------
# Hyper-parameter search on TRAIN only (grouped CV); the val split is never used for tuning
# ---------------------------------------------------------------------------------------
def tune(estimator, param_dist: dict, fb: Callable, train: pd.DataFrame, n_iter=20, seed=C.SEED,
         n_splits=4, target=C.TARGET, n_jobs=1):
    X, y = fb(train), train[target]
    if "job_id" in train:
        cv = list(GroupKFold(n_splits).split(X, y, groups=train["job_id"]))
    else:
        cv = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    size = int(np.prod([len(v) for v in param_dist.values()])) if all(isinstance(v, list) for v in param_dist.values()) else None
    n_iter = min(n_iter, size) if size else n_iter
    rs = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, cv=cv, scoring="neg_root_mean_squared_error",
                            random_state=seed, n_jobs=n_jobs, refit=False)
    rs.fit(X, y)
    return rs.best_params_, float(-rs.best_score_)


class MeanEnsemble(DFModel):
    """Average of several already-configured DFModels (used for the 5-seed neural-network ensemble)."""

    def __init__(self, name, models, family="neural_network", complexity=8):
        self.name, self.models, self.family, self.complexity = name, models, family, complexity

    def fit(self, df, y=None):
        self.models = [copy.deepcopy(m).fit(df) for m in self.models]
        return self

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.models], axis=0)
