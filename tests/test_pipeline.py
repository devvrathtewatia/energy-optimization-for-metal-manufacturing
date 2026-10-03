"""Fast sanity tests: python -m pytest -q"""
import numpy as np
import pandas as pd

from src import config as C, data_prep as dp, features as F, physics as ph


def test_split_is_grouped_and_ood_is_unseen():
    d = dp.load_synthetic()
    assert d.groupby("job_id").split.nunique().max() == 1              # a job never spans two splits
    tr_max = d[d.split == "train"].job_mrr_cm3_min.max()
    assert d[d.split == "ood_val"].job_mrr_cm3_min.min() > tr_max      # extrapolation bands lie above training MRR
    assert d[d.split == "ood_test"].job_mrr_cm3_min.min() > d[d.split == "ood_val"].job_mrr_cm3_min.max()


def test_no_gt_or_leaky_columns_in_features():
    d = dp.load_synthetic().head(200)
    for fb in (F.commanded, F.fe_generic, F.fe_physics):
        cols = list(fb(d).columns)
        assert not [c for c in cols if c.startswith("gt_") or c in C.LEAKY_COLS + C.SIGNAL_COLS + [C.TARGET]]


def test_feature_columns_identical_across_splits():
    d = dp.load_synthetic()
    ref = list(F.fe_physics(d[d.split == "train"].head(50)).columns)
    for sp in ["val", "test", "ood_val", "ood_test"]:
        assert list(F.fe_physics(d[d.split == sp].head(50)).columns) == ref


def test_physics_hand_calculation():
    # Al6061, D=12, z=4, n=6000, fz=0.08, ap=12, ae=5: Kienzle power by hand
    vf = 0.08 * 4 * 6000
    mrr = 12 * 5 * vf
    hm = 0.08 * np.sqrt(5 / 12)
    kc = 700 * hm ** -0.25
    want_kw = kc * mrr / 6e7
    got = ph.cutting_power_kw(ph.kienzle_kc(700, 0.25, ph.mean_chip_thickness(0.08, 5, 12)), ph.mrr_mm3_min(12, 5, vf))
    assert abs(got - want_kw) < 1e-9 and 1.5 < got < 4.5


def test_energy_is_power_times_time():
    d = dp.load_synthetic()
    assert np.allclose(d.energy_wh, d.power_kw * C.WINDOW_S / 3.6, atol=1e-2)


def test_real_data_duplicates_removed_and_label_not_a_feature():
    r = dp.load_real("ibarmia")
    assert not r.drop(columns=["split", "power_kw"]).duplicated().any()
    assert "power_consumption" not in F.real_base(r).columns
