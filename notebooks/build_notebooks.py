"""Builds and executes the three notebooks against the saved results (no heavy training inside notebooks).
python notebooks/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())

SETUP = """
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 60)
FIG, TAB, RES = ROOT/"results/figures", ROOT/"results/tables", ROOT/"experiments/results"
def show(name, width=760): display(Image(filename=str(FIG/name), width=width))
"""


def nb01():
    c = [
        md("""# 01 - Data and EDA
**Two datasets, never mixed.**

| | REAL public data | SYNTHETIC (simulated) data |
|---|---|---|
| source | CFAA milling tests, Zenodo 10.5281/zenodo.14445879 (CC BY 4.0), Ibarmia + GMTK machining centres | `src/synth_generator.py`, seed 42 |
| what it is | measured spindle power with speed, override, axis loads | first-principles power model + hidden factors + noise + injected inefficiencies |
| limits | no feed/depth/torque/timestamps/run ids, balanced subsample, 13 % duplicates | correct by construction only inside the simulator's assumptions |
| role | sanity check and honest limits | main experimental engine, anomaly labels, optimisation |
"""),
        code(SETUP),
        code("""
raw = {k: pd.read_csv(ROOT/'data/raw'/f) for k, f in [('ibarmia','IBARMIA_dataset.csv'), ('gmtk','GMTK_dataset.csv')]}
display(Markdown('### REAL - Ibarmia head')); display(raw['ibarmia'].head())
display(Markdown('### REAL - summary')); display(pd.concat({k: v.describe().T[['mean','std','min','max']] for k, v in raw.items()}).round(3))
syn = pd.read_csv(ROOT/'data/synthetic/synthetic_machining_windows.csv')
display(Markdown('### SYNTHETIC - head (columns starting with gt_ are ground truth, never features)')); display(syn.head())
print(json.dumps(json.load(open(ROOT/'data/processed/synthetic_prep_report.json')), indent=1))
"""),
        md("## Where does the power go? (synthetic)"),
        code("show('e01_power_by_state.png', 620); show('e01_power_vs_mrr.png', 700); show('e01_aircut_power_vs_speed.png', 700)"),
        code("""
e = json.load(open(RES/'E01_eda.json'))
s = e['synthetic_summary']
print('mean power by state [kW]:', s['mean_power_by_state'])
print('inside a cutting window [kW]:', s['cutting_power_decomposition_kw'])
print('oracle floor (true noise-free power vs metered), validation:', {k: round(v, 3) for k, v in s['oracle_floor_val'].items()})
"""),
        md("## Leakage audit\nElectrical measurements are the target in disguise (P = sqrt(3)·V·I·cos φ). Planning models use **commanded** conditions only."),
        code("pd.DataFrame(e['leakage_audit_synthetic_val']).T[['mae','rmse','r2','mape']].round(3)"),
        md("## Splits: extrapolation bands in material-removal rate"),
        code("show('e01_split_mrr.png', 700)"),
        md("## Real data: what is in it, and what is not"),
        code("show('e01_real_power_vs_speed.png', 900)"),
        code("""
r = e['real']
for k, v in r.items():
    print(k, '| duplicates:', v['duplicates'], '| share rows |speed|>100rpm:', round(v['frac_rows_spindle_abs_speed_gt_100rpm'], 3),
          '| R2 of Low/Med/High class means vs spindle power:', round(v['r2_of_class_means'], 3))
    print('   correlation with spindle power:', v['corr_with_power'])
"""),
        md("## Can the real data support predictive claims? (E14)\nTree models reach R² ≈ 0.75 with a random split, but the score collapses when neighbouring records cannot leak between train and test."),
        code("show('e14_real_feature_subsets.png', 900)"),
        code("""
p = json.load(open(RES/'E14_real_data_probes.json'))
pd.DataFrame({k: {'random 5-fold R2': v['cv_random_5fold']['r2'], 'cluster-grouped 5-fold R2': v['cv_cluster_grouped_5fold']['r2'],
                  '1-NN regressor val R2': v['nn']['one_nn_regressor_val']['r2'],
                  'val rows with train neighbour < 0.05 std': v['nn']['share_val_with_nn_dist_lt_0p05']} for k, v in p.items()}).round(3)
"""),
    ]
    return c


def nb02():
    c = [
        md("""# 02 - The experimental progression
Physics baseline → EDA → linear/polynomial → K-Means regimes → trees → random forest → XGBoost/LightGBM → feature engineering/tuning → neural network → explainability → physics-structured/hybrid → selection → anomalies → optimisation.
Narrative (why / expected / happened / learned / next) for each step: `experiments/LOG.md`. Expectations were frozen in `experiments/PROTOCOL.md` before the runs.

Columns: `v*` = validation (all windows), `ood_MAE` = extrapolation band, engaged windows."""),
        code(SETUP),
        code("""
df = pd.read_csv(TAB/'all_experiments_long.csv')
syn = df[df.dataset == 'synthetic'].copy()
def table(prefixes):
    t = syn[syn.exp_id.str.startswith(tuple(prefixes))][['exp_id','model','train.all.mae','val.all.mae','val.all.rmse','val.all.r2','val.all.mape','ood_val.engaged.mae']]
    t.columns = ['exp','model','train_MAE','val_MAE','val_RMSE','val_R2','val_MAPE','ood_MAE']
    return t.round(3).reset_index(drop=True)
"""),
        md("## E00 physics baselines, E02 linear/polynomial"), code("table(['E00','E02'])"),
        md("## E03 K-Means regimes\nSilhouette does not pick the physical k=4; clusters are speed bands, not machine states (standby and rapid are recovered perfectly)."),
        code("show('e03_kmeans_selection_synthetic.png', 900); show('e03_confusion_k4_synthetic.png', 520)"),
        code("table(['E03'])"),
        md("## E04-E06 trees, forest, boosting"), code("table(['E04','E05','E06'])"),
        md("## E07 feature engineering (bigger gain than tuning)"), code("table(['E07'])"),
        md("## E08 neural network (5 seeds; std over seeds shown for the single networks)"),
        code("""
t = table(['E08'])
t['val_MAE_seed_std'] = syn[syn.exp_id.str.startswith('E08')]['val.all.mae_std_over_seeds'].round(3).to_numpy()
t
"""),
        md("## E09 explainability"), code("show('e09_shap_beeswarm_synthetic.png', 720); show('e09_shap_vs_truth_groups.png', 720)"),
        code("pd.read_csv(TAB/'e09_shap_groups_vs_truth.csv', index_col=0)"),
        md("## E10 physics-structured and hybrid models"), code("table(['E10'])"),
        code("pd.read_csv(TAB/'e10_param_recovery.csv').round(4)"),
        md("""## E11 selection by the frozen rule
`score = 0.5·MAE_val/min + 0.5·MAE_oodval_engaged/min`, ties within 2 % go to the simpler model."""),
        code("show('e11_model_landscape.png', 760)"),
        code("""
lb = pd.read_csv(TAB/'leaderboard_synthetic.csv')
lb[['model','family','complexity','val_mae','oodval_eng_mae','score']].head(12).round(3)
"""),
        code("""
sel = json.load(open(RES/'E11_selection.json'))
print('WINNER:', sel['winner'], '| within 2% of best:', sel['candidates_within_2pct'])
pd.DataFrame({sp: {sub: m for sub, m in d.items()} for sp, d in sel['final_metrics_trainval_refit'].items()}).map(lambda m: {k: round(v, 3) for k, v in m.items()})
"""),
        code("show('e11_family_best.png', 900)"),
        md("## Real-data track (supporting; see E14 for why these scores are not generalisation evidence)"),
        code("""
for k in ['ibarmia', 'gmtk']:
    print('==', k); display(pd.read_csv(TAB/f'leaderboard_real_{k}.csv').round(3).head(8))
"""),
    ]
    return c


def nb03():
    c = [
        md("""# 03 - Final model, anomaly detection, optimisation
**Everything below the "Optimisation" heading is PREDICTED or SIMULATED, never measured.**"""),
        code(SETUP),
        md("## The final model: a fitted physics equation (selected by the frozen rule, not chosen in advance)"),
        code("""
import joblib
from src import config as C, data_prep as dp
m = joblib.load(C.MODELS/'final_model.joblib')
print(m.name)
d = dp.load_synthetic()
from src.metrics import regression_metrics
rows = {}
for sp in ['test', 'ood_test']:
    s = d[d.split == sp]
    p = m.predict(s)
    rows[sp + ' (all windows)'] = regression_metrics(s.power_kw, p)
    rows[sp + ' (cutting windows)'] = regression_metrics(s.power_kw[s.engaged], p[s.engaged.to_numpy()])
pd.DataFrame(rows).T.round(3)
"""),
        code("pd.DataFrame([m.params()]).T.rename(columns={0: 'fitted value'}).round(5)"),
        md("P_el = p0 + a_amb(T-20) + coolant(mode) + conveyor·[cutting] + [P_cut + P_idle(n)·warm(t) + r_cu T²/1000] + b·v_axis,   P_cut = k_c1.1 · h_m^(-m_c) · MRR / 6e7,  h_m = f_z √(a_e/D), T = 9550 P_cut / n"),
        md("Fitted coefficients (final model, train+val refit) against the simulator's effective truth. Coefficients that trade off against each other (k_c1.1 vs m_c) are the least well identified. The train-only fit of E10 is in `e10_param_recovery.csv`."),
        code("pd.read_csv(TAB/'final_model_param_recovery.csv').round(4)"),
        md("## Explainability"), code("show('e09_shap_bar_synthetic.png', 560); show('e09_shap_dependence_axial_depth_mm.png', 520)"),
        md("""## Energy-residual anomaly detection (E12)
residual = actual − predicted (out-of-fold). Flagged windows are **potential inefficient operating conditions** - not equipment failures."""),
        code("show('e12_residuals_pr_synthetic.png', 900); show('e12_recall_by_type.png', 620)"),
        code("""
a = json.load(open(RES/'E12_anomaly.json'))
pd.DataFrame({k: {m: v for m, v in a[k].items() if not isinstance(v, dict)} for k in ['in_distribution_oof', 'extrapolation_ood_jobs']}).round(3)
"""),
        code("""
f = pd.read_csv(TAB/'e12_flagged_windows_synthetic.csv')
f[['job_id','material','spindle_speed_rpm','axial_depth_mm','coolant_mode','power_kw','pred','resid','z','gt_job_anomaly']].sort_values('z', ascending=False).head(10).round(2)
"""),
        md("Real data: rows with unusually high spindle power given their recorded inputs (no ground truth, candidates for inspection only)."),
        code("show('e12_real_flagged.png', 900)"),
        md("""## Optimisation (E13) - PREDICTED / SIMULATED savings
Minimise energy over a fixed takt window (standby included) subject to: same cycle time, spindle power/torque, tool life, cutting-speed window, training-data ranges, MRR within the model's trust region.
Current conditions = recorded settings of 440 unseen jobs; recommended = optimiser output; savings verified in the noise-free simulator."""),
        code("""
o = json.load(open(RES/'E13_optimization.json'))
pd.DataFrame({k: {m: v for m, v in o[k].items() if m != 'label'} for k in ['A_final_model_window_objective', 'A_final_model_operation_objective']}).T.round(3)
"""),
        code("show('e13_before_after.png', 620); show('e13_parameter_shift.png', 980); show('e13_saving_by_material.png', 560)"),
        md("### Does the recommendation survive the simulator? (which model the optimiser uses matters)"),
        code("show('e13_model_comparison.png', 980)"),
        code("pd.read_csv(TAB/'e13_model_comparison.csv')[['model','regime','mean_saving_pred_pct','mean_saving_true_pct','true_overload_opt_share']].round(2)"),
        code("""
jobs = pd.read_csv(TAB/'e13_job_results_final_model.csv')
jobs[['job_id','material','n_base','n_opt','fz_base','fz_opt','ap_base','ap_opt','ae_base','ae_opt','t_op_base_min','t_op_opt_min','e_true_base','e_true_opt','save_true_pct','save_pred_pct']].head(10).round(3)
"""),
        md("""### Limits of the optimisation claim
* The baseline is a *sampled* practice distribution, often far below the simulated machine's capability; real plants are already bounded by chatter, surface finish, fixture rigidity and tool-change rules that are not modelled.
* The absolute percentage is therefore an upper bound inside the simulator. Robust take-aways: fixed loads dominate (so finishing sooner matters), spindle speed should not be higher than needed, and tree models must not be trusted at the edge of their data."""),
    ]
    return c


def build():
    for name, cells in [("01_data_and_eda", nb01()), ("02_experiment_progression", nb02()), ("03_final_model_anomaly_optimization", nb03())]:
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbf.write(nb, HERE / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    build()
