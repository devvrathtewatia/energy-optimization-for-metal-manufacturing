# data/synthetic - SIMULATED data (not industrial measurements)

`synthetic_machining_windows.csv` is produced by `python -m src.synth_generator` (seed 42, 2 500 jobs, 27 027 windows of 10 s).
It is generated from first-principles equations (Kienzle cutting force, spindle no-load losses, copper losses, inverter
efficiency, coolant/auxiliary loads, feed-axis friction - see `src/physics.py`) plus hidden factors (workpiece lot scatter,
tool-specific life, cold-start friction, injected inefficiencies), sensor noise, quantisation and glitches.

* Columns prefixed `gt_` are ground truth for evaluation only (never features).
* `power_kw` is the (noisy) metered electrical power; `energy_wh = power_kw * 10 s / 3.6`.
* `voltage_v`, `current_a`, `power_factor` are consistent with the power by construction (leakage; excluded from predictors).
* The generator's constants are in `src/config.py` and `synthetic_metadata.json` (with the sha256 of the frozen file).

Everything learned from this file describes method behaviour on a controlled problem, not a real factory.
