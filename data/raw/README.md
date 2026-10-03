# data/raw - REAL public data

| File | Machine | Rows | md5 |
|---|---|---|---|
| `IBARMIA_dataset.csv` | Ibarmia THR-16 machining centre | 6 000 | `47c4c409c9b8670d411e6a4e2291376e` |
| `GMTK_dataset.csv` | GMTK VR 2.4 machining centre | 6 000 | `f9bd66fb064a8c91b4a3131681ee0ef8` |

* **Source:** Tapia Fernandez, Sastoque Pinilla, Lopez-Novoa - *Milling tests in 2 machining centres: Energy consumption data*,
  Zenodo, doi:10.5281/zenodo.14445879 (2024-12-13). Measured July-November 2023 at the Advanced Manufacturing Centre for
  Aeronautics (CFAA), Spain. **License CC BY 4.0** - redistributed here unchanged with attribution.
* **Columns:** `load_X`, `load_Z` (axis loads), `power_Z` (Z-axis drive power), `speed_SPINDLE`, `override_SPINDLE`,
  `powerDrive_SPINDLE` (spindle drive power, kW - used as the target), `power_consumption` (Low/Medium/High label; definition
  not documented; **not used as a feature**).
* **What is NOT in the files:** feed, depth/width of cut, spindle torque, material, timestamps, run ids.
* **Properties found during the audit (see `experiments/LOG.md`, E01/E14):** classes are balanced by construction
  (2 000 rows each), ~13 % exact duplicate rows, rows are shuffled, and on the GMTK machine the spindle-speed column stays
  within +-120 (spindle essentially not rotating or different units).

Nothing in `data/synthetic/` is real; see that folder's README.
