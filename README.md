# AI-Driven Energy Optimization in Metal Manufacturing

An end-to-end CNC milling energy analytics system that combines **physics-guided modeling, machine-learning benchmarks, anomaly detection, constrained optimization, REST APIs, a Streamlit dashboard, and live IoT telemetry through AWS IoT Core**.

> **Important:** The core machine data and all energy-savings results are **simulated or model-predicted**, not measurements from a production factory. The public real dataset is used mainly for descriptive/reliability analysis because it does not contain the variables needed for the project's optimization problem.

---

## What this project does

The system answers four practical manufacturing questions:

1. **How much power should the CNC machine use?**
2. **Which operating windows are using substantially more power than expected?**
3. **What feasible operating point can reduce energy while respecting production and machine constraints?**
4. **Can the same analytics run continuously on incoming machine telemetry?**

The final deployment connects these pieces into one workflow:

```text
Simulated CNC machine
        │
        │ MQTT / TLS
        ▼
AWS IoT Core
        │
        │ plant/cnc1/telemetry
        ▼
EC2 IoT listener
        │
        ▼
FastAPI inference service
   ┌────┼───────────┐
   │    │           │
   ▼    ▼           ▼
Predict Anomaly   Optimize
   │    │           │
   └────┴──────┬────┘
                ▼
        Streamlit dashboard
```

---

## Headline results

### Energy prediction

On **440 unseen synthetic jobs**, the selected fitted physics equation achieved:

- **MAE:** 0.365 kW
- **RMSE:** 1.010 kW
- **R²:** 0.931
- **MAPE:** 3.3%

The selected model is a **21-coefficient fitted physics equation**, not a black-box neural network. The simulator is built from physical functional forms, so the experiment primarily tests whether that structure can be recovered from noisy data rather than claiming that physics models universally outperform ML on real machines.

### Anomaly detection

Residual-based detection uses:

```text
residual = measured power - predicted power
```

with robust, heteroscedastic scaling and thresholds of **z > 3.5** and **excess > 0.4 kW**.

Out-of-fold window-level performance:

- **Precision:** 0.82
- **Recall:** 0.91
- **Average precision:** 0.72
- Captures about **90% of the true excess energy** in flagged windows

The detector is intended to identify **potential inefficiency**, not diagnose a specific hardware failure.

### Energy optimization

A constrained optimizer minimizes energy over a **fixed takt window**, including standby energy, subject to:

- production-time constraint
- spindle power/torque margin
- tool-life constraint
- cutting-speed limits
- feed/depth/radial-width bounds
- model trust-region limits on material-removal rate

Across **440 unseen jobs**:

- **37.2% simulated mean energy saving** over the fixed takt window
- **95% CI: 35.8–38.6%**
- **39.6% model-predicted saving**

Operation-only savings are higher, but are treated as an upper bound because finishing early does not automatically eliminate standby energy.

> These savings are **predicted/simulated only**. They are not measured industrial savings.

---

## Why the physics-guided approach

The machine model represents electrical power through operating state, spindle behavior, coolant, feed-axis loads, cutting physics, and material-removal rate.

At the machining level:

```text
MRR = a_p · a_e · v_f
v_f = f_z · z · n
h_m = f_z · sqrt(a_e / D)
k_c = k_c1.1 · h_m^(-m_c)
P_cut = k_c · MRR / 6·10^7
```

The broader electrical model includes base loads, ambient effects, coolant, spindle losses, copper losses, servo/feed-axis loads, and cutting power.

A simpler machine-level specific-energy form, `P = P₀ + k·MRR`, explains a large fraction of the variance by itself; the full model adds speed- and state-dependent structure.

---

## Synthetic machine and data

The simulator represents a **3-axis vertical machining centre with an 18.5 kW spindle** and covers:

- Al6061
- C45 steel
- 316L stainless steel
- Ti-6Al-4V

The synthetic dataset contains **27,027 ten-second windows** and includes:

- measurement noise
- sensor glitches/spikes/dropouts
- workpiece-lot variation
- tool-life effects
- cold-start friction
- injected inefficiencies such as coolant issues, tool wear, bearing friction, auxiliary leaks, and axis drag

### Public real data

The project also includes the CFAA milling tests dataset from Zenodo:

> Tapia Fernandez E., Sastoque Pinilla L., Lopez-Novoa U., *Milling tests in 2 machining centres: Energy consumption data*, Zenodo, 2024. DOI: [10.5281/zenodo.14445879](https://doi.org/10.5281/zenodo.14445879), CC BY 4.0.

The public dataset is useful for descriptive analysis, but it lacks several variables required for the project's main optimization problem and produced unreliable predictive generalization under grouped validation. It is therefore **not used to claim industrial predictive performance**.

---

## Model benchmark

The study compares:

- linear and polynomial regression
- decision trees
- random forests
- XGBoost / LightGBM
- K-Means regime models
- MLP ensembles
- physics-feature models
- a hybrid physics + residual-ML model
- the fitted physics equation

A pre-registered protocol freezes the data split, metrics, selection rule, and experimental expectations before model selection.

One important result is that **feature engineering and physical structure mattered more than aggressive hyperparameter tuning** in several experiments.

The final model was selected using validation performance plus an extrapolation/OOD validation band rather than the untouched test set.

---

## Live IoT deployment

The final deployment turns the offline project into a live monitoring demo.

### Architecture

```text
stream_demo.csv
      │
      ▼
device_simulator.py
      │
      │ MQTT over TLS / X.509
      ▼
AWS IoT Core
      │
      │ plant/cnc1/telemetry
      ▼
iot_listener.py
      │
      ▼
FastAPI /live
      │
      ▼
Streamlit live dashboard
```

### AWS / deployment components

- **AWS IoT Core** — MQTT ingestion
- **EC2** — listener + FastAPI inference service
- **S3** — project data/object storage used by the deployment flow
- **Streamlit** — interactive dashboard
- **Paho MQTT** — MQTT client implementation
- **X.509 certificates** — device/client authentication

### Live fault demonstration

The simulator supports fault injection. For example:

```bash
python deploy/device_simulator.py \
  --endpoint "<iot-endpoint>" \
  --cert "<device-certificate>" \
  --key "<private-key>" \
  --ca "<amazon-root-ca>" \
  --interval 1 \
  --fault-after 40 \
  --fault-kw 2.5
```

This creates a simple live demonstration:

```text
Normal telemetry
      ↓
Fault injected after reading 40
      ↓
Measured power rises above model expectation
      ↓
Residual increases
      ↓
Anomaly detector flags windows
      ↓
Dashboard raises a live alert
```

The live API exposes connection and ingestion state through `/live`, including whether the IoT feed is enabled, whether MQTT is connected, the number of readings received, the latest readings, and flagged windows.

---

## Dashboard

The Streamlit dashboard has three main workflows:

### 1. Find energy savings for a job

Enter machining conditions such as material, spindle speed, feed per tooth, axial/radial depth, tool state, coolant, and machine context.

The API returns:

- current predicted energy
- recommended operating point
- predicted/simulated energy after optimization
- estimated saving percentage
- cycle-time information
- feasibility status

### 2. Spot wasteful machine windows

Upload a CSV or use the bundled synthetic sample to compare measured and expected power.

The dashboard shows:

- windows checked
- flagged windows
- estimated excess energy
- predicted-vs-measured power
- flagged windows sorted by excess
- z-scores and residuals

### 3. Live machine feed

The live tab consumes telemetry through the deployed API and displays:

- connection state
- readings received
- flagged readings
- latest power
- expected vs measured power over time
- live anomaly alerts

---

## API

The final FastAPI deployment exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service/model health check |
| `/predict` | POST | Power prediction for input windows |
| `/anomaly` | POST | Residual/z-score based anomaly detection |
| `/optimize` | POST | Constrained energy optimization for a job |
| `/live` | GET | Live MQTT ingestion and anomaly state |

Example health check:

```bash
curl http://<ec2-public-ip>:8000/health
```

Example live-feed check:

```bash
curl http://<ec2-public-ip>:8000/live
```

---

## Reproducibility

### Core research pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
python -m pytest -q
```

All experiment seeds are fixed. The pipeline regenerates the synthetic data, tables, figures, notebooks, and experiment outputs.

### Deployment dependencies

API:

```bash
pip install -r deploy/requirements-api.txt
```

Dashboard:

```bash
pip install -r deploy/requirements-dashboard.txt
```

Run the API:

```bash
uvicorn deploy.api:app --host 0.0.0.0 --port 8000
```

Run the dashboard:

```bash
API_URL=http://<ec2-public-ip>:8000 streamlit run deploy/dashboard.py
```

The IoT listener is started automatically by the FastAPI application when `IOT_ENDPOINT` (or a local broker endpoint) is configured; `deploy/iot_listener.py` is **not** intended to be launched as a standalone application.

Required IoT environment variables:

```text
IOT_ENDPOINT
IOT_CERT
IOT_KEY
IOT_CA
```

---

## Repository layout

```text
data/raw/            Public real dataset + provenance

data/synthetic/      Synthetic simulator output + metadata

data/processed/      Cleaned/processed datasets

src/
  physics.py          Machine physics and power model
  synth_generator.py  Synthetic machine/data generator
  features.py         Feature engineering
  models.py           Model definitions/training
  optimize.py         Constrained optimizer
  metrics.py          Evaluation utilities

experiments/
  PROTOCOL.md         Frozen experimental protocol
  LOG.md              Experiment narrative
  e00...e14 scripts   Reproducible study stages
  results/            Experiment JSON summaries

results/
  figures/            Final plots
  tables/             Leaderboards, anomaly and optimization tables
  models/             Final fitted model

notebooks/             Executed analysis notebooks
tests/                 Sanity and consistency tests

deploy/
  api.py              FastAPI inference + live-feed service
  dashboard.py        Streamlit dashboard
  device_simulator.py MQTT telemetry simulator
  iot_listener.py     AWS IoT Core / MQTT listener
  batch_from_s3.py    S3-backed batch flow
  build_reference.py  Anomaly reference builder
  call_api.py         API client helpers
  stream_demo.csv     Live telemetry demo stream
  sample_windows.csv  Bundled API/dashboard sample
```

---

## Limitations and honest deployment boundary

This project is intentionally explicit about what it does **not** prove.

- The headline energy savings are simulated/model-predicted, not measured factory savings.
- The winning physics equation is learned from a simulator that shares functional structure with the model family.
- The public real dataset does not contain enough process variables for the full optimization problem.
- The optimizer does not model every manufacturing constraint, including chatter stability, surface finish, fixture rigidity, tool-change logistics, and scheduling.
- The anomaly detector identifies unusual energy behavior; it does not diagnose the physical root cause.
- The live IoT demonstration uses simulated machine telemetry rather than physical sensors.
- Real deployment would require plant-specific validation, sensor calibration, machine limits, and operator/production constraints.

### Recommended real-plant validation

Before using the optimizer or anomaly detector on a production machine:

1. collect synchronized power, spindle, feed-axis, depth/width, coolant, tool-state, and machine-state measurements;
2. validate the predictor on held-out jobs and operating regimes;
3. calibrate anomaly thresholds against real maintenance/quality events;
4. verify optimizer recommendations against machine safety, tool life, quality, and production constraints;
5. run a controlled pilot before using recommendations in closed-loop operation.

---

## Why the project is useful as a digital-manufacturing case study

This project deliberately goes beyond a single ML model:

```text
Physical process understanding
        +
Data engineering
        +
Model benchmarking
        +
Explainability
        +
Anomaly detection
        +
Constrained optimization
        +
REST API deployment
        +
Cloud / IoT integration
        +
Business-facing dashboard
```

The result is a prototype **manufacturing digitalization workflow** rather than an isolated prediction notebook.

---

## Attribution

Real data:

Tapia Fernandez E., Sastoque Pinilla L., Lopez-Novoa U., *Milling tests in 2 machining centres: Energy consumption data*, Zenodo, 2024, [doi:10.5281/zenodo.14445879](https://doi.org/10.5281/zenodo.14445879), CC BY 4.0.

No code license is included by default. Add an explicit software license before redistributing the repository.
