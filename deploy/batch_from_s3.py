"""Read a CSV of machine windows from S3, predict power, report the error, write predictions back to S3.

On EC2:   python3 deploy/batch_from_s3.py --bucket YOUR-BUCKET --key sample_windows.csv
Local test (no AWS):   python3 deploy/batch_from_s3.py --local deploy/sample_windows.csv
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import PhysParamModel  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--bucket")
p.add_argument("--key", default="sample_windows.csv")
p.add_argument("--local")
a = p.parse_args()

saved = json.loads((Path(__file__).parent / "model_params.json").read_text())
model = PhysParamModel()
model.theta_ = np.array([saved["params"][n] for n in PhysParamModel.NAMES])

if a.local:
    df = pd.read_csv(a.local)
else:
    import boto3
    s3 = boto3.client("s3")
    df = pd.read_csv(io.BytesIO(s3.get_object(Bucket=a.bucket, Key=a.key)["Body"].read()))

df["predicted_power_kw"] = model.predict(df)
print(f"Rows: {len(df)}")
if "power_kw" in df:
    print(f"Mean absolute difference vs recorded power_kw: {np.mean(np.abs(df.predicted_power_kw - df.power_kw)):.3f} kW")
print(f"Total predicted energy over these windows: {df.predicted_power_kw.sum() * 10 / 3600:.1f} kWh (10 s windows)")

if not a.local:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=a.bucket, Key="predictions_" + a.key, Body=buf.getvalue().encode())
    print("Wrote predictions_" + a.key + " back to S3")
