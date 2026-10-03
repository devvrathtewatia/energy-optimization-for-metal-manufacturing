"""Send some windows to the running API and compare with the recorded power.
Usage:  python3 deploy/call_api.py http://localhost:8000
        python3 deploy/call_api.py http://YOUR-EC2-PUBLIC-IP:8000
"""
import sys
import pandas as pd
import requests

url = sys.argv[1].rstrip("/")
df = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "deploy/sample_windows.csv").head(200)
cols = [c for c in df.columns if c != "power_kw"]
r = requests.post(url + "/predict", json={"windows": df[cols].to_dict("records")}, timeout=30)
r.raise_for_status()
pred = r.json()["predicted_power_kw"]
print("API health:", requests.get(url + "/health", timeout=10).json())
print("Windows sent:", len(pred))
print("First 3 predictions (kW):", pred[:3], " recorded:", df.power_kw.head(3).round(3).tolist())
print("Mean absolute difference:", round(sum(abs(p - t) for p, t in zip(pred, df.power_kw)) / len(pred), 3), "kW")
