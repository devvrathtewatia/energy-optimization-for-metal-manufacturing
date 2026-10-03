"""Pretends to be the machine: publishes one reading at a time to AWS IoT Core (or a local broker for testing).

AWS IoT Core:
  python deploy/device_simulator.py --endpoint YOUR-ENDPOINT.iot.ap-south-1.amazonaws.com \
      --cert deploy/certs/device.pem.crt --key deploy/certs/private.pem.key --ca deploy/certs/AmazonRootCA1.pem
Local test with no AWS:
  python deploy/device_simulator.py --local-host localhost --interval 0.2
The readings are SIMULATED (replayed from deploy/stream_demo.csv, unseen test jobs of the synthetic dataset).
Use --fault-after 40 to make the machine start wasting power after 40 readings (to see the alert fire).
"""
import argparse
import json
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

TOPIC = "plant/cnc1/telemetry"
p = argparse.ArgumentParser()
p.add_argument("--endpoint")
p.add_argument("--cert")
p.add_argument("--key")
p.add_argument("--ca")
p.add_argument("--local-host")
p.add_argument("--csv", default=str(Path(__file__).parent / "stream_demo.csv"))
p.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
p.add_argument("--count", type=int, default=0, help="stop after this many readings (0 = whole file)")
p.add_argument("--fault-after", type=int, default=0, help="after this many readings, add --fault-kw to the power")
p.add_argument("--fault-kw", type=float, default=2.5)
a = p.parse_args()

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="cnc1-simulator")
if a.local_host:
    host, port = a.local_host, 1883
else:
    if not (a.endpoint and a.cert and a.key):
        raise SystemExit("Need --endpoint, --cert and --key (or use --local-host for a local test).")
    c.tls_set(ca_certs=a.ca or None, certfile=a.cert, keyfile=a.key, tls_version=ssl.PROTOCOL_TLS_CLIENT)
    host, port = a.endpoint, 8883
c.connect(host, port, keepalive=30)
c.loop_start()

df = pd.read_csv(a.csv)
if a.count:
    df = df.head(a.count)
print(f"Publishing {len(df)} simulated readings to {host} on topic {TOPIC} ...")
for i, row in enumerate(df.to_dict("records"), start=1):
    if a.fault_after and i > a.fault_after:
        row["power_kw"] = round(row["power_kw"] + a.fault_kw, 3)
    row.update(seq=i, ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    info = c.publish(TOPIC, json.dumps(row), qos=1)
    info.wait_for_publish(timeout=10)
    print(f"  sent #{i}  job {row['job_id']}  {row['material']}  {row['power_kw']} kW")
    time.sleep(a.interval)
c.loop_stop()
c.disconnect()
print("Done.")
