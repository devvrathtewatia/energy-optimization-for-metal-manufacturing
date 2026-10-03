"""Plant-manager dashboard for the CNC energy model. It only talks to the API (deploy/api.py).

Run (from the repo root):   streamlit run deploy/dashboard.py
By default it calls the API at http://localhost:8000. Change it in the sidebar, or set API_URL.
All data here is SIMULATED; nothing is measured on a real machine.
"""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="CNC Energy Dashboard", page_icon="⚡", layout="wide")

SAMPLE = Path(__file__).parent / "sample_windows.csv"
WINDOW_COLS = ["material", "spindle_speed_rpm", "axis_speed_mm_min", "feed_per_tooth_mm", "axial_depth_mm",
               "radial_width_mm", "tool_diameter_mm", "n_flutes", "coolant_mode", "tool_age_min",
               "machine_on_min", "ambient_temp_c"]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Connection")
api = st.sidebar.text_input("API address", os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")
try:
    h = requests.get(api + "/health", timeout=5).json()
    st.sidebar.success(f"API online - model: {h.get('model')}")
    api_ok = True
except Exception as e:  # noqa: BLE001
    st.sidebar.error("Cannot reach the API. Is it running, and is the address right?")
    api_ok = False

st.title("⚡ CNC milling - energy dashboard")
st.warning("All numbers come from a simulated machine and a fitted model. They are estimates, not measurements "
           "from a real factory.", icon="⚠️")

if not api_ok:
    st.stop()

tab1, tab2, tab3 = st.tabs(["Find energy savings for a job", "Spot wasteful machine windows", "Live machine feed"])

# ---------------------------------------------------------------- tab 1: optimizer
with tab1:
    st.subheader("How much energy could this job save?")
    st.caption("Enter how the job runs today. The tool recommends new settings that finish in the same cycle time.")
    c1, c2, c3 = st.columns(3)
    with c1:
        material = st.selectbox("Material", ["Ti6Al4V", "C45", "SS316L", "Al6061"])
        dia = st.number_input("Tool diameter (mm)", 3.0, 30.0, 12.0, 1.0)
        flutes = st.number_input("Number of flutes", 2, 8, 4, 1)
        coolant = st.selectbox("Coolant", [0, 1, 2], index=1, format_func=lambda v: {0: "Off", 1: "Flood", 2: "High pressure"}[v])
    with c2:
        rpm = st.number_input("Spindle speed (rpm)", 500.0, 8000.0, 1680.0, 10.0)
        fz = st.number_input("Feed per tooth (mm)", 0.005, 0.5, 0.063, 0.005, format="%.3f")
        ap = st.number_input("Axial depth of cut (mm)", 0.5, 40.0, 8.4, 0.1)
        ae = st.number_input("Radial width of cut (mm)", 0.2, 30.0, 6.4, 0.1)
    with c3:
        cut_time = st.number_input("Current cutting time (min)", 1.0, 120.0, 15.0, 1.0)
        tool_age = st.number_input("Tool age (min)", 0.0, 300.0, 18.0, 1.0)
        on_min = st.number_input("Machine on-time (min)", 0.0, 2000.0, 590.0, 10.0)
        amb = st.number_input("Ambient temperature (C)", 10.0, 40.0, 22.0, 0.5)

    if st.button("Recommend settings", type="primary"):
        body = dict(material=material, tool_diameter_mm=dia, n_flutes=int(flutes), coolant_mode=int(coolant),
                    tool_age_min=tool_age, machine_on_min=on_min, ambient_temp_c=amb, spindle_speed_rpm=rpm,
                    feed_per_tooth_mm=fz, axial_depth_mm=ap, radial_width_mm=ae, current_cut_time_min=cut_time)
        try:
            r = requests.post(api + "/optimize", json=body, timeout=60)
            r.raise_for_status()
            o = r.json()
        except Exception as e:  # noqa: BLE001
            st.error(f"The API returned an error: {e}")
            st.stop()
        if not o["feasible_solution_found"]:
            st.info("No better feasible setting was found, so the current settings are kept.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted saving", f"{o['predicted_saving_pct']} %")
        m2.metric("Simulator check", f"{o['simulated_saving_pct']} %")
        m3.metric("Cycle time (min)", f"{o['cycle_time_min']['recommended']}",
                  delta=f"{o['cycle_time_min']['recommended'] - o['cycle_time_min']['current']:.1f} vs today", delta_color="off")
        st.write(f"Energy for this operation: **{o['predicted_energy_kwh']['current']} kWh** today, "
                 f"**{o['predicted_energy_kwh']['recommended']} kWh** with the recommended settings (model estimate).")
        rec = o["recommended"]
        table = pd.DataFrame({
            "Setting": ["Spindle speed (rpm)", "Feed per tooth (mm)", "Axial depth (mm)", "Radial width (mm)"],
            "Today": [rpm, fz, ap, ae],
            "Recommended": [rec["spindle_speed_rpm"], rec["feed_per_tooth_mm"], rec["axial_depth_mm"], rec["radial_width_mm"]],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(o["note"] + " The recommendation respects spindle power, tool life and the same cycle window; "
                   "it has not been tried on a real machine.")

# ---------------------------------------------------------------- tab 2: anomalies
with tab2:
    st.subheader("Which windows used more power than expected?")
    st.caption("Each row is a 10-second window of machine data. A window is flagged when its measured power is "
               "clearly above what the model expects for those settings.")
    src = st.radio("Data", ["Bundled sample (2,000 simulated windows)", "Upload my own CSV"], horizontal=True)
    df = None
    if src.startswith("Bundled"):
        df = pd.read_csv(SAMPLE)
    else:
        up = st.file_uploader("CSV with the columns: " + ", ".join(WINDOW_COLS + ["power_kw"]), type="csv")
        if up is not None:
            df = pd.read_csv(up)
    if df is not None:
        missing = [c for c in WINDOW_COLS + ["power_kw"] if c not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            df = df[WINDOW_COLS + ["power_kw"]].head(5000).reset_index(drop=True)
            if st.button("Check these windows", type="primary"):
                flagged, pred, exc, z = [], [], [], []
                try:
                    for i in range(0, len(df), 1000):
                        chunk = df.iloc[i:i + 1000].to_dict("records")
                        r = requests.post(api + "/anomaly", json={"windows": chunk}, timeout=120)
                        r.raise_for_status()
                        j = r.json()
                        flagged += j["flagged"]; pred += j["predicted_power_kw"]; exc += j["excess_kw"]; z += j["z_score"]
                except Exception as e:  # noqa: BLE001
                    st.error(f"The API returned an error: {e}")
                    st.stop()
                out = df.assign(predicted_kw=pred, excess_kw=exc, z_score=z, flagged=flagged)
                kwh = out.loc[out.flagged, "excess_kw"].sum() * 10 / 3600
                a, b, c = st.columns(3)
                a.metric("Windows checked", f"{len(out):,}")
                b.metric("Flagged", f"{int(out.flagged.sum()):,}", delta=f"{100 * out.flagged.mean():.1f} % of windows", delta_color="off")
                c.metric("Wasted energy in flagged windows", f"{kwh:.2f} kWh")
                chart = alt.Chart(out).mark_circle(size=28, opacity=0.6).encode(
                    x=alt.X("predicted_kw:Q", title="Expected power (kW)"),
                    y=alt.Y("power_kw:Q", title="Measured power (kW)"),
                    color=alt.Color("flagged:N", scale=alt.Scale(domain=[False, True], range=["#4c78a8", "#e45756"]), title="Flagged"),
                    tooltip=["material", "spindle_speed_rpm", "predicted_kw", "power_kw", "excess_kw"])
                line = alt.Chart(pd.DataFrame({"x": [0, out.power_kw.max()], "y": [0, out.power_kw.max()]})).mark_line(
                    color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
                st.altair_chart(chart + line, width="stretch")
                st.caption("Points above the dashed line used more power than expected; red ones are far enough above to be flagged.")
                fl = out[out.flagged].sort_values("excess_kw", ascending=False)
                st.write("**Flagged windows (largest excess first)**")
                st.dataframe(fl[["material", "spindle_speed_rpm", "coolant_mode", "predicted_kw", "power_kw", "excess_kw", "z_score"]],
                             hide_index=True, width="stretch")
                st.download_button("Download flagged windows (CSV)", fl.to_csv(index=False), "flagged_windows.csv", "text/csv")

# ---------------------------------------------------------------- tab 3: live feed
with tab3:
    st.subheader("Live machine feed")
    st.caption("Readings are sent by a simulated machine through AWS IoT Core. The API checks each one against the "
               "model as it arrives.")
    auto = st.checkbox("Refresh automatically every 3 seconds", value=True)

    def render_live():
        try:
            j = requests.get(api + "/live", params={"limit": 150}, timeout=10).json()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the live feed: {e}")
            return
        if not j["enabled"]:
            st.info("The live feed is not switched on for this API (the IoT settings were not provided when it started).")
            return
        if j["connected"]:
            st.success("Connected to the IoT feed")
        else:
            st.warning("Not connected to the IoT feed yet - retrying. " + (j["last_error"] or ""))
        items = pd.DataFrame(j["items"])
        if items.empty:
            st.info("Waiting for readings. Start the machine simulator (device_simulator.py).")
            return
        recent = items.tail(5)
        if recent.flagged.sum() >= 3:
            st.error("ALERT: the machine has been using clearly more power than expected in its last readings.")
        a, b, c = st.columns(3)
        a.metric("Readings received", f"{j['received_total']:,}")
        b.metric("Flagged (last 150)", f"{j['flagged_in_view']}")
        last = items.iloc[-1]
        c.metric("Latest power", f"{last.measured_kw} kW", delta=f"{last.excess_kw:+.2f} kW vs expected", delta_color="inverse")
        long = items.melt(id_vars=["seq"], value_vars=["measured_kw", "predicted_kw"], var_name="series", value_name="kW")
        long["series"] = long.series.map({"measured_kw": "Measured", "predicted_kw": "Expected (model)"})
        line = alt.Chart(long).mark_line().encode(
            x=alt.X("seq:Q", title="Reading number"), y=alt.Y("kW:Q", title="Power (kW)"),
            color=alt.Color("series:N", title=None, scale=alt.Scale(range=["#4c78a8", "#9aa0a6"])))
        pts = alt.Chart(items[items.flagged]).mark_circle(size=70, color="#e45756").encode(
            x="seq:Q", y="measured_kw:Q", tooltip=["seq", "job_id", "material", "measured_kw", "predicted_kw", "excess_kw"])
        st.altair_chart(line + pts, width="stretch")
        st.caption("Red dots are readings flagged as using far more power than the model expects.")
        fl = items[items.flagged][["seq", "job_id", "material", "measured_kw", "predicted_kw", "excess_kw", "z_score"]]
        if len(fl):
            st.write("**Recently flagged readings**")
            st.dataframe(fl.iloc[::-1], hide_index=True, width="stretch")

    st.fragment(run_every=3 if auto else None)(render_live)()
