import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sim_engine import MultiPathSimulation

st.set_page_config(page_title="Multi-Path ML Adaptive Routing", layout="wide")

st.title("Multi-Path Network Traffic Simulation")
st.caption("3-Path Adaptive ML Re-evaluation per Interval (Offline Reinforcement Engine)")

with st.sidebar:
    st.header("Simulation Settings")
    intervals = st.slider("Simulation Steps", min_value=50, max_value=300, value=100, step=10)
    traffic_load = st.slider("Mean Arrival Packets", min_value=20, max_value=200, value=70, step=10)
    burstiness = st.slider("Burstiness Amplitude", min_value=0.0, max_value=1.5, value=0.5, step=0.1)
    routing_mode = st.radio("Strategy", ["Adaptive ML Routing", "Static Round-Robin"])
    run_btn = st.button("Run Simulation", type="primary")

sim = MultiPathSimulation(interval_sec=0.1)
is_ml = (routing_mode == "Adaptive ML Routing")

df = sim.run(
    total_intervals=intervals,
    traffic_load_pkts=traffic_load,
    burstiness=burstiness,
    enable_ml=is_ml
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Latency", f"{df['effective_latency'].mean():.2f} ms")
c2.metric("P95 Latency", f"{df['effective_latency'].quantile(0.95):.2f} ms")
c3.metric("Packet Drop Rate", f"{(df['effective_loss_rate'].mean() * 100):.2f}%")
c4.metric("Strategy", "Adaptive ML" if is_ml else "Round-Robin")

st.markdown("---")

fig_lat = go.Figure()
fig_lat.add_trace(go.Scatter(x=df["step"], y=df["effective_latency"], mode="lines", name="Observed Latency", line=dict(color="#00FFAA", width=2)))
for p_id, colr in [(0, "rgba(255, 99, 132, 0.5)"), (1, "rgba(54, 162, 235, 0.5)"), (2, "rgba(255, 206, 86, 0.5)")]:
    fig_lat.add_trace(go.Scatter(x=df["step"], y=df[f"p{p_id}_latency"], mode="lines", name=f"Path {p_id}", line=dict(color=colr, dash="dot")))
fig_lat.update_layout(title="End-to-End Latency vs Individual Paths", xaxis_title="Interval", yaxis_title="Latency (ms)")
st.plotly_chart(fig_lat, use_container_width=True)

fig_scatter = px.scatter(df, x="step", y="selected_path", color="effective_latency", size="arrival_pkts", title="Dynamic Path Re-evaluation Schedule")
st.plotly_chart(fig_scatter, use_container_width=True)