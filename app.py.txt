import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

st.set_page_config(page_title="Spatio-Temporal Gap Imputation Engine", layout="wide")

def generate_sensor_data():
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=365, freq="D")
    
    base_trend = np.sin(np.linspace(0, 3.14 * 2, 365)) * 20 + 50
    noise = np.random.normal(0, 5, 365)
    true_nox_levels = base_trend + noise
    
    df = pd.DataFrame({"Date": dates, "True_NOx": true_nox_levels})
    df["Corrupted_NOx"] = df["True_NOx"].copy()
    
    gap_indices = np.random.choice(df.index, size=int(0.25 * len(df)), replace=False)
    df.loc[gap_indices, "Corrupted_NOx"] = np.nan
    
    gap_start = 150
    gap_end = 180
    df.loc[gap_start:gap_end, "Corrupted_NOx"] = np.nan
    
    return df

st.title("Spatio-Temporal Gap Imputation Engine: WSU Prototype")
st.markdown("Engineered for the Data-Driven Solutions for Social Good and Sustainability Research Lab")

df = generate_sensor_data()

st.subheader("Phase 1: Raw Environmental Sensor Data (Sydney NOx Levels)")
st.markdown("The dataset below simulates daily Nitrogen Oxide readings over 365 days. 25 percent of the data points have been randomly corrupted to simulate sensor packet loss, alongside a deliberate 30-day continuous hardware failure event.")

fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=df["Date"], y=df["True_NOx"], mode="lines", name="True Baseline", line=dict(color="lightgrey", dash="dash")))
fig_raw.add_trace(go.Scatter(x=df["Date"], y=df["Corrupted_NOx"], mode="markers", name="Corrupted Sensor Feed", marker=dict(color="red", size=6)))
st.plotly_chart(fig_raw, use_container_width=True)

st.sidebar.header("Algorithmic Imputation Engine")
st.sidebar.markdown("Select a mathematical model to reconstruct the missing spatio-temporal data.")
algorithm = st.sidebar.selectbox("Select Imputation Algorithm", ("Linear Interpolation", "K-Nearest Neighbors (KNN)", "XGBoost Regressor"))

st.subheader("Phase 2: Algorithmic Reconstruction")

start_time = time.time()
df_imputed = df.copy()

if algorithm == "Linear Interpolation":
    df_imputed["Repaired_NOx"] = df_imputed["Corrupted_NOx"].interpolate(method="linear")

elif algorithm == "K-Nearest Neighbors (KNN)":
    temp_df = df_imputed.copy()
    temp_df['Day'] = temp_df.index
    known = temp_df[temp_df["Corrupted_NOx"].notna()]
    unknown = temp_df[temp_df["Corrupted_NOx"].isna()]
    
    if not unknown.empty:
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(known[['Day']], known["Corrupted_NOx"])
        df_imputed.loc[unknown.index, "Repaired_NOx"] = knn.predict(unknown[['Day']])
    df_imputed["Repaired_NOx"] = df_imputed["Repaired_NOx"].fillna(df_imputed["Corrupted_NOx"])

elif algorithm == "XGBoost Regressor":
    temp_df = df_imputed.copy()
    temp_df['Day'] = temp_df.index
    known = temp_df[temp_df["Corrupted_NOx"].notna()]
    unknown = temp_df[temp_df["Corrupted_NOx"].isna()]
    
    if not unknown.empty:
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        xgb.fit(known[['Day']], known["Corrupted_NOx"])
        df_imputed.loc[unknown.index, "Repaired_NOx"] = xgb.predict(unknown[['Day']])
    df_imputed["Repaired_NOx"] = df_imputed["Repaired_NOx"].fillna(df_imputed["Corrupted_NOx"])

df_imputed["Repaired_NOx"].fillna(df_imputed["Corrupted_NOx"], inplace=True)
end_time = time.time()
latency = (end_time - start_time) * 1000

mask = df["Corrupted_NOx"].isna()
rmse = np.sqrt(mean_squared_error(df.loc[mask, "True_NOx"], df_imputed.loc[mask, "Repaired_NOx"]))

fig_repaired = go.Figure()
fig_repaired.add_trace(go.Scatter(x=df_imputed["Date"], y=df_imputed["True_NOx"], mode="lines", name="True Baseline", line=dict(color="lightgrey", dash="dash")))
fig_repaired.add_trace(go.Scatter(x=df_imputed["Date"], y=df_imputed["Repaired_NOx"], mode="lines", name="Imputed Data", line=dict(color="blue")))
st.plotly_chart(fig_repaired, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("System Performance")
st.sidebar.metric(label="Computational Latency", value=f"{latency:.2f} ms")
st.sidebar.metric(label="Root Mean Square Error (RMSE)", value=f"{rmse:.4f}")

st.markdown("This live artefact demonstrates the programmatic ingestion and algorithmic repair of environmental sensor arrays with severe data gaps, ensuring zero-latency decision support for climate adaptation and precision agriculture.")