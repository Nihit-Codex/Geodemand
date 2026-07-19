"""
Mumbai Ride-Hailing Demand Forecasting & Fleet Optimization Dashboard
-----------------------------------------------------------------------
Simulates how platforms like Uber/Rapido use historical GPS ping data
to spot demand hotspots and decide where to pre-position idle drivers.
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Mumbai Ride Demand Dashboard", layout="wide")

DATA_FILE = "ride_data.csv"
NUM_ROWS = 2000

# Bounding box: Kandivali / Mumbai suburbs
LAT_MIN, LAT_MAX = 19.15, 19.25
LON_MIN, LON_MAX = 72.80, 72.90

STATUSES = ["Ride Requested", "Driver Assigned", "No Driver Available"]
STATUS_WEIGHTS = [0.50, 0.35, 0.15]


# ============================================================
# 1. SIMULATED DATA GENERATION
# ============================================================
@st.cache_data
def generate_synthetic_data(n_rows: int = NUM_ROWS, seed: int = 42) -> pd.DataFrame:
    """
    Builds a synthetic GPS ping log for one 24-hour cycle.

    Instead of pure uniform noise, points are pulled toward a handful of
    'cluster centers' (stations, malls, business districts) so the map
    shows realistic-looking hotspots rather than a flat random scatter.
    Everything here is vectorized with numpy -- no per-row loops.
    """
    rng = np.random.default_rng(seed)

    # --- Timestamps spread across a full day ---
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    random_seconds = rng.integers(0, 24 * 60 * 60, size=n_rows)
    timestamps = [base_date + timedelta(seconds=int(s)) for s in random_seconds]

    # --- Clustered coordinates inside the bounding box ---
    n_clusters = 5
    cluster_lat = rng.uniform(LAT_MIN, LAT_MAX, n_clusters)
    cluster_lon = rng.uniform(LON_MIN, LON_MAX, n_clusters)
    cluster_id = rng.integers(0, n_clusters, size=n_rows)

    lat_noise = rng.normal(0, 0.01, size=n_rows)
    lon_noise = rng.normal(0, 0.01, size=n_rows)

    latitude = np.clip(cluster_lat[cluster_id] + lat_noise, LAT_MIN, LAT_MAX)
    longitude = np.clip(cluster_lon[cluster_id] + lon_noise, LON_MIN, LON_MAX)

    status = rng.choice(STATUSES, size=n_rows, p=STATUS_WEIGHTS)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "latitude": latitude,
            "longitude": longitude,
            "status": status,
        }
    )

    # Inject a small amount of messy/null data so the cleaning step below
    # is actually doing something meaningful (mirrors real-world GPS logs).
    null_idx = rng.choice(n_rows, size=int(n_rows * 0.015), replace=False)
    df.loc[null_idx, "latitude"] = np.nan

    return df


# ============================================================
# 2. DATA PROCESSING PIPELINE
# ============================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads ride_data.csv if it exists locally, otherwise falls back to the
    synthetic generator. Then runs it through a cleaning pipeline.
    """
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = generate_synthetic_data()

    # Convert raw string timestamps -> real datetime objects.
    # errors="coerce" turns any unparseable value into NaT instead of crashing.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Drop rows with nulls in any critical column (vectorized, no loop).
    df = df.dropna(subset=["timestamp", "latitude", "longitude", "status"])

    # Drop GPS anomalies: any point that falls outside our known city bounding box.
    in_bounds = df["latitude"].between(LAT_MIN, LAT_MAX) & df["longitude"].between(
        LON_MIN, LON_MAX
    )
    df = df[in_bounds]

    # Extract hour-of-day for time-based filtering/aggregation.
    df["hour"] = df["timestamp"].dt.hour

    return df.reset_index(drop=True)


# ============================================================
# 3. STREAMLIT INTERFACE
# ============================================================
st.title("🚕 Mumbai Ride-Hailing Demand Dashboard")
st.markdown(
    """
    This dashboard simulates how ride-hailing platforms like **Uber** or **Rapido**
    mine historical GPS ping data to find high-demand hotspots.

    **Business value:** knowing *which* micro-zones spike at *which* hour lets
    operations teams pre-position drivers before demand hits — cutting rider
    wait times and reducing lost bookings from `No Driver Available` events.
    """
)

df = load_data()

# --- Sidebar controls ---
st.sidebar.header("Filters")

selected_hour = st.sidebar.slider("Hour of the Day", min_value=0, max_value=23, value=8)

selected_statuses = st.sidebar.multiselect(
    "Ride Status",
    options=sorted(df["status"].unique()),
    default=sorted(df["status"].unique()),
)

st.sidebar.caption(
    "Tip: select only **'No Driver Available'** to isolate service bottlenecks."
)

# --- Filter pipeline (vectorized boolean indexing, no loops) ---
filtered_df = df[(df["hour"] == selected_hour) & (df["status"].isin(selected_statuses))]

# --- Top-level metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Requests (this hour)", len(filtered_df))
col2.metric(
    "No Driver Available",
    int((filtered_df["status"] == "No Driver Available").sum()),
)
col3.metric(
    "Driver Assigned",
    int((filtered_df["status"] == "Driver Assigned").sum()),
)

# ============================================================
# 4. GEOSPATIAL VISUALIZATION
# ============================================================
st.subheader(f"📍 Demand Hotspots at {selected_hour:02d}:00")

if not filtered_df.empty:
    st.map(filtered_df[["latitude", "longitude"]], size=20)
else:
    st.warning("No data matches the selected hour/status filters. Try widening them.")

# --- Bonus: demand curve across the full day, for context ---
st.subheader("Demand Volume by Hour (all statuses selected)")
hourly_counts = (
    df[df["status"].isin(selected_statuses)]
    .groupby("hour")["status"]
    .count()
    .reindex(range(24), fill_value=0)
)
st.bar_chart(hourly_counts)

with st.expander("View raw filtered data"):
    st.dataframe(filtered_df)
