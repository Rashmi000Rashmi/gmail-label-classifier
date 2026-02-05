import sys
import types

# Polyfill for imghdr (removed in Python 3.13)
# Streamlit internally imports this, so we provide a dummy module to prevent crash.
if sys.version_info >= (3, 13):
    sys.modules['imghdr'] = types.ModuleType('imghdr')
    sys.modules['imghdr'].what = lambda *args, **kwargs: None

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page config
st.set_page_config(page_title="Gmail Classifier Dashboard", page_icon="📧", layout="wide")

st.title("📧 Gmail Classifier Dashboard")
st.markdown("---")

metrics_file = os.path.join('data', 'metrics.csv')

if not os.path.exists(metrics_file):
    st.warning("⚠️ No metrics found. Run the sync script first to generate `data/metrics.csv`.")
    st.stop()

# Load data
df = pd.read_csv(metrics_file)
df['date_only'] = pd.to_datetime(df['date_only'])
df = df.sort_values('date_only')

# --- SIDEBAR ---
st.sidebar.header("Filter")
labels = st.sidebar.multiselect("Select Labels", options=df['label'].unique(), default=df['label'].unique())
filtered_df = df[df['label'].isin(labels)]

# --- METRICS ---
col1, col2 = st.columns(2)
total_conf = df[df['label'] == "Application_Confirmation"]['count'].sum()
total_rej = df[df['label'] == "Rejected"]['count'].sum()

with col1:
    st.metric("Total Confirmations ✅", total_conf)
with col2:
    st.metric("Total Rejections ❌", total_rej)

st.markdown("---")

# --- GRAPHS ---

# 1. Total Cumulative Growth
st.subheader("📈 Cumulative Growth Until Now")
pivot_df = df.pivot(index='date_only', columns='label', values='count').fillna(0)
cumulative_df = pivot_df.cumsum()
fig_line = px.line(cumulative_df, labels={"value": "Total Count", "date_only": "Date"}, color_discrete_map={"Application_Confirmation": "#4CAF50", "Rejected": "#FF5252"})
st.plotly_chart(fig_line, use_container_width=True)

col3, col4 = st.columns(2)

# 2. Daily Confirmations
with col3:
    st.subheader("✅ Daily Confirmations")
    conf_df = df[df['label'] == "Application_Confirmation"]
    fig_conf = px.bar(conf_df, x='date_only', y='count', color_discrete_sequence=['#4CAF50'])
    st.plotly_chart(fig_conf, use_container_width=True)

# 3. Daily Rejections
with col4:
    st.subheader("❌ Daily Rejections")
    rej_df = df[df['label'] == "Rejected"]
    fig_rej = px.bar(rej_df, x='date_only', y='count', color_discrete_sequence=['#FF5252'])
    st.plotly_chart(fig_rej, use_container_width=True)

st.markdown("---")
st.caption("Dashboard updated automatically via your local Gmail sync pipeline.")
