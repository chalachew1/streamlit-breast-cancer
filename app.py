import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("breast_cancer.csv")
df = df.loc[:, df.columns.notnull()]
df = df.dropna(how="all", axis=1)
df = df.dropna()
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()

# Sidebar: Filters
st.sidebar.title("Filters")
diagnosis_filter = st.sidebar.multiselect("Select Diagnosis Type", options=["B", "M"], default=["B", "M"])
selected_columns = st.sidebar.multiselect("Select Features to Plot", options=df.columns[2:], default=["radius_mean", "texture_mean"])

# Filter data based on sidebar
filtered_df = df[df["diagnosis"].isin(diagnosis_filter)]

# Main Dashboard
st.title("Interactive Breast Cancer Data Dashboard")

# Diagnosis count
st.subheader("Diagnosis Count")
diag_count = filtered_df["diagnosis"].value_counts()
st.bar_chart(diag_count)

# Summary statistics (fix for older pandas)
st.subheader("Summary Statistics")
numeric_stats = filtered_df.select_dtypes(include='number').describe().T
st.dataframe(numeric_stats)

# Feature comparison chart
st.subheader("Feature Distribution by Diagnosis")
for col in selected_columns:
    st.line_chart(filtered_df.groupby("diagnosis")[col].mean())

# Radius-to-Perimeter Ratio
if "radius_mean" in df.columns and "perimeter_mean" in df.columns:
    st.subheader("Radius-to-Perimeter Ratio (Malignant Cases)")
    malignant = df[df["diagnosis"] == "M"].copy()
    malignant["radius_perimeter_ratio"] = malignant["radius_mean"] / malignant["perimeter_mean"]
    st.line_chart(malignant["radius_perimeter_ratio"].head(20))

# Sum of _se columns
st.subheader("Sum of Standard Error Features")
stddev_sum = df.filter(like="_se").sum(numeric_only=True).sum()
st.metric("Total SE Sum", f"{stddev_sum:.2f}")