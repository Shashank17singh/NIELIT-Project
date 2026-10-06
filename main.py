"""
House Price Predictor application for Mumbai properties.
Integrates SQLite persistence and Scikit-Learn modeling pipelines.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from db import init_db, save_prediction, load_history
from ml import load_and_train

st.set_page_config(page_title="House Price Predictor", layout="wide")

init_db()

@st.cache_resource
def cached_load_and_train():
    return load_and_train()

model_rf, model_lr, df, metrics = cached_load_and_train()

def format_inr(number: float) -> str:
    """Formats a number into an Indian Rupee string (e.g., ₹1,00,000)."""
    s = str(int(number))
    if len(s) <= 3:
        return s
    res = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        res = s[-2:] + "," + res
        s = s[:-2]
    res = s + "," + res
    return "₹" + res

st.title("House Price Predictor")
tab1, tab2, tab3 = st.tabs(
    ["Price Predictor", "Data Analytics", "Prediction History (SQLite)"]
)

with tab1:
    st.subheader("Estimate Property Value")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(
            "Area (in sqft):", min_value=100.0, max_value=10000.0, value=1000.0, step=50.0
        )
        bhk = st.number_input("Number of BHK:", min_value=1, max_value=10, value=2)
        prop_type = st.selectbox("Property Type:", df["type"].unique())
    with col2:
        region = st.selectbox("Region:", sorted(df["region_clean"].unique()))
        status = st.selectbox("Status:", df["status"].unique())
        age = st.selectbox("Age of Property:", df["age"].unique())

    if st.button("Predict Price", type="primary"):
        input_data = pd.DataFrame([
            {"bhk": bhk, "area": area, "region_clean": region, "type": prop_type, "status": status, "age": age}
        ])
        pred = model_rf.predict(input_data)[0]
        save_prediction(area, bhk, prop_type, region, status, age, pred)
        st.success(f"### Estimated Price: {format_inr(pred)}")

with tab2:
    st.subheader("Market Insights & Model Performance")
    st.write("### Model Evaluation Metrics (Test Set)")

    st.write("**Random Forest (Advanced Implementation)**")
    m1, m2 = st.columns(2)
    m1.metric(label="R² Score (Accuracy)", value=f"{metrics['rf']['r2']:.2f}")
    m2.metric(label="Mean Absolute Error (MAE)", value=format_inr(metrics["rf"]["mae"]))

    st.write("**Linear Regression (NIELIT Course Baseline)**")
    m3, m4 = st.columns(2)
    m3.metric(label="R² Score (Accuracy)", value=f"{metrics['lr']['r2']:.2f}")
    m4.metric(label="Mean Absolute Error (MAE)", value=format_inr(metrics["lr"]["mae"]))

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.write("#### Average Price by Top 10 Regions")
        top_10 = df[df["region_clean"] != "Other"].groupby("region_clean")["price_inr"].mean().nlargest(10)
        st.bar_chart(top_10)
    
    with col4:
        st.write("#### Price vs Area")
        sample_df = df.sample(min(2000, len(df)))
        fig, ax = plt.subplots()
        ax.scatter(sample_df["area"], sample_df["price_inr"] / 10000000, alpha=0.5, c="#00a4d6")
        ax.set_xlabel("Area (sqft)")
        ax.set_ylabel("Price (Crores INR)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)

    st.divider()
    st.write("#### Feature Correlation Heatmap")
    numeric_df = df[["bhk", "area", "price_inr"]].dropna()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

with tab3:
    st.subheader("Prediction History")
    st.write("This tab retrieves past predictions stored in a local **SQLite Database**, demonstrating data persistence.")
    history_df = load_history()
    if not history_df.empty:
        history_df["predicted_price_formatted"] = history_df["predicted_price"].apply(format_inr)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions made yet. Go to the 'Price Predictor' tab to make your first prediction!")
