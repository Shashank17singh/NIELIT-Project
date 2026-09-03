import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
@st.cache_resource
def load_and_train():
    df = pd.read_csv("Mumbai House Prices.csv")
    df = df.dropna(
        subset=["bhk", "area", "price", "price_unit", "region", "type", "status", "age"]
    )
    def convert_price(row):
        p = row["price"]
        if row["price_unit"] == "Cr":
            return p * 10000000
        elif row["price_unit"] == "L":
            return p * 100000
        return p
    df["price_inr"] = df.apply(convert_price, axis=1)
    df = df[df["area"] < 5000]
    df = df[df["price_inr"] < 500000000]
    df = df[df["bhk"] < 10]
    top_regions = df["region"].value_counts().nlargest(50).index
    df["region_clean"] = df["region"].where(df["region"].isin(top_regions), "Other")
    X = df[["bhk", "area", "region_clean", "type", "status", "age"]]
    y = df["price_inr"]
    categorical_features = ["region_clean", "type", "status", "age"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            ),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, df, r2, mae
model, df, r2, mae = load_and_train()
def format_inr(number):
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
st.set_page_config(page_title="Mumbai House Price Predictor", layout="wide")
st.title("Mumbai House Price Predictor")
tab1, tab2 = st.tabs(["Price Predictor", "Data Analytics"])
with tab1:
    st.subheader("Estimate Property Value")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(
            "Area (in sqft):",
            min_value=100.0,
            max_value=10000.0,
            value=1000.0,
            step=50.0,
        )
        bhk = st.number_input("Number of BHK:", min_value=1, max_value=10, value=2)
        prop_type = st.selectbox("Property Type:", df["type"].unique())
    with col2:
        region = st.selectbox("Region:", sorted(df["region_clean"].unique()))
        status = st.selectbox("Status:", df["status"].unique())
        age = st.selectbox("Age of Property:", df["age"].unique())
    if st.button("Predict Price", type="primary"):
        input_data = pd.DataFrame(
            [
                {
                    "bhk": bhk,
                    "area": area,
                    "region_clean": region,
                    "type": prop_type,
                    "status": status,
                    "age": age,
                }
            ]
        )
        pred = model.predict(input_data)[0]
        st.success(f"### Estimated Price: {format_inr(pred)}")
with tab2:
    st.subheader("Market Insights & Model Performance")
    st.write("### Model Evaluation Metrics (Test Set)")
    m1, m2 = st.columns(2)
    m1.metric(label="R² Score (Accuracy)", value=f"{r2:.2f}")
    m2.metric(label="Mean Absolute Error (MAE)", value=format_inr(mae))
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.write("#### Average Price by Top 10 Regions")
        top_10 = (
            df[df["region_clean"] != "Other"]
            .groupby("region_clean")["price_inr"]
            .mean()
            .nlargest(10)
        )
        st.bar_chart(top_10)
    with col4:
        st.write("#### Price vs Area")
        sample_df = df.sample(min(2000, len(df)))
        fig, ax = plt.subplots()
        ax.scatter(
            sample_df["area"], sample_df["price_inr"] / 10000000, alpha=0.5, c="#00a4d6"
        )
        ax.set_xlabel("Area (sqft)")
        ax.set_ylabel("Price (Crores INR)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
