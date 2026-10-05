import streamlit as st

st.set_page_config(page_title="House Price Predictor", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Josefin+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Josefin Sans', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Cinzel', serif !important;
    color: #0F766E !important;
    font-weight: 700 !important;
}

.stApp {
    background-color: #F0FDFA;
    color: #134E4A;
}

[data-testid="stHeader"] {
    background-color: rgba(240, 253, 250, 0.9) !important;
}

/* Glassmorphism Buttons */
.stButton > button {
    background-color: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    color: #0369A1;
    font-family: 'Cinzel', serif;
    font-weight: 600;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: all 0.2s ease-in-out;
}

.stButton > button:hover {
    background-color: rgba(255, 255, 255, 0.7);
    border-color: rgba(255, 255, 255, 0.8);
    transform: translateY(-2px);
    box-shadow: 0 8px 12px rgba(0,0,0,0.1);
}

/* Glassmorphism Containers */
[data-testid="stExpander"], [data-testid="stVerticalBlock"] > div > div > div[data-testid="stContainer"] {
    background-color: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 12px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
    padding: 15px;
}

/* Inputs */
.stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input {
    background-color: rgba(255, 255, 255, 0.8);
    border: 1px solid #99F6E4;
    border-radius: 8px;
    color: #134E4A;
}
.stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus, .stNumberInput > div > div > input:focus {
    border-color: #0F766E;
    box-shadow: 0 0 0 2px rgba(15,118,110,0.2);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def init_db():
    """Initializes the SQLite database for prediction history."""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prediction_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  area REAL, bhk INTEGER, prop_type TEXT, region TEXT, 
                  status TEXT, age TEXT, predicted_price REAL)''')
    conn.commit()
    conn.close()

def save_prediction(area, bhk, prop_type, region, status, age, price):
    """Saves a prediction instance to the SQLite database."""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO prediction_history (area, bhk, prop_type, region, status, age, predicted_price)
                 VALUES (?, ?, ?, ?, ?, ?, ?)''', (area, bhk, prop_type, region, status, age, price))
    conn.commit()
    conn.close()

def load_history():
    """Loads prediction history from the SQLite database."""
    conn = sqlite3.connect('predictions.db')
    df_hist = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
    conn.close()
    return df_hist

init_db()


@st.cache_resource
def load_and_train():
    """Loads the dataset, preprocesses it, and trains a RandomForestRegressor model."""
    df = pd.read_csv("Mumbai House Prices.csv")
    df = df.dropna(
        subset=["bhk", "area", "price", "price_unit", "region", "type", "status", "age"]
    )

    def convert_price(row: pd.Series) -> float:
        """Converts price to INR based on the price unit (Cr, L)."""
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
    numeric_features = ["bhk", "area"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )
    model_rf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ]
    )
    
    model_lr = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model_rf.fit(X_train, y_train)
    model_lr.fit(X_train, y_train)
    
    y_pred_rf = model_rf.predict(X_test)
    y_pred_lr = model_lr.predict(X_test)

    metrics = {
        "rf": {"r2": r2_score(y_test, y_pred_rf), "mae": mean_absolute_error(y_test, y_pred_rf)},
        "lr": {"r2": r2_score(y_test, y_pred_lr), "mae": mean_absolute_error(y_test, y_pred_lr)}
    }
    
    return model_rf, model_lr, df, metrics


model_rf, model_lr, df, metrics = load_and_train()


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
tab1, tab2, tab3 = st.tabs(["Price Predictor", "Data Analytics", "Prediction History (SQLite)"])
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
        pred = model_rf.predict(input_data)[0]
        save_prediction(area, bhk, prop_type, region, status, age, pred)
        st.success(f"### Estimated Price: {format_inr(pred)}")
with tab2:
    st.subheader("Market Insights & Model Performance")
    st.write("### Model Evaluation Metrics (Test Set)")
    
    st.write("**Random Forest (Advanced Implementation)**")
    m1, m2 = st.columns(2)
    m1.metric(label="R² Score (Accuracy)", value=f"{metrics['rf']['r2']:.2f}")
    m2.metric(label="Mean Absolute Error (MAE)", value=format_inr(metrics['rf']['mae']))
    
    st.write("**Linear Regression (NIELIT Course Baseline)**")
    m3, m4 = st.columns(2)
    m3.metric(label="R² Score (Accuracy)", value=f"{metrics['lr']['r2']:.2f}")
    m4.metric(label="Mean Absolute Error (MAE)", value=format_inr(metrics['lr']['mae']))
    
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
