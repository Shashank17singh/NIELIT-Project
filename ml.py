"""
Machine learning pipeline for the House Price Predictor.
Handles data loading, preprocessing, and model training.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _convert_price(row: pd.Series) -> float:
    """Normalizes the price unit to raw INR."""
    p = row["price"]
    if row["price_unit"] == "Cr":
        return p * 10000000
    elif row["price_unit"] == "L":
        return p * 100000
    return p


def load_and_train() -> tuple[
    Pipeline, Pipeline, pd.DataFrame, dict[str, dict[str, float]]
]:
    """
    Loads dataset, preprocesses features, and trains Random Forest and Linear Regression models.
    Returns: (rf_model, lr_model, cleaned_dataframe, metrics_dictionary)
    """
    df = pd.read_csv("Mumbai House Prices.csv")
    df = df.dropna(
        subset=["bhk", "area", "price", "price_unit", "region", "type", "status", "age"]
    )

    df["price_inr"] = df.apply(_convert_price, axis=1)

    # Remove extreme outliers based on simple domain knowledge
    df = df[df["area"] < 5000]
    df = df[df["price_inr"] < 500000000]
    df = df[df["bhk"] < 10]

    # Group uncommon regions into "Other"
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
            (
                "regressor",
                RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            ),
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
        "rf": {
            "r2": r2_score(y_test, y_pred_rf),
            "mae": mean_absolute_error(y_test, y_pred_rf),
        },
        "lr": {
            "r2": r2_score(y_test, y_pred_lr),
            "mae": mean_absolute_error(y_test, y_pred_lr),
        },
    }

    return model_rf, model_lr, df, metrics
