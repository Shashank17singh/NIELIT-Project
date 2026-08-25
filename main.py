import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

@st.cache_resource # Caches the model so it doesn't retrain on every click
def train_model():
    data = {
        "Area": [1200, 800, 1600, 1000, 1400, 1800, 1100, 950, 1300],
        "Bedrooms": [3, 2, 4, 2, 3, 4, 2, 1, 3],
        "Bathrooms": [2, 1, 3, 2, 2, 3, 1, 1, 2],
        "Location": ["Standard", "Budget", "Premium", "Standard", "Premium", "Premium", "Budget", "Budget", "Standard"],
        "Parking": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes"],
        "Age": [5, 10, 2, 8, 3, 1, 7, 9, 4],
        "Price": [12900000, 4850000, 19500000, 8950000, 16750000, 21100000, 7300000, 5050000, 13800000]
    }
    df = pd.DataFrame(data)
    location_map = {"Budget": 0, "Standard": 1, "Premium": 2}
    parking_map = {"No": 0, "Yes": 1}
    
    df["Location"] = df["Location"].map(location_map)
    df["Parking"] = df["Parking"].map(parking_map)
    
    X = df.drop("Price", axis=1)
    y = df["Price"]
    model = LinearRegression()
    model.fit(X, y)
    
    return model, location_map, parking_map

model, location_map, parking_map = train_model()

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
    return res

# --- Web UI ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("House Price Predictor")

# Inputs
area = st.number_input("Area (in sqft):", min_value=100.0, value=1000.0, step=50.0)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, value=2)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1, value=2)
location = st.selectbox("Location:", ["Budget", "Standard", "Premium"])
parking = st.selectbox("Parking:", ["No", "Yes"])
age = st.number_input("Age of Property (Years):", min_value=0, value=5)

if st.button("Predict Price", type="primary"):
    loc_encoded = location_map[location]
    park_encoded = parking_map[parking]
    
    input_df = pd.DataFrame([{
        "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Location": loc_encoded, "Parking": park_encoded, "Age": age
    }])
    
    predicted_price = max(0, model.predict(input_df)[0])
    st.success(f"### Estimated House Price: ₹{format_inr(predicted_price)}")
