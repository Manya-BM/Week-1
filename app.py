import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="EV Price Predictor", page_icon="🔋", layout="centered")
st.title("🔋 Electric Vehicle Price Prediction")
st.markdown("### Predict EV price based on specifications")
st.divider()

@st.cache_data
def train_model():
    # Load dataset
    df = pd.read_csv("EV_cars.csv")

    # ✅ Handle missing values safely
    df = df.dropna(subset=['Battery', 'Efficiency', 'Fast_charge', 'Range', 'Top_speed', 'acceleration..0.100.', 'Price.DE.'])
    # or df = df.fillna(df.mean()) if you prefer to fill instead of drop

    # Features and target
    X = df[['Battery', 'Efficiency', 'Fast_charge', 'Range', 'Top_speed', 'acceleration..0.100.']]
    y = df['Price.DE.']

    # ✅ Create model pipeline that automatically handles any remaining NaN
    model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model

# Train and cache model
model = train_model()

# ---- Input fields
battery = st.number_input("Battery Capacity (kWh)", min_value=10.0, max_value=200.0, value=60.0, step=1.0)
efficiency = st.number_input("Efficiency (Wh/km)", min_value=100.0, max_value=300.0, value=150.0, step=1.0)
fast_charge = st.number_input("Fast Charging Power (kW)", min_value=10.0, max_value=400.0, value=100.0, step=5.0)
range_km = st.number_input("Range (km)", min_value=50.0, max_value=1000.0, value=400.0, step=10.0)
top_speed = st.number_input("Top Speed (km/h)", min_value=80.0, max_value=300.0, value=180.0, step=5.0)
acceleration = st.number_input("0–100 km/h Acceleration (sec)", min_value=2.0, max_value=15.0, value=7.5, step=0.1)

st.divider()

if st.button("🚀 Predict EV Price"):
    input_data = np.array([[battery, efficiency, fast_charge, range_km, top_speed, acceleration]])
    pred = model.predict(input_data)
    st.success(f"💰 Estimated EV Price: €{pred[0]:,.2f}")
