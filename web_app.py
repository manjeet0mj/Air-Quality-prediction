import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Air Quality Index Prediction",
    page_icon="",
    layout="wide"
)

# ------------------ LOAD MODELS (CACHED) ------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("best_extratrees_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    pca = pickle.load(open("pca.pkl", "rb"))  # PCA LOADED

    encoders = {
        "country": pickle.load(open("Country_encoder.pkl", "rb")),
        "city": pickle.load(open("City_encoder.pkl", "rb")),
        "co_cat": pickle.load(open("CO AQI Category_encoder.pkl", "rb")),
        "ozone_cat": pickle.load(open("Ozone AQI Category_encoder.pkl", "rb")),
        "no2_cat": pickle.load(open("NO2 AQI Category_encoder.pkl", "rb")),
        "pm25_cat": pickle.load(open("PM2.5 AQI Category_encoder.pkl", "rb")),
        "aqi_category": pickle.load(open("AQI Category_encoder.pkl", "rb")),
    }
    return model, scaler, pca, encoders


model, scaler, pca, encoders = load_artifacts()

# ------------------ HEADER ------------------
st.title("Air Quality Index Prediction Dashboard")
st.markdown(
    """
    Predict **Air Quality Index (AQI)** using pollutant levels, location data,  
    and a **PCA-optimized machine learning model**.
    """
)

st.divider()

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("Location Details")

country = st.sidebar.text_input("Country", placeholder="e.g. India")
city = st.sidebar.text_input("City", placeholder="e.g. Delhi")

lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.6f")
lng = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.6f")

st.sidebar.divider()

st.sidebar.header("Pollutant Information")

co_val = st.sidebar.number_input("CO AQI Value", min_value=0.0)
co_cat = st.sidebar.text_input("CO AQI Category", placeholder="Good / Moderate / Poor")

ozone_val = st.sidebar.number_input("Ozone AQI Value", min_value=0.0)
ozone_cat = st.sidebar.text_input("Ozone AQI Category")

no2_val = st.sidebar.number_input("NO2 AQI Value", min_value=0.0)
no2_cat = st.sidebar.text_input("NO2 AQI Category")

aqi_cat = st.sidebar.text_input("AQI Category", placeholder="Good / Moderate / Poor")
pm25_cat = st.sidebar.text_input("PM2.5 AQI Category")

# ------------------ MAIN CONTENT ------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Summary")
    st.write(
        {
            "Country": country,
            "City": city,
            "Latitude": lat,
            "Longitude": lng,
            "CO AQI": co_val,
            "Ozone AQI": ozone_val,
            "NO2 AQI": no2_val,
            "AQI Category": aqi_cat,
        }
    )

with col2:
    st.subheader("Prediction")

    if st.button("Predict AQI", use_container_width=True):
        try:
            # -------- Encode categorical features --------
            country_enc = encoders["country"].transform([country])[0]
            city_enc = encoders["city"].transform([city])[0]
            co_cat_enc = encoders["co_cat"].transform([co_cat])[0]
            ozone_cat_enc = encoders["ozone_cat"].transform([ozone_cat])[0]
            no2_cat_enc = encoders["no2_cat"].transform([no2_cat])[0]
            aqi_cat = encoders["aqi_category"].transform([aqi_cat])[0]
            pm25_cat_enc = encoders["pm25_cat"].transform([pm25_cat])[0]

            # -------- Prepare feature array --------
            data = np.array([[
                country_enc, city_enc, aqi_cat,
                co_val, co_cat_enc,
                ozone_val, ozone_cat_enc,
                no2_val, no2_cat_enc,
                pm25_cat_enc,
                lat, lng
            ]])

            # -------- Scale → PCA → Predict --------
            data_scaled = scaler.transform(data)
            data_pca = pca.transform(data_scaled)  #PCA APPLIED

            prediction = model.predict(data_pca)[0]

            st.success(f"**Predicted AQI Value:** {prediction:.2f}")

        except Exception as e:
            st.error("Input error or unseen category detected.")
            st.caption(str(e))

# ------------------ FOOTER ------------------
st.divider()
st.caption("Built with using Streamlit, PCA & Machine Learning")
