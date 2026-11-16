import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path


car_model = joblib.load("model/car_price_model.pkl")

car_model_data = pd.read_csv("data/car_data.csv")


st.set_page_config(
    page_title="AutoVal AI",
    page_icon="🚗",
    layout="wide"
)


css_file = Path("assets/css/style.css")

with open(css_file) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# HEADER

# HERO SECTION

with open("assets/images/image.png", "rb") as f:
    car_image = base64.b64encode(f.read()).decode()

st.html(f"""
<div style="
    position: relative;
    height: 500px;
    padding: 60px;
    border-radius: 30px;
    overflow: hidden;
    background:
        radial-gradient(circle at 75% 50%, rgba(37, 99, 235, 0.25), transparent 35%),
        linear-gradient(135deg, #0f172a, #172554);
">

    <div style="
        position: relative;
        z-index: 2;
        width: 48%;
    ">

        <div style="
            color: #38bdf8;
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 2px;
        ">
            ✦ AI VEHICLE INTELLIGENCE
        </div>

        <h1 style="
            color: white;
            font-size: 58px;
            line-height: 1.05;
            margin: 20px 0;
        ">
            Know Your Car's
            <br>
            <span style="color:#38bdf8;">
                True Value.
            </span>
        </h1>

        <p style="
            color: #94a3b8;
            font-size: 18px;
            line-height: 1.6;
            max-width: 500px;
        ">
            AutoVal AI analyzes vehicle specifications
            and estimates its market value using machine learning.
        </p>

    </div>

    <img
        src="data:image/png;base64,{car_image}"
        style="
            position: absolute;
            right: -40px;
            bottom: -20px;
            width: 62%;
            z-index: 1;
            filter: drop-shadow(0 25px 40px rgba(0,0,0,0.7));
        "
    >

</div>
""")

# VEHICLE VALUATION SECTION

st.markdown("""
<div class="section-title">
    <span>01</span>
    VEHICLE PROFILE
</div>

<h2 class="section-heading">
    Tell us about your vehicle.
</h2>

<p class="section-description">
    Enter your vehicle specifications and AutoVal AI will estimate
    its current market value.
</p>
""", unsafe_allow_html=True)


left, right = st.columns(2, gap="large")


# VEHICLE INPUTS


left, right = st.columns(2, gap="large")

with left:

    st.markdown("### Vehicle")

    name = st.selectbox(
        "Car",
        sorted(car_model_data["name"].unique())
    )

    year = st.number_input(
        "Manufacturing Year",
        min_value=int(car_model_data["year"].min()),
        max_value=int(car_model_data["year"].max()),
        value=2018,
        step=1
    )

    km_driven = st.number_input(
        "Kilometres Driven",
        min_value=0,
        max_value=int(car_model_data["km_driven"].max()),
        value=50000,
        step=1000
    )

    fuel = st.selectbox(
        "Fuel Type",
        sorted(car_model_data["fuel"].unique())
    )


with right:

    st.markdown("### Vehicle Details")

    seller_type = st.selectbox(
        "Seller Type",
        sorted(car_model_data["seller_type"].unique())
    )

    transmission = st.selectbox(
        "Transmission",
        sorted(car_model_data["transmission"].unique())
    )

    owner = st.selectbox(
        "Owner History",
        sorted(car_model_data["owner"].unique())
    )



st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "⚡ Estimate True Market Value",
    use_container_width=True,
    key="main_predict_button"
)

if predict_button:

   
    input_data = pd.DataFrame({
        "name": [name],
        "year": [year],
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner]
    })

   
    


    prediction = car_model.predict(input_data)[0]

    # Display result
    st.markdown(f"""
<div class="prediction-card">
<div class="prediction-label">
AI VALUATION COMPLETE
</div>

<div class="prediction-title">
Estimated Market Value
</div>

<div class="prediction-price">
₹{prediction:,.0f}
</div>

<div class="prediction-subtitle">
Based on your vehicle specifications
</div>

<div class="prediction-divider"></div>

<div class="prediction-footer">
<span>⚡ Random Forest ML</span>
<span>✦ AutoVal AI</span>
</div>
</div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
    <span>02</span>
    AI VALUATION ENGINE
</div>

<h2 class="section-heading">
    More than a price.
    <br>
    <span class="blue-text">A data-driven valuation.</span>
</h2>

<p class="section-description">
    AutoVal AI evaluates multiple vehicle attributes and uses a
    machine learning model to estimate the vehicle's market value.
</p>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3, gap="large")


with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-number">01</div>
        <h3>Vehicle Data</h3>
        <p>
            We analyze key specifications such as age, mileage,
            engine size, brand and condition.
        </p>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-number">02</div>
        <h3>Machine Learning</h3>
        <p>
            Our Random Forest model learns patterns from historical
            vehicle pricing data.
        </p>
    </div>
    """, unsafe_allow_html=True)


with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-number">03</div>
        <h3>Market Estimate</h3>
        <p>
            Your vehicle specifications are transformed into a
            predicted market value.
        </p>
    </div>
    """, unsafe_allow_html=True)

# MODEL PERFORMANCE


st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">
    <span>03</span>
    MODEL PERFORMANCE
</div>

<h2 class="section-heading">
    Built on measurable
    <br>
    <span class="blue-text">machine learning performance.</span>
</h2>

<p class="section-description">
    AutoVal AI was trained and evaluated on historical vehicle
    pricing data using a Random Forest regression model.
</p>
""", unsafe_allow_html=True)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">R²</div>
        <div class="metric-label">Model Score</div>
    </div>
    """, unsafe_allow_html=True)


with metric2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">2.5K+</div>
        <div class="metric-label">Vehicles</div>
    </div>
    """, unsafe_allow_html=True)


with metric3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">42</div>
        <div class="metric-label">ML Features</div>
    </div>
    """, unsafe_allow_html=True)


with metric4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">RF</div>
        <div class="metric-label">Model Type</div>
    </div>
    """, unsafe_allow_html=True)

# WHY AUTOVAL AI


st.markdown("""
<div class="benefits-grid">
<div class="benefit-card">
<div class="benefit-icon">⚡</div>
<div>
<h3>Instant Valuation</h3>
<p>Get an estimated market value in seconds instead of relying on manual research or guesswork.</p>
</div>
</div>

<div class="benefit-card">
<div class="benefit-icon">◈</div>
<div>
<h3>Data-Driven Estimates</h3>
<p>Predictions are generated from vehicle attributes learned from historical pricing data.</p>
</div>
</div>

<div class="benefit-card">
<div class="benefit-icon">◎</div>
<div>
<h3>Simple & Transparent</h3>
<p>No complicated valuation process. Enter your vehicle details and receive a clear estimated value.</p>
</div>
</div>

<div class="benefit-card">
<div class="benefit-icon">✦</div>
<div>
<h3>Built for Modern Vehicles</h3>
<p>Designed to work across different brands, models, fuel types and transmission configurations.</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# HOW AUTOVAL THINKS


st.markdown("""
<div class="section-heading">
<div class="section-eyebrow">THE INTELLIGENCE</div>

<h2>From vehicle data to<br><span>market intelligence.</span></h2>

<p>
AutoVal AI transforms vehicle specifications into
a data-driven valuation using machine learning.
</p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="pipeline">

<div class="pipeline-card">
<div class="pipeline-number">01</div>
<div class="pipeline-icon">🚘</div>
<h3>Vehicle Data</h3>
<p>
Brand, model, year, mileage, engine,
fuel type and condition.
</p>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-card">
<div class="pipeline-number">02</div>
<div class="pipeline-icon">◈</div>
<h3>Feature Intelligence</h3>
<p>
Vehicle attributes are transformed
into machine-readable features.
</p>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-card">
<div class="pipeline-number">03</div>
<div class="pipeline-icon">✦</div>
<h3>ML Prediction</h3>
<p>
The trained model analyzes learned
patterns from historical data.
</p>
</div>

<div class="pipeline-arrow">→</div>

<div class="pipeline-card pipeline-result">
<div class="pipeline-number">04</div>
<div class="pipeline-icon">₹</div>
<h3>True Market Value</h3>
<p>
Receive an estimated value
for your vehicle.
</p>
</div>

</div>
""", unsafe_allow_html=True)