import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config FIRST
st.set_page_config(page_title="Aluminium Wire Rod Predictor", layout="centered")

# Title
st.title("Aluminium Wire Rod Property Predictor")
st.markdown("""
Predict *UTS, **Elongation, and **Conductivity* based on process parameters:
- Casting Temperature (°C)
- Rolling Speed (m/min)
- Cooling Rate (°C/s)
""")

# Load models with cache
@st.cache_resource
def load_models():
    return {
        'uts': joblib.load('models/best_uts_model.joblib'),
        'elongation': joblib.load('models/best_elongation_model.joblib'),
        'conductivity': joblib.load('models/best_conductivity_model.joblib'),
        'scaler': joblib.load('models/scaler.joblib'),
        'ranges': joblib.load('models/training_ranges.joblib')
    }

with st.spinner("Loading prediction models..."):
    models = load_models()
    scaler = models['scaler']
    ranges = models['ranges']

# Input section
st.sidebar.header("📥 Process Parameters")
user_input = {}

for feature, (min_val, max_val) in ranges.items():
    default = (min_val + max_val) / 2
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        step=0.1
    )

# Display current parameters
st.sidebar.subheader("Current Settings:")
for feature, value in user_input.items():
    st.sidebar.write(f"- {feature}: {value}")

# Create DataFrame with correct feature order
input_df = pd.DataFrame([user_input])[list(ranges.keys())]

# Prediction button
if st.button("🚀 Predict Properties"):
    # Scale input
    X_scaled = scaler.transform(input_df.values)
    
    # Predict
    uts_pred = models['uts'].predict(X_scaled)[0]
    elongation_pred = models['elongation'].predict(X_scaled)[0]
    conductivity_pred = models['conductivity'].predict(X_scaled)[0]
    
    # Display results
    st.subheader("📈 Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("UTS (MPa)", f"{uts_pred:.2f}")
    col2.metric("Elongation (%)", f"{elongation_pred:.2f}")
    col3.metric("Conductivity (% IACS)", f"{conductivity_pred:.2f}")
    
    # Range validation
    warnings = []
    if not (190 <= uts_pred <= 230):
        warnings.append(f"UTS ({uts_pred:.1f} MPa) outside typical range (190-230 MPa)")
    if not (8 <= elongation_pred <= 15):
        warnings.append(f"Elongation ({elongation_pred:.1f}%) outside typical range (8-15%)")
    if not (55 <= conductivity_pred <= 65):
        warnings.append(f"Conductivity ({conductivity_pred:.1f}% IACS) outside typical range (55-65%)")
    
    if warnings:
        st.warning("\n\n".join(warnings))
    else:
        st.success("✅ All properties within optimal ranges!")
        
    # Debug info
    with st.expander("Technical Details"):
        st.write("Scaled Input Values:", X_scaled)
        st.write("Model Input Features:", list(ranges.keys()))

# Footer
st.markdown("---")
st.caption("Models trained on synthetic aluminum wire rod data")
st.caption(f"Valid input ranges: "
           f"Casting: {ranges['Casting_Temperature_C'][0]:.1f}-{ranges['Casting_Temperature_C'][1]:.1f}°C, "
           f"Rolling: {ranges['Rolling_Speed_m_min'][0]:.1f}-{ranges['Rolling_Speed_m_min'][1]:.1f} m/min, "
           f"Cooling: {ranges['Cooling_Rate_C_s'][0]:.1f}-{ranges['Cooling_Rate_C_s'][1]:.1f}°C/s")
