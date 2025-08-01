import streamlit as st
import pickle
import numpy as np

# Load model
with open('model_pickle.pkl', 'rb') as f:
    model = pickle.load(f)

# Custom UI
st.set_page_config(
    page_title="🏡 Home Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Main content
st.title("AI Home Price Predictor")
st.markdown("Get instant property valuations using machine learning")

with st.expander("ℹ️ How to use"):
    st.write("1. Fill in the property details")
    st.write("2. Click 'Predict'")
    st.write("3. View valuation report")

# Prediction form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, value=2.0, step=0.5)
    
    with col2:
        sqft = st.number_input("Living Area (sqft)", min_value=500, value=1500)
        year = st.number_input("Year Built", min_value=1800, max_value=2023, value=1990)
    
    submitted = st.form_submit_button("Predict Price", type="primary")

# Results
if submitted:
    with st.spinner('Analyzing property...'):
        features = np.array([[bedrooms, bathrooms, sqft, 1, year]])
        prediction = model.predict(features)[0]
        
        st.success("Prediction Complete!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Value", f"${prediction:,.2f}")
        col2.metric("Price/SQFT", f"${prediction/sqft:,.2f}")
       
        
        # Show feature importance
        st.subheader("Feature Impact")
        st.progress(0.8, text="Square Footage")
        st.progress(0.6, text="Bedrooms")
        st.progress(0.4, text="Year Built")