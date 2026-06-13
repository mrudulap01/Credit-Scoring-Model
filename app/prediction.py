import joblib
import pandas as pd
import streamlit as st
import os

@st.cache_resource
def load_pipeline():
    """
    Loads the preprocessor and the trained model.
    Using @st.cache_resource prevents loading from disk on every interaction.
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.joblib')
        model_path = os.path.join(base_dir, 'models', 'best_model.joblib')
        
        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        return preprocessor, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure you have run the training pipeline first (python main.py)")
        return None, None

def predict_credit_risk(input_data, preprocessor, model):
    """
    Takes raw input data, processes it, and returns predictions.
    """
    try:
        # Create a DataFrame from the input dict
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing
        X_processed = preprocessor.transform(df)
        
        # Predict probability of class 1 ('good' credit)
        probability = model.predict_proba(X_processed)[0][1] 
        
        return probability
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

def determine_risk_level(probability):
    """
    Classifies the applicant into a risk bucket based on Approval Probability.
    Approval Probability = Probability of being 'good'.
    Therefore, Default Probability = 1 - Approval Probability.
    
    Low Risk: Approval > 70%
    Medium Risk: 50% <= Approval <= 70%
    High Risk: Approval < 50%
    """
    if probability > 0.70:
        return "Low Risk", "🟢", "success"
    elif probability >= 0.50:
        return "Medium Risk", "🟡", "warning"
    else:
        return "High Risk", "🔴", "error"
