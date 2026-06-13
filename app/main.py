import streamlit as st
import sys
import os

# Ensure the parent directory is in the path 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_components import render_sidebar, render_personal_info_section, render_financial_info_section, render_loan_info_section
from prediction import load_pipeline, predict_credit_risk, determine_risk_level

st.set_page_config(
    page_title="CreditScorer AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    /* Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0a192f, #1e3a8a, #0f172a, #111827);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Form Card */
    div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 3rem !important;
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stForm"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6) !important;
    }
    
    /* Input fields styling on hover */
    div[data-baseweb="input"] > div, 
    div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        transition: all 0.3s ease;
    }
    
    div[data-baseweb="input"] > div:hover, 
    div[data-baseweb="select"] > div:hover {
        border: 1px solid rgba(56, 189, 248, 0.8) !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Button Animation */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #0ea5e9, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6) !important;
    }
    
    /* Metrics styling */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
        text-align: center !important;
    }
    
    /* Typography */
    h1 {
        color: #ffffff !important;
        font-weight: 900 !important;
        text-align: center !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        padding-bottom: 0.5rem !important;
        text-shadow: 0 4px 20px rgba(56, 189, 248, 0.4) !important;
    }
    h3 {
        color: #38bdf8 !important;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2) !important;
        padding-bottom: 0.8rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
        margin: 2rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    inject_custom_css()
    render_sidebar()
    
    st.title("🏦 Premium Credit Dashboard")
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Evaluate applicant creditworthiness using your custom-trained Random Forest model.</p>", unsafe_allow_html=True)
    
    # Load ML Pipeline
    preprocessor, model = load_pipeline()
    
    if preprocessor is None or model is None:
        st.stop() # Halt execution if models are missing
        
    with st.form("credit_form"):
        # 1. Personal Info
        personal_data = render_personal_info_section()
        st.divider()
        
        # 2. Financial Info
        financial_data = render_financial_info_section()
        st.divider()
        
        # 3. Loan Details
        loan_data = render_loan_info_section()
        
        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Creditworthiness", type="primary", use_container_width=True)

    if submitted:
        # Combine all dictionaries into one single data record
        input_data = {**personal_data, **financial_data, **loan_data}
        
        with st.spinner("Analyzing Applicant Profile with Random Forest..."):
            probability = predict_credit_risk(input_data, preprocessor, model)
            
            if probability is not None:
                risk_label, icon, color_theme = determine_risk_level(probability)
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Display Metrics using columns
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    st.metric(
                        label="Approval Probability",
                        value=f"{probability * 100:.1f}%",
                        delta="Favorable" if probability >= 0.5 else "Unfavorable",
                        delta_color="normal"
                    )
                
                with col2:
                    st.metric(
                        label="Assessed Risk Level",
                        value=f"{icon} {risk_label}",
                        delta=None
                    )
                
                # Visual feedback and Pro-Tips based on risk
                st.markdown("<br>", unsafe_allow_html=True)
                if color_theme == "success":
                    st.success("✅ **Recommendation: APPROVE.** The applicant demonstrates a solid financial profile with a high probability of repayment.")
                    st.info("💡 **Pro-Tip (Excellent Profile):** This applicant is highly reliable. Consider offering them premium banking products (e.g., a high-tier credit card) or a promotional interest rate to build long-term loyalty.")
                elif color_theme == "warning":
                    st.warning("⚠️ **Recommendation: MANUAL REVIEW.** The applicant shows moderate risk factors. Proceed with caution or require a guarantor.")
                    st.info("💡 **Suggestion to Improve Score:** This application is on the borderline. To reduce the assessed risk, the applicant could:\n- Add a **Guarantor** or **Co-applicant** to the loan.\n- Reduce the requested **Credit Amount**.\n- Extend the **Loan Duration** to lower the monthly installment burden.")
                else:
                    st.error("🚫 **Recommendation: REJECT.** The applicant poses a high risk of default based on historical data patterns.")
                    st.info("💡 **Steps to Improve Creditworthiness:** To strengthen their profile for future applications, the applicant should focus on:\n- Building up their **Savings Account** balance to act as a safety net.\n- Establishing a flawless **Credit History** with smaller, manageable credits first.\n- Lowering their existing financial commitments before taking on new debt.")

if __name__ == "__main__":
    main()
