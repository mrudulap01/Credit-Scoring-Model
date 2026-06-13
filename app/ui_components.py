import streamlit as st

def render_sidebar():
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
    st.sidebar.title("CreditScorer AI")
    st.sidebar.markdown("""
    Welcome to the **CreditScorer AI** Dashboard. 
    
    This application uses an advanced Machine Learning model (Random Forest) 
    to evaluate applicant data and predict their creditworthiness based on 
    historical financial behaviors.
    """)
    st.sidebar.divider()
    st.sidebar.info("Model: Random Forest\n\nMetric: ROC-AUC Optimized")

def render_personal_info_section():
    st.subheader("👤 1. Personal Information")
    col1, col2 = st.columns(2)
    
    # User-Friendly Mappings
    status_map = {
        "Single Male": "male single", 
        "Divorced/Married/Separated Female": "female div/dep/mar", 
        "Married/Widowed Male": "male mar/wid", 
        "Divorced/Separated Male": "male div/sep"
    }
    housing_map = {"Homeowner": "own", "Renting": "rent", "Living for Free": "for free"}

    with col1:
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=30, help="Applicant's current age.")
        display_status = st.selectbox("Marital & Gender Status", list(status_map.keys()))
        display_housing = st.selectbox("Housing Situation", list(housing_map.keys()))
        residence_since = st.number_input("Years at Present Residence", min_value=1, max_value=4, value=2, step=1, help="How long has the applicant lived at their current address?")
        
    with col2:
        num_dependents = st.number_input("Number of Dependents", min_value=1, max_value=2, value=1, step=1, help="Number of people financially dependent on the applicant.")
        foreign_worker = st.selectbox("Foreign Worker?", ["yes", "no"])
        own_telephone = st.selectbox("Owns a registered telephone?", ["yes", "none"])
        
    return {
        "age": age,
        "personal_status": status_map[display_status],
        "housing": housing_map[display_housing],
        "residence_since": residence_since,
        "num_dependents": num_dependents,
        "foreign_worker": foreign_worker,
        "own_telephone": own_telephone
    }

def render_financial_info_section():
    st.subheader("💰 2. Financial Information")
    col1, col2 = st.columns(2)
    
    # User-Friendly Mappings
    chk_map = {
        "No Checking Account": "no checking",
        "Negative Balance (< 0 DM)": "<0",
        "Low Balance (0 - 200 DM)": "0<=X<200",
        "High Balance (>= 200 DM)": ">=200"
    }
    sav_map = {
        "No Known Savings": "no known savings",
        "Low (< 100 DM)": "<100",
        "Moderate (100 - 500 DM)": "100<=X<500",
        "High (500 - 1000 DM)": "500<=X<1000",
        "Very High (>= 1000 DM)": ">=1000"
    }
    prop_map = {
        "Real Estate": "real estate",
        "Life Insurance": "life insurance",
        "Car": "car",
        "No Known Property": "no known property"
    }
    emp_map = {
        "Unemployed": "unemployed",
        "Less than 1 year": "<1",
        "1 to 4 years": "1<=X<4",
        "4 to 7 years": "4<=X<7",
        "More than 7 years": ">=7"
    }
    job_map = {
        "Unskilled / Resident": "unskilled resident",
        "Skilled Employee": "skilled",
        "Highly Qualified / Management / Self-Employed": "high qual/self emp/mgmt",
        "Unemployed / Unskilled Non-Resident": "unemp/unskilled non res"
    }
    hist_map = {
        "All Paid Back Properly": "all paid",
        "Existing Credits Paid Back Properly": "existing paid",
        "Delayed in the Past": "delayed previously",
        "Critical Account / Other Credits Elsewhere": "critical/other existing credit",
        "No Credits / All Paid": "no credits/all paid"
    }

    with col1:
        display_chk = st.selectbox("Checking Account Status", list(chk_map.keys()), help="Current status of the applicant's primary checking account.")
        display_sav = st.selectbox("Savings Account Status", list(sav_map.keys()), help="Current status of the applicant's savings.")
        display_prop = st.selectbox("Most Valuable Property Owned", list(prop_map.keys()))
        display_emp = st.selectbox("Employment Duration", list(emp_map.keys()), help="How long has the applicant been at their current job?")
        
    with col2:
        display_job = st.selectbox("Job Type", list(job_map.keys()))
        display_hist = st.selectbox("Credit History", list(hist_map.keys()), help="Applicant's historical payment behavior.")
        existing_credits = st.number_input("Existing Credits at this Bank", min_value=1, max_value=4, value=1, step=1)
        
    return {
        "checking_status": chk_map[display_chk],
        "savings_status": sav_map[display_sav],
        "property_magnitude": prop_map[display_prop],
        "employment": emp_map[display_emp],
        "job": job_map[display_job],
        "credit_history": hist_map[display_hist],
        "existing_credits": existing_credits
    }

def render_loan_info_section():
    st.subheader("📝 3. Loan Details")
    col1, col2 = st.columns(2)
    
    # User-Friendly Mappings
    purpose_map = {
        "Radio/TV": "radio/tv",
        "Education": "education",
        "Furniture/Equipment": "furniture/equipment",
        "New Car": "new car",
        "Used Car": "used car",
        "Business": "business",
        "Domestic Appliance": "domestic appliance",
        "Repairs": "repairs",
        "Retraining": "retraining",
        "Other": "other"
    }
    parties_map = {"None": "none", "Guarantor": "guarantor", "Co-applicant": "co applicant"}
    plans_map = {"None": "none", "Bank": "bank", "Stores": "stores"}

    with col1:
        credit_amount = st.number_input("Credit Amount Requested ($)", min_value=100, max_value=25000, value=2500, step=100, help="Total loan amount requested.")
        duration = st.number_input("Loan Duration (Months)", min_value=4, max_value=72, value=24, step=1, help="How many months will the loan repayment last?")
        display_purpose = st.selectbox("Purpose of Loan", list(purpose_map.keys()))
        
    with col2:
        installment_commitment = st.number_input("Installment Rate (% of disposable income)", min_value=1, max_value=4, value=2, step=1, help="Percentage of disposable income dedicated to the loan.")
        display_parties = st.selectbox("Other Debtors / Guarantors", list(parties_map.keys()))
        display_plans = st.selectbox("Other Active Payment Plans", list(plans_map.keys()))
        
    return {
        "credit_amount": credit_amount,
        "duration": duration,
        "purpose": purpose_map[display_purpose],
        "installment_commitment": installment_commitment,
        "other_parties": parties_map[display_parties],
        "other_payment_plans": plans_map[display_plans]
    }
