import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Bank Churn Predictor", layout="wide", page_icon="🏦")

st.title("🏦 Bank Customer Churn Prediction Engine")
st.markdown("Use the control panel to adjust customer features. Observe the churn probability changes in real-time.")
st.divider()

# 2. Load Data and Train Model
@st.cache_resource
def train_model():
    df = pd.read_excel("European_Bank_Churn_Dashboard.xlsx", sheet_name="Working_Data")
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    features = ['Age', 'NumOfProducts', 'IsActiveMember', 'BalanceSalaryRatio', 'ProductDensity', 'Gender']
    
    X = df[features]
    y = df['Exited']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importance for the dashboard
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return model, importance

model, feature_importance = train_model()

# 3. Sidebar Control Panel (What-If Simulator)
st.sidebar.header("⚙️ Customer Profile Input")
age = st.sidebar.slider("Age", 18, 90, 38)
products = st.sidebar.slider("Number of Products", 1, 4, 2)
is_active = st.sidebar.selectbox("Is Active Member?", options=["No (0)", "Yes (1)"], index=1)
active_val = 0 if is_active == "No (0)" else 1

ratio = st.sidebar.slider("Balance-to-Salary Ratio", 0.0, 15.0, 3.8)
density = st.sidebar.slider("Product Density", 0.0, 5.0, 0.4)
gender = st.sidebar.selectbox("Gender", options=["Female", "Male"])
gender_val = 0 if gender == "Female" else 1

input_data = pd.DataFrame([[age, products, active_val, ratio, density, gender_val]], 
                          columns=['Age', 'NumOfProducts', 'IsActiveMember', 'BalanceSalaryRatio', 'ProductDensity', 'Gender'])

# 4. Predictions
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]
churn_prob = probabilities[1]
stay_prob = probabilities[0]

# 5. Dashboard Layout (Visualizations)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Churn Risk Calculator")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Churn!**")
    else:
        st.success(f"✅ **Safe / Loyal Customer.**")
    
    # Probability Distribution Visualization
    st.markdown("#### Probability Distribution")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['Churn (Leave)', 'Retained (Stay)'], [churn_prob, stay_prob], color=['#ff4b4b', '#21c354'])
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    for i, v in enumerate([churn_prob, stay_prob]):
        ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center', fontweight='bold')
    
    st.pyplot(fig)

with col2:
    # Feature Importance Dashboard
    st.subheader("Feature Importance Dashboard")
    st.markdown("Shows which factors drive the model's decisions globally.")
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color='#3b82f6')
    ax2.set_xlabel('Relative Importance')
    st.pyplot(fig2)