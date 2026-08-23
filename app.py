import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

model = joblib.load('models/churn_model_v1.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction + Monitoring")

tab1, tab2 = st.tabs(["Predict", "Monitoring Dashboard"])

with tab1:
    st.header("Predict Customer Churn")
    st.sidebar.header("Enter Customer Details")

    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18, 120, 65)
    total_charges = monthly_charges * tenure

    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    payment = st.sidebar.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    if st.sidebar.button("Predict"):
        input_dict = {
            'gender': 'Male', 'SeniorCitizen': senior, 'Partner': partner,
            'Dependents': dependents, 'tenure': tenure, 'PhoneService': 'Yes',
            'MultipleLines': 'No', 'InternetService': internet,
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
            'TechSupport': tech_support, 'StreamingTV': 'No', 'StreamingMovies': 'No',
            'Contract': contract, 'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }
        input_df = pd.DataFrame([input_dict])

        service_cols = ['PhoneService','OnlineSecurity','OnlineBackup',
                         'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
        input_df['total_services'] = (input_df[service_cols] == 'Yes').sum(axis=1)
        input_df['charge_per_tenure'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
        input_df['tenure_group'] = pd.cut(input_df['tenure'], bins=[0,12,24,48,72],
                                           labels=['0-1yr','1-2yr','2-4yr','4-6yr'])
        input_df['household_type'] = input_df['Partner'] + '_' + input_df['Dependents']

        input_encoded = pd.get_dummies(input_df).reindex(columns=feature_columns, fill_value=0).astype(int)

        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

        prob = model.predict_proba(input_encoded)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with col2:
            if prob > 0.5:
                st.error("⚠️ High Risk - Likely to Churn")
            else:
                st.success("✅ Low Risk - Likely to Stay")

        if prob > 0.7:
            st.warning("Recommended Action: Offer immediate retention discount")
        elif prob > 0.5:
            st.warning("Recommended Action: Schedule customer success call")
        else:
            st.info("Recommended Action: Standard engagement")

        st.subheader("Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_encoded)

        fig, ax = plt.subplots(figsize=(10,5))
        shap.waterfall_plot(shap.Explanation(
            values=shap_vals[0], base_values=explainer.expected_value,
            data=input_encoded.iloc[0], feature_names=feature_columns), show=False)
        st.pyplot(fig)
        plt.close()

        os.makedirs('data', exist_ok=True)
        conn = sqlite3.connect('data/predictions.db')
        conn.execute('''CREATE TABLE IF NOT EXISTS predictions_log
                         (timestamp TEXT, tenure REAL, monthly_charges REAL,
                          total_charges REAL, contract TEXT, prediction REAL)''')
        conn.execute("INSERT INTO predictions_log VALUES (?,?,?,?,?,?)",
                      (str(datetime.now()), tenure, monthly_charges, total_charges, contract, prob))
        conn.commit()
        conn.close()

with tab2:
    st.header("Model Monitoring Dashboard")
    try:
        ref = pd.read_csv('data/processed/reference_data.csv')
        conn = sqlite3.connect('data/predictions.db')
        logs = pd.read_sql('SELECT * FROM predictions_log', conn)
        conn.close()

        if len(logs) >= 5:
            from scipy import stats
            st.subheader("Data Drift Detection")
            results = []
            for col in ['tenure','monthly_charges','total_charges']:
                if col in ref.columns and col in logs.columns:
                    ks, p = stats.ks_2samp(ref[col], logs[col])
                    results.append({'Feature': col, 'KS Statistic': round(ks,3),
                                     'P-value': round(p,3),
                                     'Status': "⚠️ Drift" if p < 0.05 else "✅ Stable"})
            st.dataframe(pd.DataFrame(results))

            st.subheader("Churn Risk Trend Over Time")
            logs['timestamp'] = pd.to_datetime(logs['timestamp'])
            st.line_chart(logs.set_index('timestamp')['prediction'])

            st.subheader("Prediction Distribution")
            fig2, ax2 = plt.subplots()
            logs['prediction'].hist(bins=20, ax=ax2, color='coral')
            st.pyplot(fig2)
            plt.close()
        else:
            st.info("Make at least 5 predictions to see monitoring data")
    except Exception as e:
        st.info("No predictions logged yet — go to Predict tab first")