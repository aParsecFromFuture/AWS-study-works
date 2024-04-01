import streamlit as st
import pandas as pd
import os
import json
import requests


def main():
    # --- Sidebar ---
    source_data = st.sidebar.file_uploader("Predict File", type=["csv"])  # Allow CSV file upload
    activate = st.sidebar.button("Run")  # Button to trigger model training

    # --- Prediction Form ---
    with st.form("form_widget"):
        st.subheader("Prediction Form")

        # --- User Input Fields ---
        gender_col, married_col = st.columns(2)  # Create two columns for layout
        gender_value = gender_col.selectbox("Gender", options=["Male", "Female"])  # Selectbox for gender
        married_value = married_col.selectbox("Married", options=["Yes", "No"])  # Selectbox for marital status

        dependents_col, education_col = st.columns(2)
        dependents_value = dependents_col.selectbox("Dependents", options=["0", "1", "2", "3+"])  # Selectbox for dependents
        education_value = education_col.selectbox("Education", options=["Graduate", "Not Graduate"])  # Selectbox for education

        self_employed_col, property_area_col = st.columns(2)
        self_employed_value = self_employed_col.selectbox("Self Employed", options=["Yes", "No"])  # Selectbox for self-employment
        property_area_value = property_area_col.selectbox("Property Area", options=["Semiurban", "Urban", "Rural"])  # Selectbox for property area

        application_income_col, co_application_income_col = st.columns(2)
        application_income_value = application_income_col.number_input("Application Income")  # Number input for income
        co_application_income_value = co_application_income_col.number_input("Co Application Income")  # Number input for co-applicant income

        loan_amount_col, loan_amount_term_col = st.columns(2)
        loan_amount_value = loan_amount_col.number_input("Loan Amount")  # Number input for loan amount
        loan_amount_term_value = loan_amount_term_col.number_input("Loan Amount Term")  # Number input for loan term

        credit_history_col, _ = st.columns(2)
        credit_history_value = credit_history_col.selectbox("Credit History", options=["0", "1"])  # Selectbox for credit history

        submitted = st.form_submit_button("Submit")  # Submit button to trigger prediction

        if submitted:
            # --- Prepare Data for Prediction ---
            request_body = {
                "data": "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}".format(
                    gender_value,
                    married_value,
                    dependents_value,
                    education_value,
                    self_employed_value,
                    application_income_value,
                    co_application_income_value,
                    loan_amount_value,
                    loan_amount_term_value,
                    credit_history_value,
                    property_area_value
                )
            }
            request_body = json.dumps(request_body)  # Convert data to JSON format

            # --- Send Prediction Request ---
            response = requests.post(os.environ["PREDICTION_URL"], data=request_body)  # Send POST request
            prediction = json.loads(response.content.decode())["prediction"]  # Convert response to prediction
            label = "Positive" if prediction > 0.5 else "Negative"

            st.write("Loan Status: ", label)  # Display predicted loan status

    # --- File Prediction Request ---
    if activate:
        if source_data is None:
            st.error("Please upload your source data to get started")
        else:
            ds = pd.read_csv(source_data)  # Read uploaded data
            ds = ds.to_csv(index=False, header=False).encode("utf-8")

            request_body = {"data": ds}
            response = requests.post(os.environ["PREDICTION_URL"], request_body)  # Send POST request


if __name__ == "__main__":
    st.set_page_config(page_title="Deepcheck Streamlit Application", layout="wide")
    main()
