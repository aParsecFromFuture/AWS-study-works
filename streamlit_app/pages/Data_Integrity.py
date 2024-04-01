import streamlit as st
from utils import check_utils  # Importing utility functions from a module named 'check_utils'
import pandas as pd


def main():
    # Sidebar file uploaders and select box for user input
    train_source_data = st.sidebar.file_uploader("Upload Train Data", type=["csv"])
    test_source_data = st.sidebar.file_uploader("Upload Test Data", type=["csv"])
    report_type = st.sidebar.selectbox("Please select report type", check_utils.REPORT_TYPE)
    activate = st.sidebar.button("Check")  # Button to initiate the check

    if activate:  # Checking if the 'Check' button is activated
        if (train_source_data is None) | (test_source_data is None):  # Checking if any dataset is uploaded
            st.error("Please upload your datasets to get started")  # Error message if datasets are missing
        else:
            # Reading uploaded CSV files into pandas DataFrames
            df_train = pd.read_csv(train_source_data)
            df_test = pd.read_csv(test_source_data)
            result = None

            # Depending on the selected report type, call corresponding utility function
            if report_type == "Is Single Value":
                result = check_utils.is_single_value(df_train)
            if report_type == "Percent Of Nulls":
                result = check_utils.percent_of_nulls(df_train)
            if report_type == "Feature Label Correlation":
                result = check_utils.feature_label_correlation(df_train)
            if report_type == "Mixed Data Types":
                result = check_utils.mixed_data_types(df_train)
            if report_type == "Class Imbalance":
                result = check_utils.class_imbalance(df_train)
            if report_type == "Feature Drift":
                result = check_utils.feature_drift(df_train, df_test)
            if report_type == "Multivariate Drift":
                result = check_utils.multivariate_drift(df_train, df_test)
            if report_type == "New Category Train Test":
                result = check_utils.new_category_train_test(df_train, df_test)
            if report_type == "Train Test Samples Mix":
                result = check_utils.train_test_samples_mix(df_train, df_test)

            # Displaying the result if it's not None
            if result is not None:
                st.subheader(result["header"])  # Displaying a subheader with result header
                st.markdown(result["check"]["summary"], unsafe_allow_html=True)  # Displaying summary markdown
                st.json(result['conditions_results'][0])  # Displaying conditions results in JSON format


if __name__ == "__main__":
    # Setting up Streamlit page configuration
    st.set_page_config(page_title="Deepcheck Streamlit Application", layout="wide")
    st.header("Deepchecks - Train Test Validation")  # Main header
    main()  # Calling the main function to run the application
