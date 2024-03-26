import streamlit as st
import pandas as pd
import joblib
import json
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('best_model.pkl')

# Load the label encoding mappings
with open('label_encoding_mappings.json', 'r') as file:
    encoding_mappings = json.load(file)

# Function to preprocess input data
def preprocess_input(input_data, encoding_mappings):
    # Convert categorical columns to numerical using LabelEncoder and mapping
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        input_data[col] = input_data[col].map(encoding_mappings[col])
        input_data[col] = input_data[col].astype(int)
    
    return input_data

# Function to predict
def predict(input_data):
    preprocessed_data = preprocess_input(input_data, encoding_mappings)
    nan_columns = preprocessed_data.columns[preprocessed_data.isnull().any()]
    if not nan_columns.empty:
        st.error(f'Input data contains NaN values in columns: {", ".join(nan_columns)}. Please provide valid input.')
        return None
    prediction = model.predict(preprocessed_data)
    return prediction

# Create the web app interface using Streamlit
def main():
    # Set page title and favicon
    st.set_page_config(page_title='Credit Risk Prediction', page_icon='📊')
    
    # Add a title and description
    st.title('Credit Risk Prediction')
    st.markdown('This app predicts the credit risk based on user input.')

    # Create sidebar for input fields
    st.sidebar.title('Enter Customer Details')
    age = st.sidebar.slider('Age', min_value=18, max_value=100, step=1)
    sex = st.sidebar.radio('Sex', ['Male', 'Female'])
    job = st.sidebar.selectbox('Job', ['Unskilled and non-resident', 'Unskilled and resident', 'Skilled', 'Highly skilled'])
    housing = st.sidebar.selectbox('Housing', ['Own', 'Rent', 'Free'])
    saving_acct = st.sidebar.selectbox('Saving accounts', ['Little', 'Moderate', 'Quite rich', 'Rich', 'No information'])
    checking_acct = st.sidebar.selectbox('Checking account', ['Little', 'Moderate', 'Rich', 'No information'])
    credit_amount = st.sidebar.number_input('Credit amount', min_value=0, step=1)
    duration = st.sidebar.number_input('Duration (months)', min_value=0, step=1)
    purpose = st.sidebar.selectbox('Purpose', ['Car', 'Radio/TV', 'Education', 'Furniture/Equipment', 'Business', 'Domestic appliances', 'Repairs', 'Vacation/Others'])

    # Convert job selection to numeric
    job_mapping = {'Unskilled and non-resident': 0, 'Unskilled and resident': 1, 'Skilled': 2, 'Highly skilled': 3}
    job_numeric = job_mapping[job]

    # Create a dictionary to store user input data
    input_data = {
        'Age': [age],
        'Sex': [sex],
        'Job': [job_numeric],
        'Housing': [housing],
        'Saving accounts': [saving_acct],
        'Checking account': [checking_acct],
        'Credit amount': [credit_amount],
        'Duration': [duration],
        'Purpose': [purpose]
    }
    st.write('Data Types of Input Fields:')
    st.write(f'Age: {type(age).__name__}')
    st.write(f'Sex: {type(sex).__name__}')
    st.write(f'Job: {type(job).__name__}')
    st.write(f'Housing: {type(housing).__name__}')
    st.write(f'Saving accounts: {type(saving_acct).__name__}')
    st.write(f'Checking account: {type(checking_acct).__name__}')
    st.write(f'Credit amount: {type(credit_amount).__name__}')
    st.write(f'Duration: {type(duration).__name__}')
    st.write(f'Purpose: {type(purpose).__name__}')

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    if st.sidebar.button('Predict'):
        # Get prediction
        prediction = predict(input_df)

        # Display prediction result
        if prediction[0] == 0:
            st.error('High Risk')
        else:
            st.success('Low Risk')

if __name__ == '__main__':
    main()
