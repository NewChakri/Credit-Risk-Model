import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('best_model.pkl')

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert categorical columns to numerical using LabelEncoder
    label_encoders = {}
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        input_data[col] = label_encoders[col].fit_transform(input_data[col])
    
    return input_data

# Function to predict
def predict(input_data):
    preprocessed_data = preprocess_input(input_data)
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
