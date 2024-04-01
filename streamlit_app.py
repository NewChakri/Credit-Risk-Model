import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('best_model.pkl')

# Function to preprocess input data
def preprocess_input(input_data):
    label_encoders = {}
    for col in input_data.select_dtypes(include=['object']).columns:
        label_encoders[col] = LabelEncoder()
        input_data[col] = input_data[col].map(mapping[col]).astype(int)
    return input_data

# Define the mapping dictionary
mapping = {
    "Sex": {"female": 0, "male": 1},
    "Housing": {"free": 0, "own": 1, "rent": 2},
    "Saving accounts": {"little": 0, "moderate": 1, "no_inf": 2, "quite rich": 3, "rich": 4},
    "Checking account": {"little": 0, "moderate": 1, "no_inf": 2, "rich": 3},
    "Purpose": {"business": 0, "car": 1, "domestic appliances": 2, "education": 3,
                "furniture/equipment": 4, "radio/TV": 5, "repairs": 6, "vacation/others": 7}
}

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
    sex = st.sidebar.radio('Sex', ['female', 'male'])
    job = st.sidebar.selectbox('Job', ['Unskilled and non-resident', 'Unskilled and resident', 'Skilled', 'Highly skilled'])
    housing = st.sidebar.selectbox('Housing', ['free', 'own', 'rent'])
    saving_acct = st.sidebar.selectbox('Saving accounts', ['little', 'moderate', 'no_inf', 'quite rich', 'rich'])
    checking_acct = st.sidebar.selectbox('Checking account', ['little', 'moderate', 'no_inf', 'rich'])
    credit_amount = st.sidebar.number_input('Credit amount', min_value=0, step=1)
    duration = st.sidebar.number_input('Duration (months)', min_value=0, step=1)
    purpose = st.sidebar.selectbox('Purpose', ['business', 'car', 'domestic appliances', 'education',
                                               'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'])

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
        if prediction[0] == 0:
            st.error('High Risk')
        else:
            st.success('Low Risk')

if __name__ == '__main__':
    main()
