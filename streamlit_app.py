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

    # Add input fields for user input
    age = st.slider('Age', min_value=18, max_value=100, step=1)
    sex = st.selectbox('Sex', ['male', 'female'])
    job_options = ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled']
    job = st.selectbox('Job', job_options)
    housing = st.selectbox('Housing', ['own', 'rent', 'free'])
    saving_acct = st.selectbox('Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'no_inf'])
    checking_acct = st.selectbox('Checking account', ['little', 'moderate', 'rich', 'no_inf'])
    
    # Freeform text input for Credit amount and Duration
    credit_amount = st.text_input('Credit amount', '')
    duration = st.text_input('Duration (months)', '')
    
    purpose = st.selectbox('Purpose', ['car', 'radio/TV', 'education', 'furniture/equipment', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

    # Convert job selection to numeric
    job_mapping = {'unskilled and non-resident': 0, 'unskilled and resident': 1, 'skilled': 2, 'highly skilled': 3}
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

    if st.button('Predict'):
        # Check if Credit amount and Duration are entered
        if credit_amount and duration:
            # Convert Credit amount and Duration to numeric
            input_df['Credit amount'] = pd.to_numeric(input_df['Credit amount'])
            input_df['Duration'] = pd.to_numeric(input_df['Duration'])
            
            # Get prediction
            prediction = predict(input_df)

            # Display prediction result
            if prediction[0] == 0:
                st.error('High Risk')
            else:
                st.success('Low Risk')
        else:
            st.warning('Please enter Credit amount and Duration.')

if __name__ == '__main__':
    main()
