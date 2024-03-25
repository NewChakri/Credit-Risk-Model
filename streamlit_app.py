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
    st.title('Credit Risk Prediction')

    # Add input fields for user input
    age = st.slider('Age', min_value=18, max_value=100, step=1)
    sex = st.selectbox('Sex', ['male', 'female'])
    job = st.slider('Job', min_value=0, max_value=3, step=1)
    housing = st.selectbox('Housing', ['own', 'rent', 'free'])
    saving_acct = st.selectbox('Saving accounts', ['little', 'moderate', 'quite rich', 'rich', 'no_inf'])
    checking_acct = st.selectbox('Checking account', ['little', 'moderate', 'rich', 'no_inf'])
    credit_amount = st.slider('Credit amount', min_value=0, max_value=100000, step=100)
    duration = st.slider('Duration (months)', min_value=0, max_value=100, step=1)
    purpose = st.selectbox('Purpose', ['car', 'radio/TV', 'education', 'furniture/equipment', 'business', 'domestic appliances', 'repairs', 'vacation/others'])

    # Create a dictionary to store user input data
    input_data = {
        'Age': [age],
        'Sex': [sex],
        'Job': [job],
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
        # Get prediction
        prediction = predict(input_df)

        # Display prediction result
        if prediction[0] == 0:
            st.error('High Risk')
        else:
            st.success('Low Risk')

if __name__ == '__main__':
    main()
