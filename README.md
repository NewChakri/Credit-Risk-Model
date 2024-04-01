# Credit Risk Prediction App Explanation

This repository contains code for a credit risk prediction application. The application uses machine learning models to predict whether a customer is at high or low risk of credit default based on their input data.

## Code Explanation

### 1. Data Preprocessing and Analysis

The code starts by importing necessary libraries and loading the dataset (`german_credit_data.csv`). It then performs data preprocessing steps such as handling missing values, encoding categorical variables using LabelEncoder, and balancing the classes using SMOTE.

### 2. Model Training and Evaluation

Next, the code defines hyperparameter grids for various machine learning models such as Logistic Regression, Random Forest, SVM, Gradient Boosting, LightGBM, XGBoost, and CatBoost. It initializes these models and trains them using GridSearchCV for hyperparameter tuning. The best-performing model is saved as `best_model.pkl`.

### 3. Streamlit Web App

The second part of the code is a Streamlit web application (`streamlit_app.py`) that loads the saved best model and allows users to input customer details. It preprocesses the input data, predicts the credit risk using the loaded model, and displays the prediction result (high or low risk) with appropriate styling and emojis.

## How to Use

To run the Streamlit app locally:
1. Install the required libraries (`pip install -r requirements.txt`).
2. Run the Streamlit app (`streamlit run streamlit_app.py`).
3. Enter customer details in the sidebar and click "Predict" to see the risk prediction.

## Files Included

- `german_credit_data.csv`: Dataset used for training and testing.
- `best_model.pkl`: Saved best-performing machine learning model.
- `streamlit_app.py`: Streamlit web application code for credit risk prediction.

Feel free to explore the code and experiment with different models or input data to understand how the credit risk prediction works.
