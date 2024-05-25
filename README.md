# Credit Risk Prediction

The project credit risk prediction application developed using Python and machine learning classification models. The application's primary objective is to assess whether a customer is at high or low risk of credit default based on their input data. It is deployed using Streamlit, providing a user-friendly and interactive platform for credit risk analysis.

Web App : https://newchakri-credit-risk-model.streamlit.app

![image](https://github.com/NewChakri/Credit-Risk-Model/assets/99199609/8d866f05-8139-4784-81c8-96be27a6f9c1)



## Data

The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes.

- `Age` (numeric)
- `Sex` (text: male, female)
- `Job` (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
- `Housing` (text: own, rent, or free)
- `Saving accounts` (text - little, moderate, quite rich, rich)
- `Checking account` (numeric, in DM - Deutsch Mark)
- `Credit amount` (numeric, in DM)
- `Duration` (numeric, in month)
- `Purpose` (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

## Code Explanation

### 1. Data Preprocessing and Analysis

The code starts by importing necessary libraries and loading the dataset (`german_credit_data.csv`). It then performs data preprocessing steps such as handling missing values, encoding categorical variables using LabelEncoder, and balancing the classes using SMOTE.

### 2. Model Training and Evaluation

Next, the code defines hyperparameter grids for various machine learning models, including Logistic Regression, Random Forest, SVM, Gradient Boosting, LightGBM, XGBoost, and CatBoost. It initializes these models and trains them using GridSearchCV for hyperparameter tuning. After evaluating the models based on metrics such as accuracy, precision, recall, and F1-score, the best-performing model, LightGBM with the hyperparameters {'learning_rate': 0.1, 'num_leaves': 50}, is selected and saved as best_model.pkl. This model demonstrated an accuracy of 79.64%, a precision of 79.56%, a recall of 79.63%, and an F1-score of 79.58%, along with a relatively low training time of 2.65 seconds, making it an efficient choice for credit risk prediction.

### 3. Streamlit Web App

The second part of the code is a Streamlit web application (`streamlit_app.py`) that loads the saved best model and allows users to input customer details. It preprocesses the input data, predicts the credit risk using the loaded model, and displays the prediction result (high or low risk) with appropriate styling.


## Files Included
- `german_credit_data.csv` : Dataset used for training and testing.
- `Credit-Risk-Model.py` : Python script containing code for data preprocessing, model training, and evaluation.
- `best_model.pkl` : Saved best-performing machine learning model.
- `streamlit_app.py` : Streamlit web application code for credit risk prediction.
- `model_evaluation_results.csv` : CSV file containing the evaluation results of different models, including accuracy, precision, recall, F1-score, and training time.
