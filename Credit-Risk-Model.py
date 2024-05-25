import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import time
import joblib

############## Read data ##############
df = pd.read_csv('german_credit_data.csv')

############## Data Exploration and Visualization ##############
grouped_data = df.groupby('Risk')
columns_to_plot = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
fig, axes = plt.subplots(nrows=len(columns_to_plot), ncols=1, figsize=(10, 8 * len(columns_to_plot)))
for idx, col in enumerate(columns_to_plot):
    if df[col].dtype == 'object':
        sns.countplot(x=col, hue='Risk', data=df, ax=axes[idx])
    else:
        sns.histplot(x=col, hue='Risk', data=df, ax=axes[idx], bins=20)
    axes[idx].set_title(f'{col} by Risk')
plt.tight_layout()
plt.show()

############## Data Preprocessing ##############
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df['Saving accounts'] = df['Saving accounts'].fillna('no_inf')
df['Checking account'] = df['Checking account'].fillna('no_inf')
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

############## Model Training and Evaluation ##############
X = df.drop('Risk', axis=1)
y = df['Risk']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define hyperparameter grids for each model
param_grid = {
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
    'Random Forest': {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 20]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Gradient Boosting': {'learning_rate': [0.1, 0.5, 1], 'n_estimators': [50, 100, 200]},
    'LightGBM': {'num_leaves': [31, 50, 100], 'learning_rate': [0.1, 0.01]},
    'XGBoost': {'max_depth': [3, 5, 7], 'learning_rate': [0.1, 0.01]},
    'CatBoost': {'iterations': [100, 200, 300], 'learning_rate': [0.1, 0.01]}
}

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'LightGBM': LGBMClassifier(),
    'XGBoost': XGBClassifier(),
    'CatBoost': CatBoostClassifier(silent=True)  # Set silent=True to suppress CatBoost output
}

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Best Parameters', 'Accuracy', 'Precision', 'Recall', 'F1-score','Training Time'])
best_model = None
best_f1_score = 0
lowest_training_time = float('inf')

# Train and evaluate models with hyperparameter tuning
for name, model in models.items():
    start_time = time.time()
    
    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=5)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_
    
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate average metrics across classes (0 and 1)
    precision = (classification_rep['0']['precision'] + classification_rep['1']['precision']) / 2
    recall = (classification_rep['0']['recall'] + classification_rep['1']['recall']) / 2
    f1_score = (classification_rep['0']['f1-score'] + classification_rep['1']['f1-score']) / 2
    
    # Append results to DataFrame
    results_df = results_df.append({'Model': name, 'Best Parameters': best_params, 'Accuracy': accuracy, 'Training Time': training_time,
                                    'Precision': precision, 'Recall': recall, 'F1-score': f1_score}, ignore_index=True)
    
    # Check if the current model has a higher F1-score and lower training time than the best model
    if f1_score > best_f1_score or (f1_score == best_f1_score and training_time < lowest_training_time):
        best_model = model
        best_f1_score = f1_score
        lowest_training_time = training_time
    

############## Export the best model ##############
if best_model is not None:
    joblib.dump(best_model, 'best_model.pkl')
    
############## Export Evaluation results ##############
results_df.to_csv('model_evaluation_results.csv', index=False)
