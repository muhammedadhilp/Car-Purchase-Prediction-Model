# Car Purchase Prediction Model
This project aims to predict whether a customer will purchase a car based on their demographic features (gender, age, annual salary) using a logistic regression model.

## Project Structure
The project follows the following steps:

Data Loading
Exploratory Data Analysis (EDA)
Data Preprocessing
Model Training
Evaluation
## Files
car_data.csv: Dataset containing customer demographic information and whether they purchased a car or not.
README.md: This file.
## Prerequisites
The project uses the following Python libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

## Steps
### 1. Data Loading
#### code:
df = pd.read_csv('car_data.csv')
The dataset consists of five columns:

UserID: Unique identifier for each user (dropped as it's not relevant).
Gender: Gender of the customer (requires encoding).
Age: Age of the customer.
AnnualSalary: Annual salary of the customer.
Purchased: Whether the customer purchased a car (target variable).
### 2. Exploratory Data Analysis (EDA)
Shape and Description
#### code:
print('Shape of the data:', df.shape)
df.describe()
df.info()
Observations:

No null or duplicate values.
Age ranges from 18 to 63 years.
Annual salary ranges from $15,000 to $152,500.
Data Visualizations
Continuous columns (Age and AnnualSalary) were visualized using distribution plots, box plots, and histograms. Some key insights:

Age: People under 45 are more likely to buy cars.
Annual Salary: Salaries between $40,000 and $85,000 show lower purchasing rates.
Gender distribution and class imbalance were also visualized. There was a slight class imbalance (non-purchases lead by 19.6%).
### 3. Data Preprocessing
Categorical Encoding and Transformation
The Gender column is one-hot encoded, and the Age and AnnualSalary columns are transformed using the Yeo-Johnson transformation to handle skewness.
#### code:
transformer = ColumnTransformer(transformers=[
    ('Encoder', OneHotEncoder(drop='first', sparse=False), ['Gender']),
    ('Yeo-Johnson', PowerTransformer(), ['Age', 'AnnualSalary'])
])
Train-Test Split
The data is split into training and testing sets with stratification to maintain the distribution of the target variable.
#### code:
X_train, X_test, y_train, y_test = train_test_split(df.drop(['Purchased'], axis=1), df['Purchased'], random_state=42, stratify=df['Purchased'])
### 4. Model Training
We used logistic regression and performed hyperparameter tuning using RandomizedSearchCV to find the optimal parameters for the model.
#### code:
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [50, 75, 100, 200, 300, 400, 500, 700]}
log = RandomizedSearchCV(LogisticRegression(solver='lbfgs'), param_grid, cv=5)
log.fit(X_train, y_train)
### 5. Evaluation
The model achieved an accuracy of 84%. The confusion matrix and classification report provide further details on model performance.
#### code:
y_pred_log = log.predict(X_test)
confusion_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(confusion_log, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test, y_pred_log))

## Observations and Insights
Age: Younger individuals (under 45) are more likely to purchase a car.
Annual Salary: Salaries between $40,000 and $85,000 show a lower tendency to purchase cars.
Gender: There are no significant differences between genders regarding car purchases.
Model Performance: The logistic regression model provides 84% accuracy, with no significant class imbalance issues.
