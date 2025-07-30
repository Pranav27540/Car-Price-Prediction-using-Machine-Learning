# Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn import metrics

# Load dataset
dataset = pd.read_csv("car data.csv")

# Display first few rows
print(dataset.head())

# Preprocessing
dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
                 'Seller_Type': {'Dealer': 0, 'Individual': 1},
                 'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Creating new column 'Current Year' to calculate car age
dataset['Current_Year'] = 2020
dataset['Age'] = dataset['Current_Year'] - dataset['Year']
dataset.drop(['Year', 'Current_Year', 'Car_Name'], axis=1, inplace=True)

# Features and label
X = dataset.drop(['Selling_Price'], axis=1)
y = dataset['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Model 1: Linear Regression
# ------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinear Regression R2 Score:", metrics.r2_score(y_test, y_pred_lr))

# ------------------------------
# Model 2: Lasso Regression
# ------------------------------
lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso Regression R2 Score:", metrics.r2_score(y_test, y_pred_lasso))

# ------------------------------
# Model 3: Ridge Regression
# ------------------------------
ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge Regression R2 Score:", metrics.r2_score(y_test, y_pred_ridge))

# ------------------------------
# Model 4: Random Forest Regressor
# ------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Regressor R2 Score:", metrics.r2_score(y_test, y_pred_rf))

# Visualizing Feature Importance
importances = rf.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.show()

# Save model (Optional)
import joblib
joblib.dump(rf, "car_price_model.pkl")
