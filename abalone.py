# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost (make sure to install it via: pip install xgboost)
from xgboost import XGBRegressor

# ---------------------------
# Data Loading and Pre-processing
# ---------------------------
# Read the abalone dataset (ensure abalone.csv is in your working directory)
df = pd.read_csv('abalone.csv')

# According to the dataset description (&#8203;:contentReference[oaicite:2]{index=2}), the target age is Rings + 1.5
df['Age'] = df['Rings'] + 1.5

# Encode the categorical 'Sex' column using LabelEncoder (see :contentReference[oaicite:3]{index=3})
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Optionally, you can drop any unneeded columns (for example, if you want to drop 'Rings' now)
# If you want to predict the actual age, retain all features.
# For this implementation, we use all features except the original Rings.
X = df.drop(['Age', 'Rings'], axis=1)
y = df['Age']

# Scale the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (40% test size as used in the original workflow :contentReference[oaicite:4]{index=4})
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# ---------------------------
# Model Training and Evaluation
# ---------------------------
# Define a function to evaluate models
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    
    print(f"{name} Performance:")
    print(f"  Mean Absolute Error: {mae:.3f}")
    print(f"  Mean Squared Error:  {mse:.3f}")
    print(f"  R^2 Score:           {r2:.3f}")
    print("-" * 40)
    
    # Optionally, plot actual vs predicted values
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title(f"{name}: Actual vs Predicted Age")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

# 1. Linear Regression Model
lin_reg = LinearRegression()
evaluate_model("Linear Regression", lin_reg, X_train, X_test, y_train, y_test)

# 2. XGBoost Regressor
# Using default parameters; you can tune these parameters as needed.
xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)
evaluate_model("XGBoost Regressor", xgb_reg, X_train, X_test, y_train, y_test)

# 3. MLP Regressor (Multi-Layer Perceptron)
# Here we set a hidden layer structure and early stopping for better convergence.
mlp_reg = MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                       max_iter=500, random_state=42, early_stopping=True)
evaluate_model("MLP Regressor", mlp_reg, X_train, X_test, y_train, y_test)
