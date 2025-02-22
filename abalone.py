# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

# ---------------------------
# Data Loading and Pre-processing
# ---------------------------
df = pd.read_csv('abalone.csv')
df['Age'] = df['Rings'] + 1.5  # target defined as Rings + 1.5 (per documentation :contentReference[oaicite:0]{index=0})
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
X = df.drop(['Age', 'Rings'], axis=1)
y = df['Age']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# ---------------------------
# Evaluation Function
# ---------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test, plot=True):
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
    
    if plot:
        plt.figure(figsize=(6,4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Age")
        plt.ylabel("Predicted Age")
        plt.title(f"{name}: Actual vs Predicted Age")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.show()
    return {'MAE': mae, 'MSE': mse, 'R2': r2}

# ---------------------------
# Self-Optimization Function using GridSearchCV
# ---------------------------
def optimize_model(model, param_grid, X_train, y_train, model_name="Model"):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Optimizing {model_name}...")
    print("Best parameters found:", grid.best_params_)
    print("Best CV MSE:", -grid.best_score_)
    return grid.best_estimator_

# ---------------------------
# Model 1: Linear Regression (Baseline)
# ---------------------------
lin_reg = LinearRegression()
metrics_lin = evaluate_model("Linear Regression", lin_reg, X_train, X_test, y_train, y_test)

# ---------------------------
# Model 2: XGBoost Regressor with Expanded Hyperparameter Grid
# ---------------------------
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
best_xgb = optimize_model(xgb, xgb_param_grid, X_train, y_train, model_name="XGBoost Regressor")

# Visualize training curves for XGBoost by splitting a validation set
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
best_xgb_eval = XGBRegressor(**best_xgb.get_params())
eval_set = [(X_train_sub, y_train_sub), (X_val, y_val)]
try:
    best_xgb_eval.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)
except TypeError:
    print("Early stopping is not supported in your version of XGBoost. Fitting without it.")
    best_xgb_eval.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)
evals_result = best_xgb_eval.evals_result()

if 'validation_0' in evals_result and 'rmse' in evals_result['validation_0']:
    plt.figure(figsize=(8,5))
    epochs = len(evals_result['validation_0']['rmse'])
    x_axis = range(epochs)
    plt.plot(x_axis, evals_result['validation_0']['rmse'], label='Train')
    plt.plot(x_axis, evals_result['validation_1']['rmse'], label='Validation')
    plt.xlabel('Boosting Round')
    plt.ylabel('RMSE')
    plt.title('XGBoost Training vs Validation RMSE')
    plt.legend()
    plt.show()
else:
    print("No evaluation results available to plot.")

metrics_xgb = evaluate_model("XGBoost Regressor (Optimized)", best_xgb, X_train, X_test, y_train, y_test)

# ---------------------------
# Model 3: MLP Regressor with Expanded Hyperparameter Grid
# ---------------------------
mlp = MLPRegressor(max_iter=500, random_state=42, early_stopping=True)
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50), (100,100,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'learning_rate_init': [0.001, 0.005, 0.01]
}
best_mlp = optimize_model(mlp, mlp_param_grid, X_train, y_train, model_name="MLP Regressor")

# Fit to capture the loss curve and then plot it
_ = best_mlp.fit(X_train, y_train)
plt.figure(figsize=(8,5))
plt.plot(best_mlp.loss_curve_, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP Regressor Loss Curve")
plt.show()

metrics_mlp = evaluate_model("MLP Regressor (Optimized)", best_mlp, X_train, X_test, y_train, y_test)

# ---------------------------
# Comparison of Model Performance
# ---------------------------
performance_df = pd.DataFrame({
    'Model': ['Linear Regression', 'XGBoost', 'MLP'],
    'MAE': [metrics_lin['MAE'], metrics_xgb['MAE'], metrics_mlp['MAE']],
    'MSE': [metrics_lin['MSE'], metrics_xgb['MSE'], metrics_mlp['MSE']],
    'R2':  [metrics_lin['R2'],  metrics_xgb['R2'],  metrics_mlp['R2']]
})

print("\nSummary of Model Performance:")
print(performance_df)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(x='Model', y='MAE', data=performance_df, ax=ax[0])
ax[0].set_title("Mean Absolute Error")
sns.barplot(x='Model', y='MSE', data=performance_df, ax=ax[1])
ax[1].set_title("Mean Squared Error")
sns.barplot(x='Model', y='R2', data=performance_df, ax=ax[2])
ax[2].set_title("R^2 Score")
plt.tight_layout()
plt.show()
