## 1. Introduction
This project focuses on predicting the age of abalones using various machine learning techniques. The dataset used contains different physical measurements of abalones, and the target age is computed as the number of shell rings plus 1.5 years. We explore **Linear Regression**, **XGBoost**, and an **MLP Regressor (Neural Network)** and compare their performances.

---

## 2. Data Overview and Preprocessing
1. **Dataset**:  
   - Features: `Sex` (categorical), `Length`, `Diameter`, `Height`, `Whole Weight`, `Shucked Weight`, `Viscera Weight`, `Shell Weight`, and `Rings` (the original age indicator).  
   - Target (`Age`): Defined as `Rings + 1.5`.

2. **Label Encoding**: The `Sex` column is transformed into numerical values using `LabelEncoder`.

3. **Scaling & Splitting**:
   - We apply `StandardScaler` to the feature matrix to ensure that all features contribute on a similar scale.
   - We split the data into training and testing sets with a 60/40 ratio.

---

## 3. Model Implementations

### 3.1 Linear Regression (Baseline)
A **Linear Regression** model is used as a baseline. It provides a quick view of how a simple linear model predicts abalone age.

### 3.2 XGBoost Regressor
**XGBoost** is a gradient boosting framework known for strong performance. We use `GridSearchCV` over a range of hyperparameters (learning rate, max depth, number of estimators) to select the best configuration.

### 3.3 MLP Regressor (Neural Network)
We also test a **Multilayer Perceptron (MLP)** Regressor with multiple hidden layer configurations, activation functions, and learning rates. A grid search is employed to identify the best settings, optimizing for the lowest mean squared error (MSE).

---

## 4. Evaluation Metrics
Each model’s predictions on the held-out test set are compared to the true values using:
- **Mean Absolute Error (MAE)**: Average of absolute differences between predicted and true ages.  
- **Mean Squared Error (MSE)**: Average of squared differences between predicted and true ages.  
- **R² Score**: Proportion of variance explained by the model.

---

## 5. Results and Discussion

Below are the reported final metrics of each model:

| Model                | MAE    | MSE    | R²    |
|----------------------|--------|--------|-------|
| Linear Regression    | 1.628  | 4.857  | 0.509 |
| XGBoost (Optimized)  | 1.561  | 4.710  | 0.524 |
| MLP (Optimized)      | 1.486  | 4.315  | 0.564 |

- **Linear Regression**: Serves as a baseline with an MAE of ~1.63 years and an R² of 0.51.  
- **XGBoost**: Improves upon the linear baseline (MAE ~1.56, R² ~0.52).  
- **MLP**: Achieves the best performance (MAE ~1.49, R² ~0.56), indicating that a neural network can capture nonlinear relationships more effectively in this data.

---

## 6. Explanation of Figures

### Figure 1: Linear Regression: Actual vs. Predicted Age  
![Linear Regression_Eval](https://github.com/user-attachments/assets/9f9a5de5-94bc-488f-855e-7ee777de0a7e)  

- This scatter plot compares the **actual abalone age** to what the linear model predicts.  
- Points deviate more noticeably from the diagonal line than in the XGBoost or MLP plots, reflecting larger errors and more systematic bias.  
- The cluster of points at higher actual ages (beyond ~15) show a tendency for underestimation, which aligns with the lower R² and higher MAE/MSE for Linear Regression.  

---

### Figure 2: XGBoost Training vs. Validation RMSE  
![XGBoost_Train](https://github.com/user-attachments/assets/8151a53e-04d0-4806-aa05-542dee70c2d8)  

- **Vertical axis**: Root Mean Squared Error (RMSE).  
- **Horizontal axis**: Boosting rounds (the number of trees grown).  
- The **blue curve** is training RMSE, and the **orange curve** is validation RMSE.  
- Both curves continuously decline, indicating that the model is learning. The training curve goes lower than the validation curve, which is typical as the training data can be fit more closely than unseen validation data.  
- Around ~150 to 200 boosting rounds, the validation curve flattens, suggesting diminishing returns on further rounds.  

---

### Figure 3: XGBoost Regressor (Optimized): Actual vs. Predicted Age  
![XGBoost_Eval](https://github.com/user-attachments/assets/bc0f4ae9-79bc-49b4-8b7c-287d6b4b5e3a)  

- Similar to the MLP scatter plot, **Actual Age** is on the horizontal axis, and **Predicted Age** is on the vertical axis.  
- XGBoost also places a large fraction of the predictions near or around the diagonal, but slightly more scatter is visible than with the MLP.  
- The model still performs better than Linear Regression, but is outshined by the final MLP results.  

---

### Figure 4: MLP Training Loss Curve  
![MLP_Train](https://github.com/user-attachments/assets/6be4b81d-197c-4518-b0f5-79765318fe46)  

- This plot shows the **loss** (vertical axis) over **training iterations** (horizontal axis) for the MLP Regressor.  
- The loss starts high (above 50) but rapidly decreases within the first few iterations. After about 10–20 iterations, it begins to stabilize and slowly converges to a small value.  
- A sharp decline at the beginning indicates that the model learns quickly once training starts, which is common for neural networks with proper scaling.  

---

### Figure 5: MLP Regressor (Optimized): Actual vs. Predicted Age  
![MLP_Eval](https://github.com/user-attachments/assets/ad9ead29-f9a0-4cab-9b4b-9a941f5fa90b)  

- **Horizontal axis**: The actual abalone age (from ~4 to ~25).  
- **Vertical axis**: The predicted age by the MLP model.  
- The red dashed line is a perfect 1:1 reference line—if every point were on this line, the prediction would be exactly correct.  
- Most points cluster near the diagonal, indicating good predictive performance. The spread of points is relatively narrow compared to the baseline model, reflecting MLP’s stronger accuracy.  

---

### Figure 6: Overall Performance Bar Charts  
![Overall_Eval](https://github.com/user-attachments/assets/1709d1f7-c1f9-426a-ae8b-8f97ded588c7)  

- **Left chart (Mean Absolute Error)**: Depicts the MAE for each model. Linear Regression has the highest error, while MLP has the lowest.  
- **Center chart (Mean Squared Error)**: Shows the MSE for each model. Again, MLP achieves the lowest MSE, followed by XGBoost and then the linear baseline.  
- **Right chart (R² Score)**: Compares how much variance is explained by each model. MLP attains the highest R², closely followed by XGBoost, then Linear Regression.  

These side-by-side bar plots visually confirm that the **MLP Regressor** outperforms the other two models.

---

## 7. Conclusions
1. **Baseline vs. Advanced Models**: Linear Regression provides a starting point but struggles with underestimation at higher ages.  
2. **Boosting & Neural Networks**: XGBoost addresses some of the linear model’s shortcomings, and MLP further improves results by capturing complex, nonlinear patterns.  
3. **Best Performer**: The MLP Regressor, with tuned hyperparameters (`(50, 50)` hidden layers, `tanh` activation, `learning_rate_init=0.005`, `solver='adam'`), achieves the best scores across all evaluation metrics.

Future enhancements may include:
- More feature engineering or inclusion of additional attributes.
- Even more extensive hyperparameter searches or use of alternative optimization strategies.
- Exploring ensemble methods that combine multiple model predictions.

Overall, this project demonstrates that while simple linear models are easy to implement and interpret, more complex models like XGBoost and MLP can substantially improve performance on the Abalone age prediction task.

**References**:
- Code snippets from `abalone.py`.  
- Dataset and attribute details from the UCI Abalone Dataset.  
- Figures and metrics courtesy of the final training and testing runs in this project.

---

*End of Report*
