# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Data preprocessing:
3. Cleanse data,handle missing values,encode categorical variables.
4. Model Training:Fit logistic regression model on preprocessed data.
5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6. Prediction: Predict placement status for new student data using trained model.
7. End the program.
 
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Suseel Kumar
RegisterNumber: 212225240163
*/
import pandas as pd
import numpy as np

np.random.seed(42)
m = 215
cats = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]

data = pd.DataFrame({"ssc_p": np.random.uniform(50, 95, m), "degree_p": np.random.uniform(50, 95, m), "etest_p": np.random.uniform(50, 95, m)})
for c in cats[:-1]: data[c] = np.random.choice(["A", "B"], m)
data["status"] = np.where(data["ssc_p"] + data["degree_p"] + data["etest_p"] > 200, "Placed", "Not Placed")
data["salary"] = np.nan

print("Original Data:\n", data.head().to_string())
data = data.drop(['salary'], axis=1)

for c in cats: data[c] = data[c].astype('category')
print("\nData types:\n", data.dtypes.head())
for c in cats: data[c] = data[c].cat.codes

print("\nNumeric Data:\n", data.head().to_string())
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
print(f"\nShapes: X={x.shape}, y={y.shape}\nInitial theta:\n", np.random.randn(x.shape[1]))

def sigmoid(z): return 1 / (1 + np.exp(-z))

def gradient_descent(theta, X, y, alpha, iters):
    for i in range(iters):
        h = sigmoid(X.dot(theta))
        theta -= alpha * (X.T.dot(h - y) / len(y))
        if (i + 1) % 100 == 0: print(f"Iteration {i+1}, Loss: {-np.sum(y * np.log(h+1e-10) + (1-y) * np.log(1-h+1e-10)):.4f}")
    return theta

theta = gradient_descent(np.zeros(x.shape[1]), x, y, 0.01, 1000)
print("\nFinal theta:\n", theta)

y_pred = np.where(sigmoid(x.dot(theta)) >= 0.5, 1, 0)
print("\nAccuracy:", np.mean(y_pred == y), "\nPredicted:", y_pred[:20])
print("\nNew student 1:", y_pred[0], "\nNew student 2:", y_pred[1])
```

## Output:
<img width="1137" height="512" alt="image" src="https://github.com/user-attachments/assets/3afc3761-ac79-4666-93de-868b87d96c17" />
<img width="757" height="537" alt="image" src="https://github.com/user-attachments/assets/56edeea7-7ac6-4d45-b6d9-af70a62832ce" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

