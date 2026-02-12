# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by removing unnecessary columns.
2. Convert categorical data into numerical format and scale features.
3. Initialize parameters and apply Gradient Descent using the sigmoid function.
4. Update weights iteratively to minimize cost and compute model accuracy.

## Program:
```
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_excel("Placement_Data.xlsx")

# Drop unnecessary columns
if 'sl_no' in data.columns:
    data = data.drop('sl_no', axis=1)

if 'salary' in data.columns:
    data = data.drop('salary', axis=1)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop('status', axis=1).values
y = data['status'].values

# Feature Scaling (important for Gradient Descent)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
m, n = X.shape
X = np.c_[np.ones(m), X]

# Initialize parameters
theta = np.zeros(n + 1)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Training
learning_rate = 0.01
iterations = 1000

theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

print("Final Parameters:", theta)

# Predictions
probabilities = sigmoid(X @ theta)
predictions = probabilities >= 0.5

# Accuracy
accuracy = np.mean(predictions == y)
print("Model Accuracy:", accuracy)

# Plot Cost Reduction
plt.figure()
plt.plot(cost_history)
plt.title("Cost Function Decreasing")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

```

## Output:
<img width="959" height="664" alt="image" src="https://github.com/user-attachments/assets/7b1fb206-3c3f-4af5-81ba-484e5980fe88" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

