# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset into a DataFrame and explore its contents to understand the data structure.
2. Separate the dataset into independent (X) and dependent (Y) variables and split them into training and testing sets.
3. Create a simple linear regression model and fit it using the training data.
4. predict the results for the testing set and plot the training and testing sets with the fitted regression line. 
5. Calculate error metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to evaluate the model’s performanc

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Vinodhini M.K 
RegisterNumber:212225230305
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

# -------- Feature Scaling --------
x_mean = np.mean(x)
x_std = np.std(x)

x = (x - x_mean) / x_std

# Parameters
w = 0.0
b = 0.0
alpha = 0.01
epochs = 100
n = len(x)
losses = []

# Gradient Descent
for _ in range(epochs):
    y_hat = w * x + b
    loss =np.mean((y_hat - y) ** 2)
    losses.append(loss)
    
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)
    w -= alpha * dw
    b -= alpha * db

    # Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y)

x_sorted = np.argsort(x)
plt.plot(x[x_sorted], (w * x + b)[x_sorted], color='red')

plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()

print("Final weight (w):", w)
print("Final bias (b):", b)
```
## Output:
![alt text](<Screenshot 2026-01-30 114232.png>)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
