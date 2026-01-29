import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading data form the csv file
df = pd.read_csv('data.csv')

# loading numpy arrays
x = df['SquareFeet'].values.astype(float)
y = df['Price'].values.astype(float)

# Normalize input for stable gradient descent
x_mean = np.mean(x)
x_std = np.std(x)

x_norm = (x - x_mean) / x_std

lr = 0.01        # learning rate
epochs = 2000    # number of gradient descent iterations

# Initialize model parameters
w = 0.0   # slope in normalized space
b = 0.0   # intercept in normalized space

N = len(x_norm)

for _ in range(epochs):
    y_pred = w * x_norm + b
    error = y_pred - y
    
    dw = (2/N) * np.sum(error * x_norm)
    db = (2/N) * np.sum(error)
    
    w -= lr * dw
    b -= lr * db

# Convert parameters back to original x scale
w_real = w / x_std
b_real = b - (w_real * x_mean)

print("Trained slope (w):", w_real)
print("Trained intercept (b):", b_real)

# Predictions for visualization
y_hat = w_real * x + b_real

#Plotting data using matplotlib
plt.scatter(x, y, label='Data')
plt.plot(x, y_hat, label='Regression Line', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

ss_total = np.sum((y - np.mean(y))**2)
ss_res = np.sum((y - y_hat)**2)
r2 = 1 - (ss_res / ss_total)

print("R^2 Score:", r2)
