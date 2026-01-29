# Linear Regression from Scratch  
### Winter in Data Science (WiDS) – Final Assignment

## Overview
This project is the final assignment submitted as part of the **Winter in Data Science (WiDS)** program.  
The goal of this assignment is to implement **Simple Linear Regression from scratch** using Python, without relying on machine learning libraries such as `scikit-learn`.

The model predicts **house prices** based on a single input feature: **Square Feet**, using **gradient descent**.

---

## Objectives
- Understand the mathematical foundation of linear regression  
- Implement gradient descent manually  
- Apply feature normalization for stable training  
- Visualize regression results  
- Evaluate model performance using the R² score  

---

## Project Structure

---

## Dataset Description
The dataset contains the following columns:

| Column Name  | Description |
|-------------|------------|
| SquareFeet  | Area of the house in square feet |
| Price       | Price of the house |

---

## Methodology

### 1. Data Loading
The dataset is loaded using Pandas and converted into NumPy arrays for numerical computation.

### 2. Feature Normalization
To ensure stable gradient descent, the input feature is normalized using mean and standard deviation.

### 3. Model Definition
The linear regression model is defined as:

y = w * x + b

where:
- w is the slope
- b is the intercept

### 4. Training using Gradient Descent
Batch gradient descent is implemented manually to update model parameters over multiple epochs.

### 5. Parameter Rescaling
After training on normalized data, parameters are converted back to the original scale.

### 6. Visualization
Matplotlib is used to plot:
- Original data points  
- The regression line  

### 7. Model Evaluation
The model is evaluated using the **R² (coefficient of determination)** score.

---

## Results
The script outputs:
- Trained slope (w)
- Trained intercept (b)
- R² score

A regression plot is displayed to visually assess the model fit.

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  

---

## Constraints Followed
- No use of `scikit-learn` or ML frameworks  
- Only NumPy and Pandas for computation  
- Matplotlib used for visualization  

---

## Learning Outcomes
- Clear understanding of how linear regression works internally  
- Hands-on experience with gradient descent  
- Importance of feature normalization  
- Confidence in implementing ML algorithms from scratch  

---

## How to Run
Ensure `data.csv` is in the same directory as the script.
