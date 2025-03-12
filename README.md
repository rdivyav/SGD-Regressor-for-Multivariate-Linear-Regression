# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation and Scaling 
2.Model Initialization
3.Model Training 
4.Prediction and Evaluation 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Divya R V
RegisterNumber: 212223100005 
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = fetch_california_housing()

# Prepare the input (x) and output (y) variables
x = data.data[:, :3]  # Selects the first 3 features (Longitude, Latitude, Housing Median Age)
y = np.column_stack((data.target, data.data[:, 6]))  # Combines house price and number of occupants

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Initialize the scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Scale the training and testing data
x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Initialize the SGD regressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Wrap the SGD regressor in a MultiOutputRegressor
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(x_train, y_train)

# Make predictions on the test set
y_pred = multi_output_sgd.predict(x_test)

# Transform the predicted and actual outputs back to their original scale
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the first 5 predictions
print("\nPredictions:\n", y_pred[:5])

```

## Output:
![multivariate linear regression model for predicting the price of the house and number of occupants in the house](sam.png)


![Screenshot 2025-03-12 035704](https://github.com/user-attachments/assets/f554efaf-1d33-4d92-b88e-95a2997fcde7)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
