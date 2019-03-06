''' Generating a model with Random Forest '''

#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data_source = './datasets/melb_data.csv'
housing_data = pd.read_csv(data_source).dropna(axis = 0)

y = housing_data.Price
housing_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = housing_data[housing_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# Define the model - Random forest regressor
housing_model = RandomForestRegressor(random_state = 1)

# Fit the model
housing_model.fit(X_train, y_train)

# Predict the price for testing data
predicted_price = housing_model.predict(X_test)

# Validate the model using MAE
housing_model_mae = mean_absolute_error(y_test, predicted_price)

print("Mean absolute error: ", housing_model_mae)

