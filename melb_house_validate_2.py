# Previously built model - melb_house_model.py

import pandas as pd
data_set_path = './datasets/melb_data.csv'
melb_house_data = pd.read_csv(data_set_path)
melb_house_data = melb_house_data.dropna(axis=0)
y = melb_house_data.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melb_house_data[melb_features]
from sklearn.tree import DecisionTreeRegressor
melb_housing_model = DecisionTreeRegressor(random_state = 1)
melb_housing_model.fit(X,y)

#------------- Calculating MAE of model ("In-Sample")-------------------
from sklearn.metrics import mean_absolute_error
predicted_prices = melb_housing_model.predict(X)
melb_house_model_mae = mean_absolute_error(y, predicted_prices)
print('MAE for "In-Sample" data: ', melb_house_model_mae)

'''Calculating MAE by validating against non trained data
using train_test_split function'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

#fit model
melb_housing_model.fit(X_train, y_train)

#predict model
new_predicted_prices = melb_housing_model.predict(X_test)
print(new_predicted_prices.tolist()[:10])
print(y_test.head())

#New MAE for the model
updated_mae = mean_absolute_error(new_predicted_prices, y_test)
print('Updated MAE: ', updated_mae)
