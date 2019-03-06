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

#------------- Calculating MAE of model -------------------

# importing necessary libraries
from sklearn.metrics import mean_absolute_error

#Predicting the price using model

predicted_prices = melb_housing_model.predict(X)

# calculatingf mae
melb_house_model_mae = mean_absolute_error(y, predicted_prices)

print(melb_house_model_mae)
