import pandas as pd

#specify the path to training data sets
data_set_path = './datasets/melb_data.csv'

#load data into pandas data frame
melb_house_data = pd.read_csv(data_set_path)

#filter rows with missing data
melb_house_data = melb_house_data.dropna(axis=0)

#get the column names for building model
# print(melb_house_data.columns)

#set prediction target
y = melb_house_data.Price

#select the model feature for prediction
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = melb_house_data[melb_features]

#import necessary tools for building model
from sklearn.tree import DecisionTreeRegressor

#define the model
melb_housing_model = DecisionTreeRegressor(random_state = 1)

#fit the model
melb_housing_model.fit(X,y)

#predict the target
predicted_prices = melb_housing_model.predict(X)
print(predicted_prices)