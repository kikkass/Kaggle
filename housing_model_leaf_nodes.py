''' This excercise demonstrate how to find the
best deapth of leaf node in order to get better accuracy'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val):
	''' This function will build the model with specified number
	of leaf nodes, predict the price on validation data and 
	returns the MAE for corresponding number of leaf nodes'''

	# Defining the model with max_leaf_nodes
	model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 1)

	# Fitting the model with training data sets
	model.fit(X_train, y_train)

	# Predict the price for validation data and find the MAE and return
	return mean_absolute_error(model.predict(X_val), y_val)




data_source = './datasets/melb_data.csv'
housing_data = pd.read_csv(data_source)
housing_data = housing_data.dropna(axis = 0)

y = housing_data.Price
model_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = housing_data[model_features]

# Get the training and validation data sets
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

# Define list to store leaf numbers and MAE
leafs = []
mean_abs_error = []

# Compare MAE with different leaf node numbers
for max_leaf_nodes in range(2,1000,1):
	mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
	leafs.append(max_leaf_nodes)
	mean_abs_error.append(mae)
	# print("Max leaf nodes: {:<6}\t Mean Absolute Error: {:<10}".format(max_leaf_nodes, mae))

print("Minimum MAE: ",min(mean_abs_error))
print("Optimal number of leaf nodes: ",leafs[mean_abs_error.index(min(mean_abs_error))])

plt.plot(leafs, mean_abs_error, 'k-')
plt.show()






