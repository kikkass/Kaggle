from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split

def get_mae(X_train, X_val, y_train, y_val):
	''' THis function builds a random forest model
	and returns the 'Out of sample MAE'''

	model = RandomForestRegressor(random_state = 1)
	model.fit(X_train, y_train)

	return mean_absolute_error(model.predict(X_val), y_val)

data_source = './datasets/melb_data.csv'
housing_data = pd.read_csv(data_source)

melb_target = housing_data.Price
melb_numeric_predictor = housing_data.select_dtypes(exclude = ['object'])

X_train, X_val, y_train, y_val = train_test_split(melb_numeric_predictor, melb_target, train_size = 0.7, random_state = 1)

#---------------------------------------------------------------------------------------------------------------------------
# Get Model Score from Dropping Columns with Missing Values
#---------------------------------------------------------------------------------------------------------------------------

# Columns with mssing alues
cols_with_missing_data = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing_data, axis = 1)
reduced_X_val = X_val.drop(cols_with_missing_data, axis = 1)

print("MEA when dropping the columns with  missing data: ", get_mae(reduced_X_train, reduced_X_val, y_train, y_val))

#---------------------------------------------------------------------------------------------------------------------------
# Get Model Score from Imputation
#---------------------------------------------------------------------------------------------------------------------------

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_val = my_imputer.transform(X_val)

print("MEA when imputing missing data: ", get_mae(imputed_X_train, imputed_X_val, y_train, y_val))

#---------------------------------------------------------------------------------------------------------------------------
# Get Score from Imputation with Extra Columns Showing What Was Imputed
#---------------------------------------------------------------------------------------------------------------------------

# Create the copy of original data
X_train_plus = X_train.copy()
X_val_plus = X_val.copy()

# Columns with mising data are store in cols_with_missing_data
for col in cols_with_missing_data:
	X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
	X_val_plus[col + '_was_missing'] = X_val_plus[col].isnull()

# Imputation
imputed_X_train_plus = my_imputer.fit_transform(X_train_plus)
imputed_X_val_plus = my_imputer.transform(X_val_plus)

print("MEA when imputed with Extra Columns Showing What Was Imputed: ", get_mae(imputed_X_train_plus, imputed_X_val_plus, y_train, y_val))
