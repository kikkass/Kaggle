import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ------------- Preparing train ad test data (using imputation) -----------------
data_source = "./datasets/melb_data.csv"
raw_data = pd.read_csv(data_source)

raw_data.dropna(axis = 0, subset = ['Price'], inplace = True)

target = raw_data.Price
predictors = raw_data.drop(['Price'], axis = 1).select_dtypes(exclude = ["object"])

X_train, X_test, y_train, y_test =train_test_split(predictors, target, test_size = 0.25)

my_imputer = SimpleImputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.transform(X_test)

# ------------- Building the model using XGBoost -------------------
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

predictions = xgb_model.predict(X_test)

# ------ Calculate MAE ------
from sklearn.metrics import mean_absolute_error

print("MEA of XGBRegressr :", mean_absolute_error(predictions, y_test))


# ------ using n_estimators and early_stopping_rounds -------

my_new_model = XGBRegressor(n_estimators = 374)
my_new_model.fit(X_train, y_train, 
					# early_stopping_rounds = 10,
					# eval_set = [(X_test, y_test)],
					verbose = True)

predictions = my_new_model.predict(X_test)
print("MEA of XGBRegressr (tuned - n-estimaors):", mean_absolute_error(predictions, y_test))

# ------ Using learning_rate ---------

my_new_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05)
my_new_model.fit(X_train, y_train, verbose = False)

predictions = my_new_model.predict(X_test)
print("MEA of XGBRegressr (tuned n-estiators + learning_rate):", mean_absolute_error(predictions, y_test))

