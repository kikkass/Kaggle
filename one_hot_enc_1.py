import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

def get_cv_mae(X, y):
	''' This functions caculates 
	mean absolute error by cross_val_score'''

	return -1 * cross_val_score(RandomForestRegressor(100),
					X, y,
					scoring = 'neg_mean_absolute_error').mean()


data_source = "./datasets/melb_data.csv"
housing_data = pd.read_csv(data_source)

# drop the rows where Price is not available
housing_data.dropna(axis = 0, subset = ['Price'], inplace = True)

# Set the target (y)
target = housing_data.Price

# Find the columns with missing data
missing_data_columns = [col for col in housing_data.columns if housing_data[col].isnull().any()]

# Drop the missing data columns and target
housing_data = housing_data.drop(missing_data_columns + ['Price'], axis = 1)

# print(housing_data.dtypes.to_dict())

# get low cordinality columns
low_codinality_columns = [col for col in housing_data.columns if
								housing_data[col].nunique() < 10 and
								housing_data[col].dtype == 'object']
# get numeric columns
numeric_columns = [col for col in housing_data.columns if
								housing_data[col].dtype in ['int64', 'float64']]


housing_data_predictors = housing_data[low_codinality_columns + numeric_columns]
# print(housing_data_predictors.dtypes)
# Type              object
# Method            object
# Regionname        object
# Rooms              int64
# Distance         float64
# Postcode         float64
# Bedroom2         float64
# Bathroom         float64
# Landsize         float64
# Lattitude        float64
# Longtitude       float64
# Propertycount    float64
# dtype: object

# One-Hot encoding of predictor dataframe
numeric_housing_data = housing_data[numeric_columns]
encoded_housing_data = pd.get_dummies(housing_data_predictors)
# print(encoded_housing_data.dtypes)
# Rooms                                      int64
# Distance                                 float64
# Postcode                                 float64
# Bedroom2                                 float64
# Bathroom                                 float64
# Landsize                                 float64
# Lattitude                                float64
# Longtitude                               float64
# Propertycount                            float64
# Type_h                                     uint8
# Type_t                                     uint8
# Type_u                                     uint8
# Method_PI                                  uint8
# Method_S                                   uint8
# Method_SA                                  uint8
# Method_SP                                  uint8
# Method_VB                                  uint8
# Regionname_Eastern Metropolitan            uint8
# Regionname_Eastern Victoria                uint8
# Regionname_Northern Metropolitan           uint8
# Regionname_Northern Victoria               uint8
# Regionname_South-Eastern Metropolitan      uint8
# Regionname_Southern Metropolitan           uint8
# Regionname_Western Metropolitan            uint8
# Regionname_Western Victoria                uint8
# dtype: object

print("MAE for One-Hot encoding: " + str(get_cv_mae(encoded_housing_data, target)))
print("MAE for only numeric columns: " + str(get_cv_mae(numeric_housing_data, target)))

