import pandas as pd
train_data_path="home-data/train.csv"
test_data_path="home-data/test.csv"
train_data=pd.read_csv(train_data_path)
test_data=pd.read_csv(train_data_path)
print(train_data.describe())
y_train=train_data.SalePrice
features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X_train=train_data[features]
print(X_train.head())
print(X_train.describe())
print(y_train.describe())

y_test=test_data.SalePrice
X_test=test_data[features]
print(X_train.head())
print(X_train.describe())
print(y_train.describe())

from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(X_train,y_train)
predict_y=forest_model.predict(X_test)
mae=mean_absolute_error(y_test,predict_y)
print(mae)
