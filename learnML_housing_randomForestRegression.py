#load dataset
import pandas as pd
tdata_path="home-data/train.csv"
tdata=pd.read_csv(tdata_path)

print(tdata.head())
print(tdata.describe())
print(tdata.columns)
y=tdata.SalePrice
#Select features
features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X=tdata[features]
print(X.describe())
print(X.columns)

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
forest_predict=forest_model.predict(val_X)
mae=mean_absolute_error(val_y,forest_predict)
print(mae)
