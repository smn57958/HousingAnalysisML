#Import pandas for csv
import pandas as pd

#Load path for train and test data
data_path="home-data/train.csv"

#Load data
data=pd.read_csv(data_path)

#Describe data
print(data.describe())

#Check first 5 rows of data_path
print(data.head())
print(data.columns)

#Target prediction column
y=data.SalePrice

#Select features
features=['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

#Select feature columns from the dataset
X=data[features]
print(X.describe())
print(X.head())

#import scikitlearn decisiontree regressor- fits sine curve with additional noise, higher the depth, higher chance of learning from noise or overfitting

from sklearn.tree import DecisionTreeRegressor
import numpy as np

#Fit regression model
regr_model=DecisionTreeRegressor(random_state=1)
regr_model.fit(X,y)

predictions=regr_model.predict(X)
print(predictions[:6])

#Mean absolute error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y,predictions))

#Since Mean absolute error doesn't make sense in 'in sampling' of the data, we need validation set. We can make use of sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=1)
regr_model.fit(train_X,train_y)
val_predictions=regr_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    regr_model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    regr_model.fit(train_X,train_y)
    prediction=regr_model.predict(val_X)
    mae=mean_absolute_error(val_y,prediction)
    return mae

candidate_max_leaf_nodes=[5,10,40,70,71,73,80,300]
scores={}
for max_leaf_node in candidate_max_leaf_nodes:
    scores[max_leaf_node]=get_mae(max_leaf_node,train_X,val_X,train_y,val_y)


best_tree_size=min(scores, key=scores.get)
print(scores)
print(best_tree_size)

regr_model=DecisionTreeRegressor(max_leaf_nodes=71,random_state=1)
regr_model.fit(train_X,train_y)
prediction=regr_model.predict(val_X)
mae=mean_absolute_error(val_y,prediction)

print(mae)
