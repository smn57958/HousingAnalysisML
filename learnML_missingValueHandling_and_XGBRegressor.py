import pandas as pd

data_path="home-data/train.csv"
data=pd.read_csv(data_path)
data_copy=data.copy()
print(data.head())
print(data.describe())
print(data.columns)
print(data.shape)
data.dropna(axis=0,subset=['SalePrice'],inplace=True)
y=data.SalePrice
X=data.drop(['SalePrice'],axis=1).select_dtypes(exclude=['object'])

from sklearn.model_selection import train_test_split
train_X,val_X,train_y,val_y=train_test_split(X.as_matrix(),y.as_matrix(),train_size=0.75,test_size=0.25)

from sklearn.preprocessing import Imputer
my_imputer=Imputer()
train_X=my_imputer.fit_transform(train_X)
val_X=my_imputer.transform(val_X)

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
score={};
def get_mae(estimators,learning_rate):
    xgbrgr_model=XGBRegressor(n_estimators=estimators,learning_rate=learning_rate)
    xgbrgr_model.fit(train_X,train_y,verbose=False)
    predicted_y=xgbrgr_model.predict(val_X)
    score[estimators]=mean_absolute_error(val_y,predicted_y)
    bestFit=min(score,key=score.get)
    return bestFit;

def get_mae_estimators(estimators,learning_rate):
    xgbrgr_model=XGBRegressor(n_estimators=estimators,learning_rate=learning_rate)
    xgbrgr_model.fit(train_X,train_y,verbose=False)
    predicted_y=xgbrgr_model.predict(val_X)
    score[estimators]=mean_absolute_error(val_y,predicted_y)
    bestFit=min(score,key=score.get)
    return bestFit;

def get_mae_learning_rate(estimators,learning_rate):
    xgbrgr_model=XGBRegressor(n_estimators=estimators,learning_rate=learning_rate)
    xgbrgr_model.fit(train_X,train_y,verbose=False)
    predicted_y=xgbrgr_model.predict(val_X)
    score[learning_rate]=mean_absolute_error(val_y,predicted_y)
    bestFit=max(score,key=score.get)
    return bestFit;

flag=True;
estimator=get_mae(100,0.01)
error=score.get(estimator)
oldError=error;
count=0;
while(flag==True and count<5):
    estimator=get_mae_estimators(estimator,0.01)
    error=score.get(estimator)
    print("Estimator:",estimator,"\tError:",score.get(estimator))
    if(oldError<error):
        flag==False
        count=0;
    else:
        if((int(oldError/1000))==(int((error/1000)))):
            count+=1
        else:
            count=0;
        oldError=error
        estimator+=50
print(estimator)
flag=False
count=0
oldError=0
error=0;
learning_rate=0.01

while(flag==True and count<5 and learning_rate<0.06):
    estimator=get_mae_learning_rate(estimator,learning_rate)
    error=score.get(estimator)
    print("Learning rate:",learning_rate,"\tError:",score.get(estimator))
    if(oldError<error):
        flag==False
        count=0;
    else:
        if((int(oldError/1000))==(int((error/1000)))):
            count+=1
        else:
            count=0;
        oldError=error
        learning_rater+=0.01
