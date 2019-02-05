print("Low variance feature selection")

import pandas as pd
data=pd.read_csv("home-data/train.csv")
print(data.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
y=data['SalePrice']
cols=data.columns
X=data[cols[:80]]
print(X.shape)
X_new=SelectKBest(chi2,k=2).fit_transform(X,y)
print(X_new.shape)
