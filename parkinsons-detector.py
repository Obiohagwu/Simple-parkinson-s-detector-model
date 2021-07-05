import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.data')
#df.head()

#To get the featues and labels from dataset. The features are the columns except 'status' column.
features = df.loc[:,df.columns!='status'].values[:,1:]
labels = df.loc[:, 'status'].values

#Now we can proceed to get the count of each label(0 and 1) in labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
#from the output, we see that there are 147 1's and 48 0's in status column

#Now we can proceed to scale our feature dataset.
#We will normailze the feature dataset to a range [-1,1]
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

#Now we can split the datasets into train and test.
#We should keep 20% for testing
# test_size is usually between 0.0 and 1.0. We chose 0.2 because we want 20% test data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
model = XGBClassifier()
model.fit(x_train, y_train)

#Then we finally proceed to get y_pred.
#y_pred is the predicted value for our feature testset
y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)
