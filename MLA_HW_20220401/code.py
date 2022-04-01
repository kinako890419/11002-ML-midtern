#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

#import dataset
df = pd.read_csv("data/train_data_titanic.csv")

#df.head()
#df.info() #檢視各欄位資料形式

#Remove the columns model will not use
df.drop(['Name','Ticket'],axis=1,inplace=True)
df.drop(['PassengerId','Fare'],axis=1,inplace=True)

#處理缺失值
#Cabin has too many missing values
df.drop('Cabin',axis=1,inplace=True)
#缺失值男生就用男生的中位數(29)、女生就用女生的中位數(27)來填補
df['Age'] = df.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
#Embarked缺失值
#df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(),inplace=True)

#將Sex, Embarked進行轉換
#Sex轉換成是否爲男生、是否爲女生，Embarked轉換爲是否爲S、是否爲C、是否爲Q
df = pd.get_dummies(data=df, columns=['Sex','Embarked'])
#df = pd.get_dummies(data=df, columns=['Sex'])
#df.head()
#是否爲男生與是否爲女生只要留一個就好，留下是否爲男生
df.drop(['Sex_female'], axis=1, inplace=True)
#df.head()

#Prepare training data
#把Survived, Pclass丟掉
X = df.drop(['Survived','Pclass'],axis=1)
y = df['Survived']
#split to training data & testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=67)

#using Logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)

#Evaluate
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
accuracy_score(y_test, predictions)
recall_score(y_test, predictions)
precision_score(y_test, predictions)
pd.DataFrame(confusion_matrix(y_test, predictions), columns=['Predict not Survived','Predict Survived'], index=['True not Survived', 'True Survived'])

#Model Export
import joblib
joblib.dump(lr,'Titanic-LR-20220401_10.pkl',compress=3)#compress=壓縮率(0-9)

#Model Using
#import joblib
model_pretrained = joblib.load('Titanic-LR-20220401_10.pkl')
#import pandas as pd
#for submission
df_test = pd.read_csv("test.csv")

df_test.drop(['Name','Ticket'],axis=1,inplace=True)
df_test.drop(['PassengerId','Fare'],axis=1,inplace=True)
df_test.drop('Cabin',axis=1,inplace=True)

df_test['Age'] = df_test.groupby('Sex')['Age'].apply(lambda x: x.fillna(x.median()))
#df_test.isnull().sum()
#df_test['Fare'].value_counts()
#df_test['Fare'].fillna(df_test['Fare'].value_counts().idxmax(),inplace=True)
df_test = pd.get_dummies(data=df_test, columns=['Sex','Embarked'])
#df_test = pd.get_dummies(data=df_test, columns=['Sex'])
df_test.drop('Sex_female',axis=1,inplace=True)
df_test.drop('Pclass',axis=1,inplace=True)

#執行、輸出
predictions2 = model_pretrained.predict(df_test)
predictions2

#Prepare submit file
forSubmissionDF = pd.DataFrame(columns=['PassengerId','Survived'])
forSubmissionDF['PassengerId'] = range(892,1310)
forSubmissionDF['Survived'] = predictions2
forSubmissionDF
forSubmissionDF.to_csv('for_submission_20220401_10.csv', index=False)