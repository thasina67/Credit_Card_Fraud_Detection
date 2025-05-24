import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score



data= pd.read_csv("C:\\credit_card_fraud_detection_project\\dataset\\creditcard.csv")
print(data.head())

pd.options.display.max_columns= None
print(data.head())
print(data.tail())
print(data.shape)

print("Number of columns: {}".format(data.shape[1]))
print("Number of rows:{}".format(data.shape[0]))

print(data.info())

print(data.isnull().sum())
sc= StandardScaler()
data['Amount']  = sc.fit_transform(pd.DataFrame(data['Amount']))
print(data.head())

data= data.drop(['Time'], axis=1)
print(data.head())

data.duplicated().any()
data=data.drop_duplicates()
print(data.shape)
data['Class'].value_counts()

plt.style.use('ggplot')
#sns.countplot(data['Class'])
sns.countplot(data=data, x='Class')
plt.show()

X= data.drop('Class', axis=1)
y=data['Class']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)


classifier= {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree Classifier": DecisionTreeClassifier()
}

for name, clf in classifier.items():
    print(f"\n======{name}====")
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test,y_pred)}")
    print(f"\n Recall: {recall_score(y_test,y_pred)}")
    print(f"\n F1 Score: {accuracy_score(y_test, y_pred)}")

#Undersampling
normal= data[data['Class']==0]
fraud= data[data['Class']==1]

print(normal.shape)
print(fraud.shape)
print(normal.sample())
normal_sample=normal.sample(n=473)
print(normal_sample.shape)
new_data=pd.concat([normal_sample,fraud], ignore_index=True)
print(new_data.head())

print(new_data['Class'].value_counts())

#OVERSAMPLING

X=data.drop('Class',axis=1)
y=data['Class']

print(X.shape)
print(y.shape)
from imblearn.over_sampling import SMOTE
X_res, y_res =SMOTE().fit_resample(X,y)
print(y_res.value_counts())

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

classifier={
    "Logistic Regression": LogisticRegression(),
    "Decision Tree  classifier": DecisionTreeClassifier()
}
for name, clf in classifier.items():
    print(f"\n======{name}====")
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"\n Precision: {precision_score(y_test,y_pred)}")
    print(f"\n Recall: {recall_score(y_test,y_pred)}")
    print(f"\n F1 Score: {accuracy_score(y_test, y_pred)}")

dtc= DecisionTreeClassifier()
dtc.fit(X_res,y_res)

import joblib
joblib.dump(dtc, "credit_card_model.pkl")
model=joblib.load("credit_card_model.pkl")
pred= model.predict([[-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]])

print(pred[0])

if pred[0]==0:
    print("Normal Transaction")
else:
    print("Fraud Transaction")

 