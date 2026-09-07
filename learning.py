import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

cols=["sepal length","sepal width","petal length","petal width","class"]
df=pd.read_csv("iris.data",names=cols)
# print(df["class"].unique())
df["class"]=df["class"].map({
    "Iris-setosa":0,
    "Iris-versicolor":1,
    "Iris-virginica":2
})

# for label in cols[:-1]:
#     plt.hist(df[df["class"]==0],color="blue",label="Iris-setosa",alpha=0.7,density=True)
#     plt.hist(df[df["class"]==1],color="red",label="Iris-versicolor",alpha=0.7,density=True)
#     plt.hist(df[df["class"]==2],color="orange",label="Iris-virginica",alpha=0.7,density=True)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train = df[:int(0.6 * len(df))]
valid = df[int(0.6 * len(df)):int(0.8 * len(df))]
test  = df[int(0.8 * len(df)):]

# print(len(train[train["class"]==0]))
# print(len(train[train["class"]==1]))
# print(len(train[train["class"]==2]))

X_train=train.drop("class",axis=1)
Y_train=train["class"]

X_valid=valid.drop("class",axis=1)
Y_valid=valid["class"]

X_test=test.drop("class",axis=1)
Y_test=test["class"]

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_valid=scaler.transform(X_valid)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

y_pred=knn.predict(X_test)
print(y_pred)
print(classification_report(Y_test,y_pred))

nb=GaussianNB()
nb.fit(X_train,Y_train)
y_pred=nb.predict(X_test)
print(classification_report(Y_test,y_pred))

lb=LogisticRegression()
lb.fit(X_train,Y_train)
y_pred=lb.predict(X_test)
print(classification_report(Y_test,y_pred))

svm=SVC()
svm.fit(X_train,Y_train)
y_pred=svm.predict(X_test)
print(classification_report(Y_test,y_pred))

