import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('gender.csv')
X = df[["Height", "Weight", "BMI Index"]]
y = df["Gender"]

y = y.map({"Male": 1, "Female": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Training Data : \n")
print(X_train)
print(y_train)

print("\n Testing Data : \n")
print(X_test)
print(y_train)

log = LogisticRegression()
log.fit(X_train, y_train)
logpreds = log.predict(X_test)

dec = DecisionTreeClassifier()
dec.fit(X_train, y_train)
decpreds = dec.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
svmpreds = svm.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
knnpreds = knn.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logpreds))
print("Decision Tree Accuracy:", accuracy_score(y_test, decpreds))
print("SVM Accuracy:", accuracy_score(y_test, svmpreds))
print("k-NN Accuracy:", accuracy_score(y_test, knnpreds))

print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, logpreds))

print("\nClassification Report (Decision Tree Classifier):\n")
print(classification_report(y_test, decpreds))

print("\nClassification Report (SVM):\n")
print(classification_report(y_test, svmpreds))

print("\nClassification Report (KNearest Neighbours):\n")
print(classification_report(y_test, knnpreds))

