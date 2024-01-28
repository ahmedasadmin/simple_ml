# coding: utf-8
import pandas as pd 
dataset = pd.read_csv('wine.csv')

y = dataset['Wine']

x = dataset.iloc[:,1:]

x = dataset.iloc[:,1:].values

y = dataset['Wine'].values


from sklearn.model_selection import train_test_split

x_train, X_test, y_train, Y_test = train_test_split(x, y, test_size=  0.25, random_state=0)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, Y_train)
# predict  the test set results 
y_pred = classifier.predict(X_test)
# generate confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, y_pred, labels=classifier.classes_)
disp =  ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.show()
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_test, y_pred)

