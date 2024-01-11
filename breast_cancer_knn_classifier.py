# import pandas as pd 
# from IPython.display import display
# import sys 
# import matplotlib as plt 
# import scipy as sp 
# import numpy as np
# import IPython
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# iris_dataset  = load_iris()
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
#                                                      iris_dataset['target'], random_state = 0)


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
cancer  = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66)


training_acuracy = []
test_acuracy = []

# try n_neighbors from  1 to 10
neighbors_settings = range(1,11);
for n_neighbors in neighbors_settings:
    # building the model 
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # record training accuracy 
    training_acuracy.append(clf.score(X_train, y_train))
    # record generalization accuracy 
    test_acuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_acuracy, label="training accuracy")
plt.plot(neighbors_settings, test_acuracy, label="test accuray")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")

plt.legend()
plt.show()
