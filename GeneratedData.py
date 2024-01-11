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

import mglearn 
import matplotlib.pyplot as plt
x, y = mglearn.datasets.make_wave(n_samples=40)

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)


print("Test set accuracy: {:2f}".format(clf.score(X_test, y_test)))

# This plots size is in inches ..lla
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # the  fit method returns the object self, so we can instantiate
    # and fit in one line 
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend(loc=3)
plt.show()



