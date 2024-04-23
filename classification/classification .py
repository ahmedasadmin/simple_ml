# classification 

import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
(LinearDiscriminantAnalysis as LDA,
QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 


Smarket = load_data('Smarket')
print(Smarket)

print(Smarket.columns)
print(Smarket.corr())
print(Smarket.plot(y='Volume'))
# plt.show()

allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X=design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm=sm.GLM(y,
           X,
           family=sm.families.Binomial())
results = glm.fit()
summarize(results)
