import numpy as np 
import pandas as pd 
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
     import  variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
class draw:

    #####################################################
    def abline(ax, b, m):
        "Add a line with slope 'm' and intercept 'b' to ax"
        xlim = ax.get_xlim()
        ylim = [m * xlim[0] + b, m*xlim[1] + b]
        ax.plot(xlim, ylim)

    ######################################################
    def abline(ax, b, m, *args, **kwargs):
        "Add a line with slope m and intercept b to ax"
        xlim = ax.get_xlim()
        ylim = [m * xlim[0] + b, m*xlim[1] + b]
        ax.plot(xlim, ylim, *args, **kwargs)
    ########################################################

d1 = draw


Boston = load_data("Boston")
print(Boston.columns)
X = pd.DataFrame({'intercept':np.ones(Boston.shape[0]),
                  'lstat':Boston['lstat']})
y = Boston['medv']
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))
design = MS(['lstat'])
print(design)
design = design.fit(Boston)
print(design)
X = design.transform(Boston)
print(X)

new_df = pd.DataFrame({'lstat':[5, 10, 15]})
newX = design.transform(new_df)

print(newX)

new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print( new_predictions.conf_int(alpha=0.05) )
ax = Boston.plot.scatter('lstat', 'medv')
d1.abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3
       )

plt.show()
