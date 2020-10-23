import numpy
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
"""
from sklearn import preprocessing

# Read training data and associated labels
data = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
labels = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")

data = preprocessing.normalize(data)
"""

# importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

# Loading the dataset
df = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
a = []
for i in range(2304):
    a.append(str(i))
X = df[a]

df2 = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
y = df2["0"]



from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
lr = MultinomialNB()
print("im on this bit now")
rfe = RFE(estimator=lr, n_features_to_select=200, step=1)
rfe.fit(X, y)

print(rfe.ranking_)

for i in range(len(rfe.ranking_)):
    if rfe.ranking_[i] == 1:
        print(a[i])

