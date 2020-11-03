"""import pandas
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
# https://towardsdatascience.com/k-means-and-pca-for-image-clustering-a-visual-analysis-8e10d4abba40

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


file = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")

pca_2d = PCA(n_components=2)

print("Dimension of our data after PCA = " + str(file.shape))

for i in range(0, 10):
    classifier = pandas.read_csv("randomized_data/y_train_smpl_"+str(i)+"_randomized.csv")
    k = KMeans(n_clusters=2, init="random", n_init=1, algorithm="full", max_iter=5, random_state=1).fit(file)
    print(k.labels_)
    print("Accuracy of file number " + str(i) + " = " + str(accuracy_score(k.labels_, classifier)))
    print(k.inertia_)
    plt.plot(k[:, 0], k[:, 1], 'k.', markersize=2)


"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans
import pandas
# import the data
x = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
y = pandas.read_csv("randomized_data/y_train_smpl_0_randomized.csv")
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', "principal component 2"])

finalDf = pandas.concat([principalDf, y], axis=1)

k = KMeans(n_clusters=2, init="random", n_init=1, algorithm="full", max_iter=5, random_state=1).fit(principalDf)
print(y.values)
print(k.labels_)

accuracy = sklearn.metrics.accuracy_score(k.labels_,y)
import matplotlib.pyplot as plot

for entry in principalDf.values:
    print(entry)



"""
X = np.array([[1, 2, 3, 4], [1, 4, 2, 1], [1, 0, 1,1],[10, 2,3 ,5], [10, 4,6,7], [10, 0,8,9]])
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
print(kmeans.labels_)

kmeans.predict([[0, 0,1,1], [12, 3,2,3]])

print(kmeans.score(X))
print()

"""
