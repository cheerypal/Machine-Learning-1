import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# import the data
x = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
y = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pandas.DataFrame(data=principalComponents, columns = ['principal component 1', "principal component 2"])

finalDf = pandas.concat([principalDf, y], axis=1)

k = KMeans(n_clusters=10, init="random", n_init=1, algorithm="full", max_iter=5, random_state=1).fit(principalDf)
print(y.values)
print(k.labels_)

accuracy = sklearn.metrics.accuracy_score(k.labels_, y)


for entry, oh in zip(principalDf.values, k.labels_):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    plt.scatter(entry[0], entry[1], s=1, c=colors[oh], label=colors[oh])


#plt.legend()
plt.show()
