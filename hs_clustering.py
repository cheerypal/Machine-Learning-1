import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

# import the data
x = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
ooh = x
y = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
x = StandardScaler().fit_transform(x)


def pcaLimiting(data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', "principal component 2"])
    finalDf = pandas.concat([principalDf, y], axis=1)
    return principalDf, finalDf

#############################################################


def aggloCluster(data, principalDf):
    globb = AgglomerativeClustering(n_clusters=10).fit(data)
    print("globb!")
    print(globb.labels_)
    print(accuracy_score(globb.labels_, y))

    for entry, oh in zip(principalDf.values, globb.labels_):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.show()


#############################################################


def gaussianCluster(data, principalDf):
    print("Gaussian is starting ....")
    gaus = GaussianMixture(n_components=10).fit_predict(data)
    print(gaus)
    print(accuracy_score(gaus, y))

    for entry, oh in zip(principalDf.values, gaus):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.show()

#############################################################


def EMCluster(data, principalDf):
    print("EM starting .....")
    k = KMeans(n_clusters=10, algorithm="full", random_state=1).fit(data)
    print(k.labels_)
    print(accuracy_score(k.labels_,y))

    for entry, oh in zip(principalDf.values, k.labels_):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.show()

###########################################################


def birchCluster(data, principalDf):
    print("Birch starting .....")
    brch = Birch(n_clusters=10).fit(data)
    print(brch.labels_)
    print(brch.subcluster_labels_)
    print(accuracy_score(brch.labels_, y))

    for entry, oh in zip(principalDf.values, brch.labels_):
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.show()


