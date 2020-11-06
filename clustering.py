import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# import the data
def cluster_initializer(file):
    x = pandas.read_csv(file)
    unaltered = x
    labels = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pandas.DataFrame(data=principalComponents, columns=['principal component 1', "principal component 2"])
    finalDf = pandas.concat([principalDf, labels], axis=1)
    return principalDf, unaltered, labels, finalDf


def k_means_cluster(labels, principalDf, optimal):

    print("KMeans Starting ......")
    Sum_of_squared_distances = []
    K = range(2, 16)
    if optimal:
        for k in K:
            km = KMeans(n_clusters=k, random_state=1).fit(principalDf)
            Sum_of_squared_distances.append(math.floor(km.inertia_))

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal KMeans k-clusters')
        plt.show()

    k = KMeans(n_clusters=10, init="random", n_init=1, max_iter=300, random_state=1).fit(principalDf)
    print(k.labels_)

    accuracy = sklearn.metrics.accuracy_score(k.labels_, labels)
    print(accuracy)
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    for entry, oh in zip(principalDf.values, k.labels_):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh], label=colors[oh])

    plt.suptitle("KMeans Cluster")
    plt.show()

