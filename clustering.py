import math
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cluster_initializer(file):
    """
    Initialise the cluster files using PCA
    :param file: input file for the
    :return: principleDf, unaltered data, labels - classifiers, full_pca_file - PCA(data) with classifiers.
    """
    x = pandas.read_csv(file)
    unaltered = x
    labels = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")
    x = StandardScaler().fit_transform(x)
    # reduce dataset to 2 points - x and y values.
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    # create a dataframe
    pca_file = pandas.DataFrame(data=principalComponents, columns=['principal component 1', "principal component 2"])
    full_pca_file = pandas.concat([pca_file, labels], axis=1)
    return pca_file, unaltered, labels, full_pca_file


def k_means_cluster(labels, pca_file, optimal, clusters):
    """
    outputs a KMeans cluster scatter plot
    :param clusters: number of clusters
    :param labels: classifiers
    :param pca_file: reduced data set
    :param optimal: takes in boolean and outputs optimal number of clusters.
    """
    print("KMeans Starting ......")
    Sum_of_squared_distances = []
    K = range(2, 16)
    if optimal:
        for k in K:
            km = KMeans(n_clusters=k, random_state=1).fit(pca_file)
            Sum_of_squared_distances.append(math.floor(km.inertia_))

        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal KMeans k-clusters')
        plt.show()

    # KMeans clustering on the reduced dataset
    k = KMeans(n_clusters=clusters, init="random", n_init=1, max_iter=300, random_state=1).fit(pca_file)
    print(k.labels_)
    # accuracy between cluster labels and the original dataset classifiers.
    accuracy = sklearn.metrics.accuracy_score(k.labels_, labels)
    print(accuracy)
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]

    # plots the scatter plot.
    for entry, oh in zip(pca_file.values, k.labels_):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh], label=colors[oh])

    plt.suptitle("KMeans Cluster")
    plt.show()

