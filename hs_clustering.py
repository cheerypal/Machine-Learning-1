import math

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import clustering

# Files for use if testing is required in the file
rand_test_data = "randomized_data/x_train_gr_smpl_randomized.csv"
rand_label_file = "randomized_data/y_train_smpl_randomized.csv"
top5Pixels = "top_pixels/top5pixels.csv"
top10Pixels = "top_pixels/top10pixels.csv"
top20Pixels = "top_pixels/top20pixels.csv"


#############################################################


def aggloCluster(labels, pca_file, clusters):
    """
    Function that outputs a scatter plot of the Agglomerative Cluster

    params:
        clusters : number of clusters
        labels : file
            The classifier for the dataset
        pca_file : file
            The dataset
    """
    print("Agglomerative is starting ....")
    agglo = AgglomerativeClustering(n_clusters=clusters).fit(pca_file)
    print("globb!")
    # agglo cluster labels.
    print(agglo.labels_)
    print(accuracy_score(agglo.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    # populate the scatter plot
    for entry, oh in zip(pca_file.values, agglo.labels_):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.suptitle("Agglomerative Cluster")
    plt.show()


#############################################################


def gaussianCluster(labels, pca_file, clusters):
    """
       Function that outputs a scatter plot of the Gaussian Mixture Cluster

       params:
        clusters : number of clusters
        labels : file
            The classifier for the dataset
        pca_file : file
            The dataset
    """
    print("Gaussian is starting ....")
    gaus = GaussianMixture(n_components=clusters).fit_predict(pca_file)
    # Print the cluster labels
    print(gaus)
    print(accuracy_score(gaus, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    # Plot the scatter plot.
    for entry, oh in zip(pca_file.values, gaus):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.suptitle("Gaussian Mixture Cluster")
    plt.show()


#############################################################


def EMCluster(labels, pca_file, optimal, clusters):
    """
           Function that outputs a scatter plot of the Gaussian Mixture Cluster

           params:
                clusters: number of clusters
                labels : file
                    The classifier for the dataset
                pca_file : file
                    The dataset
                optimal : bool :
                    Find the optimal number of clusters for EM.
    """

    print("EM starting .....")
    # List of inertia's
    Sum_of_squared_distances = []
    # Range of clusters
    K = range(2, 16)
    # Find the optimal number of clusters
    if optimal:
        for k in K:
            km = KMeans(n_clusters=k, algorithm="full", random_state=1).fit(pca_file)
            Sum_of_squared_distances.append(math.floor(km.inertia_))

        # Plot the elbow curve for optimal clusters.
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal EM k-clusters')
        plt.show()

    # Run the KMeans cluster algorithm using 10 clusters.
    km = KMeans(n_clusters=clusters, algorithm="full", random_state=1).fit(pca_file)
    # Print the cluster labels
    print(km.labels_)
    # Print the accuracy of cluster labels compared to the classifier labels.
    print(accuracy_score(km.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    # Plot the scatter plot
    for entry, oh in zip(pca_file.values, km.labels_):
        plt.scatter(entry[0], entry[1], s=2, c=colors[oh])

    plt.suptitle("EM Cluster")
    plt.show()

###########################################################


def birchCluster(labels, pca_file, clusters):
    """
       Function that outputs a scatter plot of the Gaussian Mixture Cluster

       params:
            clusters: number of clusters
            labels : file
                The classifier for the dataset
            pca_file : file
                The dataset
    """

    print("Birch starting .....")
    # Run the birch cluster algo
    brch = Birch(n_clusters=clusters).fit(pca_file)
    # Print the birch cluster labels.
    print(brch.labels_)
    # Print the birch sub-cluster labels.
    print(brch.subcluster_labels_)
    # Print the accuracy of cluster labels compared to the classifier labels.
    print(accuracy_score(brch.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    # Plot the scatter plot
    for entry, oh in zip(pca_file.values, brch.labels_):
        plt.scatter(entry[0], entry[1], s=2, c=colors[oh])

    plt.suptitle("Birch Cluster")
    plt.show()


"""
# Used for internal testing
pca_file, unaltered, labels, full_pca_file = clustering.cluster_initializer(rand_test_data)

birchCluster(labels, pca_file)
aggloCluster(labels, full_pca_file)
EMCluster(labels, pca_file, True)
gaussianCluster(labels, pca_file)
"""



