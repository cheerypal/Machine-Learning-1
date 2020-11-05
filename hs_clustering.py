import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import clustering


rand_test_data = "randomized_data/x_train_gr_smpl_randomized.csv"
rand_label_file = "randomized_data/y_train_smpl_randomized.csv"
top5Pixels = "top_pixels/top5pixels.csv"
top10Pixels = "top_pixels/top10pixels.csv"
top20Pixels = "top_pixels/top20pixels.csv"


#############################################################
def aggloCluster(labels, principalDf):
    print("Agglomerative is starting ....")
    globb = AgglomerativeClustering(n_clusters=10).fit(principalDf)
    print("globb!")
    print(globb.labels_)
    print(accuracy_score(globb.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    for entry, oh in zip(principalDf.values, globb.labels_):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.suptitle("Agglomerative Cluster")
    plt.gca().legend(colors)
    plt.show()


#############################################################


def gaussianCluster(labels, principalDf):
    print("Gaussian is starting ....")
    gaus = GaussianMixture(n_components=10).fit_predict(principalDf)
    print(gaus)
    print(accuracy_score(gaus, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    for entry, oh in zip(principalDf.values, gaus):
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh])

    plt.suptitle("Gaussian Mixture Cluster")
    plt.gca().legend(colors)
    plt.show()

#############################################################


def EMCluster(labels, principalDf):
    print("EM starting .....")
    k = KMeans(n_clusters=10, algorithm="full", random_state=1).fit(principalDf)
    print(k.labels_)
    print(accuracy_score(k.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    for entry, oh in zip(principalDf.values, k.labels_):
        plt.scatter(entry[0], entry[1], s=2, c=colors[oh])

    plt.suptitle("EM Cluster")
    plt.gca().legend(colors)
    plt.show()

###########################################################


def birchCluster(labels, principalDf):
    print("Birch starting .....")
    brch = Birch(n_clusters=10).fit(principalDf)
    print(brch.labels_)
    print(brch.subcluster_labels_)
    print(accuracy_score(brch.labels_, labels))
    colors = ["blue", "orange", "green", "red", "purple",
              "brown", "pink", "grey", "yellow", "cyan"]
    for entry, oh in zip(principalDf.values, brch.labels_):
        plt.scatter(entry[0], entry[1], s=2, c=colors[oh])

    plt.suptitle("Birch Cluster")
    plt.gca().legend(colors)
    plt.show()


principalDf, unaltered, labels = clustering.cluster_initializer(top5Pixels)

birchCluster(labels, principalDf)

