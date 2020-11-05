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
    return principalDf, finalDf, unaltered, labels


def k_means_cluster(labels, principalDf):
    k = KMeans(n_clusters=2, init="random", n_init=1, algorithm="full", max_iter=300, random_state=1).fit(principalDf)
    print(k.labels_)

    accuracy = sklearn.metrics.accuracy_score(k.labels_, labels)
    print(accuracy)

    for entry, oh in zip(principalDf.values, k.labels_):
        # colors = [blue, orange, green, red, purple, brown, pink, grey, yellow, cyan]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
        plt.scatter(entry[0], entry[1], s=1, c=colors[oh], label=colors[oh])

    plt.suptitle("KMeans Cluster")
    # plt.legend()
    plt.show()

