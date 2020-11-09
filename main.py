import data_preparation
import naive_bayes
import correlation
import clustering
import hs_clustering

# datasets
rand_test_data = "randomized_data/x_train_gr_smpl_randomized.csv"
top5Pixels = "top_pixels/top5pixels.csv"
top10Pixels = "top_pixels/top10pixels.csv"
top20Pixels = "top_pixels/top20pixels.csv"
rand_label_file = "randomized_data/y_train_smpl_randomized.csv"

# Randomize training training_data - q1.
data_preparation.randomize_data()

# naive_bayes - q3
print("\n### Naive Bayes rand_test_data ###\n")
naive_bayes.naive_bayes_fun(rand_test_data)

# correlation - q4 - q5
print("\n### Correlated data ###\n")
corr, unCorrData = correlation.init_correlation()
print("#### Top 5 Pixels ####")
correlation.runCorrelation(5, corr, unCorrData)
print("#### Top 10 Pixels ####")
correlation.runCorrelation(10, corr, unCorrData)
print("#### Top 20 Pixels ####")
correlation.runCorrelation(20, corr, unCorrData)

# q5 - naive bayes on on top 5,10,20 pixels
print("\n### Naive Bayes on top 5,10,20 pixel datasets ###\n")
print("\n#### Top 5 Pixels NB ####\n")
naive_bayes.naive_bayes_fun(top5Pixels)
print("\n#### Top 10 Pixels NB ####\n")
naive_bayes.naive_bayes_fun(top10Pixels)
print("\n#### Top 20 Pixels NB ####\n")
naive_bayes.naive_bayes_fun(top20Pixels)


# q7 - Generate weka ready files.
print("\n### Add classifier to the top pixel files - these files can be used in weka ###\n")
print("\n#### Top 5 Pixels Weka File ####\n")
data_preparation.addClassifier(5)
print("\n#### Top 10 Pixels Weka File ####\n")
data_preparation.addClassifier(10)
print("\n#### Top 20 Pixels Weka File ####\n")
data_preparation.addClassifier(20)

print("\n### Clustering section - the following section can take up to 20 minutes ###\n")
# initialize cluster section using pca
print("\n### Create PCA reduced files ###\n")
# This will take in a file - rand_test_data, top5Pixels, top10Pixels, top20Pixels
pca_file, unaltered, labels, full_pca_file = clustering.cluster_initializer(rand_test_data)

# q9 - KMeans cluster with optimal cluster finder
print("\n### KMeans Clusters with KMeans with Optimal Cluster Finder ###\n")
# The file used in the cluster methods are the resulting file from the function above
# The number of optimal cluster is given in the cluster functions
clustering.k_means_cluster(labels, pca_file, True, 4)


# q11 - Cluster Algorithms Used
print("\n### Hard and Soft Clustering Techniques ###\n")
hs_clustering.aggloCluster(labels, full_pca_file, 4)
# EM cluster with optimal clustering
hs_clustering.EMCluster(labels, pca_file, True, 4)
hs_clustering.gaussianCluster(labels, pca_file, 4)
hs_clustering.birchCluster(labels, pca_file, 4)

