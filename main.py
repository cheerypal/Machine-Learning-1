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
print("\n### Naive Bayes on top 5,10,20 pixel datasets\n")
naive_bayes.naive_bayes_fun(top5Pixels)
naive_bayes.naive_bayes_fun(top10Pixels)
naive_bayes.naive_bayes_fun(top20Pixels)


# q7
print("\n### Add classifier to the top pixel files ###\n")
data_preparation.addClassifier(5)
data_preparation.addClassifier(10)
data_preparation.addClassifier(20)


# initialize cluster section using pca
principalDf, unaltered, labels, finalDf = clustering.cluster_initializer(rand_test_data)

# q9
clustering.k_means_cluster(labels, principalDf, True)

# q11
hs_clustering.aggloCluster(labels, finalDf)
hs_clustering.EMCluster(labels, principalDf, True)
hs_clustering.gaussianCluster(labels, principalDf)
hs_clustering.birchCluster(labels, principalDf)

