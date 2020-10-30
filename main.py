import data_preparation
import naive_bayes
import correlation

# datasets
rand_test_data = "randomized_data/x_train_gr_smpl_randomized.csv"
top5Pixels = "top_pixels/top5pixels.csv"
top10Pixels = "top_pixels/top10pixels.csv"
top20Pixels = "top_pixels/top20pixels.csv"

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



