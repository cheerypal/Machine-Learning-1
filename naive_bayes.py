import sklearn as sk, numpy, pandas as pd
import os, csv, math

from sklearn.naive_bayes import MultinomialNB

x_set = pd.read_csv("randomized_data/x_train_gr_smpl_randomized.csv", header=None)
s = pd.DataFrame(x_set)
length = len(s.index)

# training_X_Set = x_set[1:math.floor(0.7 * length)+1].astype("float")/255
# = x_set[math.floor(0.7 * length)+1:].astype("float")/255
training_X_Set = x_set[1:math.floor(0.7 * length) + 1]
test_X_Set = x_set[math.floor(0.7 * length) + 1:]
# print(training_X_Set)
# print(test_X_Set)


y_set = pd.read_csv("randomized_data/y_train_smpl_randomized.csv", header=None)
training_Y_Set = y_set[1:math.floor(0.7 * length) + 1]
test_Y_Set = y_set[math.floor(0.7 * length) + 1:]
# print(training_Y_Set)
# print(test_Y_Set)

print(training_Y_Set[0])

naive_bayes_X_set = MultinomialNB()
naive_bayes_X_set.fit(training_X_Set, training_Y_Set[0])

print(naive_bayes_X_set.predict(test_X_Set))
print(naive_bayes_X_set.score(test_X_Set, test_Y_Set))
