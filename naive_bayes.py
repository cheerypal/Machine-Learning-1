# Import train_test_split function
import numpy
import pandas
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing


def naive_bayes_fun(pixelFile):

    # Read training data and associated labels
    data = pandas.read_csv(pixelFile)

    labels = pandas.read_csv("randomized_data/y_train_smpl_randomized.csv")

    # Change the shape of labels to a 1d array, since it is a column-vector
    labels = numpy.ravel(labels)

    # 30/70 split testing/training
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=109)

    # Declare Multinomial Naive Bayes model
    naive_bayes = MultinomialNB()

    # Train the model using the training sets
    print("Training Naive Bayes classifier.")
    naive_bayes.fit(X_train, y_train)

    # Predict the response for test dataset
    y_prediction = naive_bayes.predict(X_test)

    # Model Accuracy - how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))

    # Confusion Matrix
    print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_prediction))


