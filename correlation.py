import os

import pandas

# Array that contains the headers for the class columns
ARR = ["speed-20", "speed-30", "speed-50", "speed-60", "speed-70", "left-t", "right-t", "pedestrian", "children",
       "cycle"]


# initialise full data correlation
def init_correlation():
    """
    This is used to initialise the correlation of the main dataset. To correlate the data faster and only once
    all the classes have been appended to the main dataset.

    :return correlation : this is the correlated data\n
    :return data : uncorrelated data
    """

    # Read training data and associated labels
    data = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")

    # Loop through all the classifier files whilst concatenating to the training data
    for x in range(0, 10):
        labels = pandas.read_csv("randomized_data/y_train_smpl_" + str(x) + "_randomized.csv")
        labels = labels.rename(columns={"0": ARR[x]})
        data = pandas.concat([data, labels], axis=1)

    # print full data table with all the classifier files were concatenated
    print("Current Data\n")
    print(data)
    # Get the absolute correlation values of the data
    print("Correlating Data .......")
    correlation = abs(data.corr())
    print("Correlation Data\n")
    print(correlation)
    return correlation, data


# run correlation on individual classes to find the best number of pixels
def runCorrelation(pixels, corr, data):
    """
    Outputs the Top five pixels as two different files, one for python reuse and another for weka use.
    :param pixels: number of top pixels for each classifier
    :param corr: the correlated data
    :param data: the uncorrelated data

    """
    if os.path.exists("top_pixels"):
        print("Directory already exists")
    else:
        os.mkdir("top_pixels")

    allPixel = []
    # loop through all ten classifier columns to find the best 10 pixels and print the arrays containing them.
    for x in range(0, 10):
        sortedCorr = corr[ARR[x]][0:2304].sort_values(ascending=False)
        # Get the pixel numbers
        sortedCorr = sortedCorr[0:pixels].index
        pixARR = sortedCorr.values.tolist()
        print("#### Top Pixels For" + ARR[x] + " ####")
        pandas.array(sortedCorr)
        sortedPix = []
        # print the pixel numbers
        print(pandas.array(sortedCorr))
        for i in pixARR:
            item = int(i)
            sortedPix.append(item)

        # Sort the top pixels in ascending order.
        sortedPix = sorted(sortedPix)
        # append all the pixels together.
        for i in sortedPix:
            allPixel.append(i)

    print(allPixel)
    print(len(allPixel))
    # Get all the pixel data for the top pixels generated
    z = data[[str(i) for i in allPixel]]
    # Generate top pixel files for weka and python
    z.to_csv("top_pixels/top" + str(pixels) + "pixels.csv", index=False)
    z.to_csv("top_pixels/top" + str(pixels) + "pixels_weka.csv", index=False, header=[str(allPixel[i]) + "-" + str(i) for i in range(len(allPixel))])
    print(z)

