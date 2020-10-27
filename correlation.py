import os

import pandas

# Array that contains the headers for the class columns
ARR = ["speed-20", "speed-30", "speed-50", "speed-60", "speed-70", "left-t", "right-t", "pedestrian", "children",
       "cycle"]


# initialise full data correlation
def init_correlation():
    # Read training data and associated labels
    data = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")

    # Loop through all the classifier files whilst concatenating to the training data
    for x in range(0, 10):
        labels = pandas.read_csv("randomized_data/y_train_smpl_" + str(x) + "_randomized.csv")
        labels = labels.rename(columns={"0": ARR[x]})
        data = pandas.concat([data, labels], axis=1)

    # print full data table with all the classifier files were concatenated
    print(data)
    # Get the absolute correlation values of the data
    correlation = abs(data.corr())
    return correlation, data


# run correlation on individual classes to find the best number of pixels
def runCorrelation(pixels):
    if os.path.exists("top_pixels"):
        print("Directory already exists")
    else:
        os.mkdir("top_pixels")

    allPixel = []
    # loop through all ten classifier columns to find the best 10 pixels and print the arrays containing them.
    for x in range(0, 10):
        sortedCorr = corr[ARR[x]][0:2304].sort_values(ascending=False)
        sortedCorr = sortedCorr[0:pixels].index
        pixARR = sortedCorr.values.tolist()
        print("#### " + ARR[x] + " ####")
        pandas.array(sortedCorr)
        sortedPix = []
        print(pandas.array(sortedCorr))
        for i in pixARR:
            item = int(i)
            sortedPix.append(item)

        sortedPix = sorted(sortedPix)
        for i in sortedPix:
            allPixel.append(i)

    print(allPixel)
    print(len(allPixel))
    z = data[[str(boi) for boi in allPixel]]
    z.to_csv("top_pixels/top" + str(pixels) + "pixels.csv", index=False)
    print(z)


# start
corr, data = init_correlation()
print("#### Top 5 Pixels ####")
runCorrelation(5)

print("#### Top 10 Pixels ####")
runCorrelation(10)

print("#### Top 20 Pixels ####")
runCorrelation(20)
