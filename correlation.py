
import pandas

# Read training data and associated labels
data = pandas.read_csv("randomized_data/x_train_gr_smpl_randomized.csv")
ARR = ["speed-20", "speed-30", "speed-50", "speed-60", "speed-70", "left-t", "right-t", "pedestrian", "children",
       "cycle"]
for x in range(0, 10):
    labels = pandas.read_csv("randomized_data/y_train_smpl_" + str(x) + "_randomized.csv")
    labels = labels.rename(columns={"0": ARR[x]})
    data = pandas.concat([data, labels], axis=1)

print(data)

corr = abs(data.corr())

for x in range(0, 10):

    sortedCorr = corr[ARR[x]][0:2304].sort_values(ascending=False)
    sortedCorr = sortedCorr[0:10].index
    print("#### " + ARR[x] + " ####")
    print(pandas.array(sortedCorr))
